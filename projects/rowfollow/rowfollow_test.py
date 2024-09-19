import json
import os
import sys
from dataclasses import dataclass
from typing import Optional

import mlflow
import pandas as pd
import torch
import torch.utils.data

import maml_api
import maml_eval
import maml_logging

import rowfollow_task
import rowfollow_utils


class RowfollowValDataset(torch.utils.data.Dataset):
    def __init__(self, validation_collection_path: str, validation_annotations_file_path: str, device: torch.device):
        self.validation_collection_path: str = validation_collection_path
        self.validation_annotations_file_path: str = validation_annotations_file_path

        self.annotations_df: pd.DataFrame = pd.read_csv(self.validation_annotations_file_path)
        self.annotations_df = self._filter_existing_images()
        self.sigma = 10
        self.device = device

    def _filter_existing_images(self):
        # Create a list of available image paths from the directories
        available_image_names = []
        for root, _, files in os.walk(self.validation_collection_path):
            for file in files:
                if file.endswith(('jpg', 'jpeg', 'png')):  # Adjust extensions as needed
                    available_image_names.append(file)

        # Convert available image paths to a set for faster lookup
        available_image_names = set(available_image_names)

        # Filter the annotations to keep only those with available image files
        filtered_annotations = self.annotations_df[self.annotations_df['image_name'].isin(available_image_names)]
        return filtered_annotations.reset_index(drop=True)

    def __len__(self):
        return len(self.annotations_df)

    def __getitem__(self, idx):
        annotation_row = self.annotations_df.iloc[idx]
        image_name = annotation_row['image_name']
        collection_name = image_name.split('_cam')[0]

        image_path = os.path.join(self.validation_collection_path, image_name)

        vp, ll, lr = rowfollow_task.RowfollowTaskOldDataset.get_kps_for_image(image_name, annotation_row=annotation_row)

        pre_processed_image, _ = rowfollow_utils.pre_process_image_old_data(image_path, new_size=(320, 224))
        pre_processed_image = torch.from_numpy(pre_processed_image)

        # vp, ll, lr are coordinates, but we need distributions
        vp_gt = rowfollow_utils.dist_from_keypoint(vp, sig=self.sigma, downscale=4)
        ll_gt = rowfollow_utils.dist_from_keypoint(ll, sig=self.sigma, downscale=4)
        lr_gt = rowfollow_utils.dist_from_keypoint(lr, sig=self.sigma, downscale=4)

        return pre_processed_image.to(self.device), torch.stack([vp_gt, ll_gt, lr_gt]).to(self.device)


MLFLOW_CACHE_DIR = 'mlflow-cache/'


def get_local_artifact_path(run_id: str, episode: int):
    return os.path.join(MLFLOW_CACHE_DIR, 'mlruns', run_id, f'ep{episode}')


def load_model(run_id: str, episode: int) -> maml_api.MamlModel:
    model_uri = 'runs:/{}/{}/{}'.format(run_id, 'ep{}'.format(episode), 'model')
    local_path = get_local_artifact_path(run_id, episode)

    try:
        os.makedirs(local_path, exist_ok=True)
        print('Trying to load cached model...')
        return mlflow.pytorch.load_model(os.path.join(local_path, 'model'))

    except (OSError, mlflow.exceptions.MlflowException) as e:
        print('No cached model. Downloading model...')
        return mlflow.pytorch.load_model(model_uri, local_path)


def load_inner_lrs(run_id: str, episode: int) -> maml_api.InnerLrs:
    artifact_uri = 'runs:/{}/{}/{}'.format(run_id, 'ep{}'.format(episode), 'inner_lrs.json')
    return mlflow.artifacts.load_dict(artifact_uri)


def load_inner_buffers(run_id: str, episode: int) -> maml_api.InnerBuffers:
    artifact_uri = 'runs:/{}/{}/{}'.format(run_id, 'ep{}'.format(episode), 'inner_buffers.json')
    inner_buffers = mlflow.artifacts.load_dict(artifact_uri)

    # at this point the inner_buffers do not contain tensors, but only a list -> this needs to be transformed
    for i, named_buffers in inner_buffers.items():
        for n, b in named_buffers.items():
            inner_buffers[i][n] = torch.tensor(b)
    return inner_buffers


def test_main(run_id: str,
              episode: int,
              k: int,
              inner_steps: int,
              support_collection_path: str,
              support_annotations_file_path: str,
              target_collection_path: str,
              target_annotations_file_path: str,
              device: torch.device,
              seed: Optional[int],
              use_mlflow: bool):

    model = load_model(run_id, episode)
    inner_lrs = load_inner_lrs(run_id, episode)
    inner_buffers = load_inner_buffers(run_id, episode)

    task = rowfollow_task.RowfollowTaskOldDataset(support_annotations_file_path,
                                                  support_collection_path,
                                                  k,
                                                  device,
                                                  seed=seed)

    finetuner = maml_eval.MamlFinetuner(model, inner_lrs, inner_buffers, inner_steps, task, use_mlflow)
    finetuner.finetune()  # TODO add do_finetune parameter to train_config

    if use_mlflow:
        print('Uploading model to mlflow server...')
        example_x = task.sample(maml_api.SetToSetType.SUPPORT)[0].cpu().numpy()
        mlflow.pytorch.log_model(model, 'finetuned_model', input_example=example_x)

    model.eval()
    val_dataset = RowfollowValDataset(target_collection_path,
                                      target_annotations_file_path,
                                      device)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False)
    total_loss = 0.0
    batches_processed = 0
    for x, y in val_dataloader:
        y_hat = model(x)
        total_loss += task.calc_loss(y_hat, y, maml_api.Stage.VAL, maml_api.SetToSetType.TARGET).item()
        batches_processed += 1
        print('total_loss:', total_loss)
        print('avg_loss:', total_loss / batches_processed)

        if use_mlflow:
            mlflow.log_metric('total_loss', total_loss, batches_processed)
            mlflow.log_metric('avg_loss', total_loss / batches_processed, batches_processed)

    if use_mlflow:
        mlflow.log_metric('final_total_loss', total_loss)
        mlflow.log_metric('final_avg_loss', total_loss / batches_processed)

    print('total loss: {}'.format(total_loss))


def calc_val_loss_for_train(current_episode: int,
                            model: maml_api.MamlModel,
                            inner_buffers: maml_api.InnerBuffers,
                            inner_lrs: maml_api.InnerLrs,
                            k: int,
                            inner_steps: int,
                            support_collection_path: str,
                            support_annotations_file_path: str,
                            device: torch.device,
                            seed: Optional[int],
                            use_mlflow: bool,
                            logger: maml_logging.Logger):
    task = rowfollow_task.RowfollowTaskOldDataset(support_annotations_file_path,
                                                  support_collection_path,
                                                  k,
                                                  device,
                                                  seed=seed)

    finetuner = maml_eval.MamlFinetuner(model, inner_lrs, inner_buffers, inner_steps, task, use_mlflow=False)
    finetuner.finetune()

    model.eval()
    val_dataset = RowfollowValDataset(support_collection_path,
                                      support_annotations_file_path, device=device)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False)
    total_loss = 0.0
    batches_processed = 0
    for x, y in val_dataloader:
        y_hat = model(x)
        total_loss += task.calc_loss(y_hat, y, maml_api.Stage.VAL, maml_api.SetToSetType.TARGET).item()
        batches_processed += 1

    if use_mlflow:
        collection_name = support_collection_path.split('/')[-1]
        logger.log_metric(f'{collection_name}_val_loss', total_loss / batches_processed, step=current_episode)


@dataclass
class TestConfig:
    run_id: str
    episode: int
    k: int
    inner_steps: int
    base_path: str
    support_collection: str
    support_annotations_file_path: str
    seed: Optional[int]
    device: str
    target_collection: Optional[str] = None
    target_annotations_file_path: Optional[str] = None
    # only asl-then-0.5 can be used atm
    lr_strategy: str = 'asl-then-0.5'  # possible values in the future: asl-then-0.5, 0.5-then-0.5, asl-then-anneal
    dataset_info_path: Optional[str] = None
    use_mlflow: bool = False
    mlflow_experiment: Optional[str] = None


def run_main_from_test_config(test_config: TestConfig):
    support_collection_path = os.path.join(test_config.base_path, test_config.support_collection)
    support_annotations_file_path = os.path.join(test_config.base_path, test_config.support_annotations_file_path)
    if test_config.target_collection is None:
        target_collection_path = support_collection_path
    else:
        target_collection_path = os.path.join(test_config.base_path, test_config.target_collection)

    if test_config.target_annotations_file_path is None:
        target_annotations_file_path = support_annotations_file_path
    else:
        target_annotations_file_path = os.path.join(test_config.base_path, test_config.target_annotations_file_path)

    if test_config.use_mlflow:
        print('Setting up mlflow run...')
        if test_config.mlflow_experiment is None:
            mlflow.set_experiment(f'/run({test_config.run_id})-val')
        else:
            mlflow.set_experiment(test_config.mlflow_experiment)
        mlflow.start_run()

        mlflow.log_params(vars(test_config))
        mlflow.log_dict(vars(test_config), 'test_config.json')
        data_info = {
            'support_collection': support_collection_path,
            'target_collection': target_collection_path
        }
        mlflow.log_dict(data_info, 'data_info.json')
        print('Finished setting up mlflow run.')

    test_main(run_id=test_config.run_id,
              episode=test_config.episode,
              k=test_config.k,
              inner_steps=test_config.inner_steps,
              support_collection_path=support_collection_path,
              support_annotations_file_path=os.path.join(test_config.base_path,
                                                         test_config.support_annotations_file_path),
              target_collection_path=target_collection_path,
              target_annotations_file_path=target_annotations_file_path,
              device=torch.device(test_config.device),
              seed=test_config.seed,
              use_mlflow=test_config.use_mlflow)

    mlflow.end_run()


def get_config_from_file(path: str) -> TestConfig:
    with open(path, 'r') as f:
        config_dict = json.load(f)
        test_config = TestConfig(**config_dict)
        test_config.dataset_info_path = os.path.join(test_config.base_path, 'dataset_info.csv')
        return test_config


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] == '-h':
        print('USAGE: python rowfollow_test.py path/to/test_config.json')
        sys.exit(1)

    if not os.path.exists(sys.argv[1]):
        print(f'Path {sys.argv[1]} does not exist')
        sys.exit(1)

    path_to_config = sys.argv[1]
    run_main_from_test_config(get_config_from_file(path_to_config))
