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

import rowfollow_model
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


@dataclass
class TestConfig:
    run_id: str
    episode: int
    k: int
    inner_steps: int
    base_path: str
    support_collection_path: str
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
    sigma: int = 10
    path_to_ckpt_file: Optional[str] = None


def test_main(config: TestConfig):
    if config.path_to_ckpt_file is None:
        model = load_model(config.run_id, config.episode)
        inner_lrs = load_inner_lrs(config.run_id, config.episode)
        inner_buffers = load_inner_buffers(config.run_id, config.episode)

        calc_val_loss_for_train(current_episode=-1, model=model, inner_lrs=inner_lrs, inner_buffers=inner_buffers,
                                k=config.k, inner_steps=config.inner_steps,
                                support_collection_path=config.support_collection_path,
                                support_annotations_file_path=config.support_annotations_file_path,
                                device=torch.device(config.device), seed=config.seed, use_mlflow=False,
                                logger=None, sigma=config.sigma)

    else:
        model = get_model_from_ckpt_file(config.path_to_ckpt_file)
        model.eval()
        task = rowfollow_task.RowfollowTaskOldDataset(config.support_annotations_file_path,
                                                      config.support_collection_path,
                                                      config.k,
                                                      torch.device(config.device),
                                                      seed=config.seed,
                                                      sigma=config.sigma)

        calc_loss(config.support_collection_path, config.support_annotations_file_path, torch.device(config.device),
                  model, task, None, False, -1)


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
                            logger: maml_logging.Logger,
                            sigma: int):
    task = rowfollow_task.RowfollowTaskOldDataset(support_annotations_file_path,
                                                  support_collection_path,
                                                  k,
                                                  device,
                                                  seed=seed,
                                                  sigma=sigma)

    finetuner = maml_eval.MamlFinetuner(model, inner_lrs, inner_buffers, inner_steps, task, use_mlflow=False)
    finetuner.finetune()

    model.eval()

    calc_loss(target_collection_path=support_collection_path,
              target_annotations_file_path=support_annotations_file_path,
              device=device,
              model=model,
              task=task,
              logger=logger,
              use_mlflow=use_mlflow,
              current_episode=current_episode)


def calc_loss(target_collection_path: str, target_annotations_file_path: str, device: torch.device,
              model: maml_api.MamlModel, task: maml_api.MamlTask,
              logger: maml_logging.Logger, use_mlflow: bool, current_episode: int):
    val_dataset = RowfollowValDataset(target_collection_path,
                                      target_annotations_file_path, device=device)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False)
    total_loss = 0.0
    batches_processed = 0
    for x, y in val_dataloader:
        y_hat = model(x)
        total_loss += task.calc_loss(y_hat, y, maml_api.Stage.VAL, maml_api.SetToSetType.TARGET).item()
        batches_processed += 1
        print('avg_loss:', total_loss / batches_processed)

    if use_mlflow:
        collection_name = target_collection_path.split('/')[-1]
        logger.log_metric(f'{collection_name}_val_loss', total_loss / batches_processed, step=current_episode)

    collection_name = target_collection_path.split('/')[-1]
    print(f'{collection_name}_val_loss', total_loss / batches_processed)


def get_model_from_ckpt_file(path_to_ckpt_file: str):
    print('Loading model from file at:', path_to_ckpt_file)
    ckpt = torch.load(path_to_ckpt_file, map_location=torch.device('cpu'))

    model = rowfollow_model.RowfollowModel()
    model.load_state_dict(ckpt)

    return model


def get_config_from_file(path: str) -> TestConfig:
    with open(path, 'r') as f:
        config_dict = json.load(f)
        test_config = TestConfig(**config_dict)
        test_config.dataset_info_path = os.path.join(test_config.base_path, 'dataset_info.csv')
        test_config.support_collection_path = os.path.join(test_config.base_path,
                                                           test_config.support_collection_path)
        test_config.support_annotations_file_path = os.path.join(test_config.base_path,
                                                                 test_config.support_annotations_file_path)
        return test_config


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] == '-h':
        print('USAGE: python rowfollow_test.py path/to/test_config.json')
        sys.exit(1)

    if not os.path.exists(sys.argv[1]):
        print(f'Path {sys.argv[1]} does not exist')
        sys.exit(1)

    path_to_config = sys.argv[1]
    test_main(get_config_from_file(path_to_config))
