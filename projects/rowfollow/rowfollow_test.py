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

import rowfollow_task
import rowfollow_utils


class RowfollowValDataset(torch.utils.data.Dataset):
    def __init__(self, validation_collections_paths: list[str], validation_annotations_file_path: str):
        self.validation_collections_paths: list[str] = validation_collections_paths
        self.validation_annotations_file_path: str = validation_annotations_file_path

        self.annotations_df: pd.DataFrame = pd.read_csv(self.validation_annotations_file_path)
        self.annotations_df = self._filter_existing_images()
        self.sigma = 10

    def _filter_existing_images(self):
        # Create a list of available image paths from the directories
        available_image_names = []
        for path in self.validation_collections_paths:
            for root, _, files in os.walk(path):
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

        collection_path = None
        for path in self.validation_collections_paths:
            if collection_name in path:
                collection_path = path
                break

        if collection_path:
            image_path = os.path.join(collection_path, image_name)
        else:
            raise FileNotFoundError(f"Collection {collection_name} not found in the provided paths.")

        vp, ll, lr = rowfollow_task.RowfollowTaskOldDataset.get_kps_for_image(image_name, annotation_row=annotation_row)

        pre_processed_image, _ = rowfollow_utils.pre_process_image_old_data(image_path, new_size=(320, 224))
        pre_processed_image = torch.from_numpy(pre_processed_image)

        # vp, ll, lr are coordinates, but we need distributions
        vp_gt = rowfollow_utils.dist_from_keypoint(vp, sig=self.sigma, downscale=4)
        ll_gt = rowfollow_utils.dist_from_keypoint(ll, sig=self.sigma, downscale=4)
        lr_gt = rowfollow_utils.dist_from_keypoint(lr, sig=self.sigma, downscale=4)

        return pre_processed_image, torch.stack([vp_gt, ll_gt, lr_gt])


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
              validation_collections_paths: list[str],
              validation_annotations_file_path: str,
              device: torch.device,
              seed: Optional[int]):
    model = load_model(run_id, episode)
    inner_lrs = load_inner_lrs(run_id, episode)
    inner_buffers = load_inner_buffers(run_id, episode)

    task = rowfollow_task.RowfollowTaskOldDataset(support_annotations_file_path,
                                                  support_collection_path,
                                                  k,
                                                  device,
                                                  seed=seed)

    finetuner = maml_eval.MamlFinetuner(model, inner_lrs, inner_buffers, inner_steps, task)
    finetuner.finetune()

    model.eval()
    val_dataset = RowfollowValDataset(validation_collections_paths,
                                      validation_annotations_file_path)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False)
    total_loss = 0.0
    batches_processed = 0
    for x, y in val_dataloader:
        y_hat = model(x)
        total_loss += task.calc_loss(y_hat, y, maml_api.Stage.VAL, maml_api.SetToSetType.TARGET).item()
        batches_processed += 1
        print('total_loss:', total_loss)
        print('avg_loss:', total_loss / batches_processed)

    print('total loss: {}'.format(total_loss))


@dataclass
class TestConfig:
    run_id: str
    episode: int
    k: int
    inner_steps: int
    base_path: str
    support_collection_id: str
    support_annotations_file_path: str
    validation_collections_id: str
    validation_annotations_file_path: str
    seed: Optional[int]
    device: torch.device


def get_support_collection_path(base_path: str, support_collection_id: str) -> str:
    # TODO Integrate more complicated ids
    return os.path.join(base_path, 'train', support_collection_id)


def get_validation_collections_paths(base_path: str, validation_collections_id: str) -> list[str]:
    val_dir_base_path = os.path.join(base_path, 'val')
    if validation_collections_id == 'all_val':
        return [os.path.join(val_dir_base_path, d) for d in os.listdir(val_dir_base_path)
                if os.path.isdir(os.path.join(val_dir_base_path, d))]
    # TODO add other options


def run_main_from_test_config(test_config: TestConfig):
    test_main(run_id=test_config.run_id,
              episode=test_config.episode,
              k=test_config.k,
              inner_steps=test_config.inner_steps,
              support_collection_path=get_support_collection_path(test_config.base_path,
                                                                  test_config.support_collection_id),
              support_annotations_file_path=os.path.join(test_config.base_path,
                                                         test_config.support_annotations_file_path),
              validation_collections_paths=get_validation_collections_paths(test_config.base_path,
                                                                            test_config.validation_collections_id),
              validation_annotations_file_path=os.path.join(test_config.base_path,
                                                            test_config.validation_annotations_file_path),
              device=test_config.device,
              seed=test_config.seed)


def get_config_from_file(path: str) -> TestConfig:
    with open(path, 'r') as f:
        config_dict = json.load(f)
        test_config = TestConfig(**config_dict)
        test_config.device = torch.device(test_config.device)
        return test_config


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] == '-h' or not os.path.exists(sys.argv[1]):
        print('USAGE: python rowfollow_test.py path/to/test_config.json')
        sys.exit(1)

    path_to_config = sys.argv[1]
    run_main_from_test_config(get_config_from_file(path_to_config))
