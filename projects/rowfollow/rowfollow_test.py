import copy
import json
import os
import sys
from dataclasses import dataclass
from typing import Optional

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.utils.data

import maml_api
import maml_eval
import maml_logging

import rowfollow_model
import rowfollow_task
import rowfollow_utils


# TODO this file should be split into multiple smaller ones


class RowfollowValDataset(torch.utils.data.Dataset):
    # TODO Remove 'Val' from name
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
    sigma: int
    use_anil: bool
    base_path: str
    loss_calc_info: dict[str, tuple[list[str], int]]
    annotations_file_path: str
    device: str
    seed: Optional[int]


@dataclass
class RowfollowMamlLossCalculator:
    current_episode: int
    base_model: maml_api.MamlModel  # Make sure that the model is in the right mode
    inner_buffers: maml_api.InnerBuffers
    inner_lrs: maml_api.InnerLrs
    k: int
    inner_steps: int
    use_anil: bool
    sigma: int
    # Maps a loss_name to a list of paths to collections used to calculate said loss and an int indicating the
    # number of iterations to use to calculate this loss
    loss_calc_info: dict[str, tuple[list[str], int]]
    annotations_file_path: str
    device: torch.device
    base_seed: int
    use_mlflow: bool
    logger: Optional[maml_logging.Logger]

    def calc_losses(self):
        for loss_name, (collections, num_iterations) in self.loss_calc_info.items():
            print('\n++++++++++++++++++++++++++++++++++++++++++++++++++++')
            print(f'Calculating loss "{loss_name}"')
            self._calc_loss(loss_name, collections, num_iterations)
            print(f'Finished calculating loss "{loss_name}"')
            print('++++++++++++++++++++++++++++++++++++++++++++++++++++\n')

    def _calc_loss(self, loss_name: str, collection_paths: list[str], num_iterations: int):
        iteration_losses = []
        for i in range(num_iterations):
            print('\n------------------------')
            print(f'Iteration {i} started')
            # Create trackers. This takes into account that different collections have a different number of images
            losses_batches = []
            total_n_batches = 0
            for collection_path in collection_paths:
                iteration_seed = self.base_seed + i

                collection_loss, collection_n_batches = self._finetune_and_calc_loss(collection_path, iteration_seed)

                losses_batches.append((collection_loss, collection_n_batches))
                total_n_batches += collection_n_batches

            # calculate the loss for this iteration
            iteration_loss = 0.0
            for loss, n_batches in losses_batches:
                iteration_loss += loss * n_batches / total_n_batches
            iteration_losses.append(iteration_loss)

            # print and log iteration loss
            print(f'\n{loss_name}/iter{i}: {iteration_loss:.4f}')
            print(f'Iteration {i} finished')
            print('------------------------\n')
            if self.use_mlflow:
                self.logger.log_metric(f'{loss_name}/iter{i}', iteration_loss, self.current_episode)

        # print and log loss as calculated across iterations
        iteration_losses = np.array(iteration_losses)
        print(f'avg-{loss_name}: {iteration_losses.mean():.4f}')
        print(f'std{loss_name}: {iteration_losses.std():.4f}')
        if self.use_mlflow:
            self.logger.log_metric(f'avg-{loss_name}', iteration_losses.mean(), self.current_episode)
            self.logger.log_metric(f'std-{loss_name}', iteration_losses.std(), self.current_episode)

    def _finetune_and_calc_loss(self, collection_path, seed: int):
        # copy model and set it in eval mode. This is needed even for finetuning to use the inner_buffers
        model = copy.deepcopy(self.base_model).to(self.device)
        model.eval()

        # get task and finetune model
        task = self._get_task(collection_path, seed)
        self._finetune_model(model, task)  # in-place operation

        collection_loss, collection_n_batches = calc_loss_for_one_collection(collection_path,
                                                                             self.annotations_file_path,
                                                                             model,
                                                                             task,
                                                                             self.device)

        return collection_loss, collection_n_batches

    def _get_task(self, collection_file_path: str, seed: int) -> maml_api.MamlTask:
        return rowfollow_task.RowfollowTaskOldDataset(self.annotations_file_path,
                                                      collection_file_path,
                                                      self.k,
                                                      self.device,
                                                      self.sigma,
                                                      seed=seed)

    def _finetune_model(self, model: maml_api.MamlModel, task: maml_api.MamlTask):
        finetuner = maml_eval.MamlFinetuner(model,
                                            self.inner_lrs,
                                            self.inner_buffers,
                                            self.inner_steps,
                                            task,
                                            self.use_anil,
                                            use_mlflow=False)
        finetuner.finetune()


def calc_loss_for_one_collection(
        collection_path: str,
        annotations_file_path: str,
        model: maml_api.MamlModel,
        task: maml_api.MamlTask,
        device: torch.device,
) -> tuple[float, int]:
    with torch.no_grad():
        val_dataset = RowfollowValDataset(collection_path,
                                          annotations_file_path, device=device)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False)
        total_loss = 0.0
        batches_processed = 0
        for x, y in val_dataloader:
            y_hat = model(x)
            total_loss += task.calc_loss(y_hat, y, maml_api.Stage.VAL, maml_api.SetToSetType.TARGET).item()
            batches_processed += 1
            print('Current loss:', total_loss / batches_processed)

        return total_loss / batches_processed, batches_processed


def evaluate_from_config(config: TestConfig):
    model = load_model(config.run_id, config.episode)
    inner_lrs = load_inner_lrs(config.run_id, config.episode)
    inner_buffers = load_inner_buffers(config.run_id, config.episode)

    loss_calculator = RowfollowMamlLossCalculator(
        current_episode=-1,
        base_model=model,
        inner_buffers=inner_buffers,
        inner_lrs=inner_lrs,
        k=config.k,
        inner_steps=config.inner_steps,
        use_anil=config.use_anil,
        sigma=config.sigma,
        loss_calc_info=config.loss_calc_info,
        annotations_file_path=config.annotations_file_path,
        device=torch.device(config.device),
        base_seed=config.seed,
        # mlflow val logging might be supported in the future:
        use_mlflow=False,
        logger=None
    )

    loss_calculator.calc_losses()


def get_model_from_ckpt_file(path_to_ckpt_file: str):
    print('Loading model from file at:', path_to_ckpt_file)
    ckpt = torch.load(path_to_ckpt_file, map_location=torch.device('cpu'))

    model = rowfollow_model.RowfollowModel()
    model.load_state_dict(ckpt)

    return model


def convert_dict_to_loss_calculation_info(config_dict: dict, base_path: str):
    # In place operation
    for loss_name, loss_info in config_dict['loss_calc_info'].items():
        collections = loss_info['collections']
        num_iterations = loss_info['num_iterations']
        for idx, collection in enumerate(collections):
            collection_path = os.path.join(base_path, collection)
            collections[idx] = collection_path
        config_dict['loss_calc_info'][loss_name] = (collections, num_iterations)


def get_config_from_file(path: str) -> TestConfig:
    with open(path, 'r') as f:
        config_dict = json.load(f)
        base_path = config_dict['base_path']
        config_dict['annotations_file_path'] = os.path.join(base_path, config_dict['annotations_file_path'])

        convert_dict_to_loss_calculation_info(config_dict, base_path)

        test_config = TestConfig(**config_dict)
        return test_config


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] == '-h':
        print('USAGE: python rowfollow_test.py path/to/test_config.json')
        sys.exit(1)

    if not os.path.exists(sys.argv[1]):
        print(f'Path {sys.argv[1]} does not exist')
        sys.exit(1)

    path_to_config = sys.argv[1]
    config = get_config_from_file(path_to_config)

    evaluate_from_config(config)
    # TODO add possibility to evaluate non-MAML
