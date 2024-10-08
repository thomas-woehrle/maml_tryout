import copy
import json
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
import random
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


def is_very_late(validation_collection_path: str):
    base_path = '/'.join(validation_collection_path.split('/')[:-2])
    dataset_info_path = os.path.join(base_path, 'dataset_info.csv')
    info_df = pd.read_csv(dataset_info_path)

    collection_name = validation_collection_path.split('/')[-1]
    growth_stage = rowfollow_utils.get_collection_growth_stage(info_df, collection_name)

    return growth_stage == 'very late'


class RowfollowValDataset(torch.utils.data.Dataset):
    # TODO Remove 'Val' from name
    def __init__(self,
                 validation_collection_path: str,
                 validation_annotations_file_path: str,
                 device: torch.device):
        self.validation_collection_path: str = validation_collection_path
        self.validation_annotations_file_path: str = validation_annotations_file_path

        self.annotations_df: pd.DataFrame = pd.read_csv(self.validation_annotations_file_path)
        self.annotations_df = self._filter_existing_images()
        self.sigma = 10
        self.device = device
        self.is_very_late = is_very_late(self.validation_collection_path)

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

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        annotation_row = self.annotations_df.iloc[idx]
        image_name = annotation_row['image_name']
        collection_name = image_name.split('_cam')[0]

        image_path = os.path.join(self.validation_collection_path, image_name)

        original_size = (320, 224) if self.is_very_late else (1280, 720)
        vp, ll, lr = rowfollow_task.RowfollowTaskOldDataset.get_kps_for_image(image_name, original_size=original_size,
                                                                              annotation_row=annotation_row)

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


def load_model2(run_id: str, episode: int) -> maml_api.MamlModel:
    artifact_uri = 'runs:/{}/{}/{}'.format(run_id, 'ep{}'.format(episode), 'model.pth')
    local_path = get_local_artifact_path(run_id, episode)

    print('Loading model...')
    final_path = mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri, dst_path=local_path)
    print(f'model.pth at {final_path}.')

    model = torch.load(final_path)
    return model


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
    visual_test_collection: str
    annotations_file_path: str
    device: str
    seed: Optional[int]
    use_from_pth: bool = False
    path_to_pth: Optional[str] = None
    do_finetune_non_maml: bool = False
    non_maml_lr: Optional[float] = None


@dataclass(kw_only=True)
class RowfollowLossCalculator(ABC):
    current_episode: int
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
            print(f'Calculating loss "{loss_name}" in episode {self.current_episode}')
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
        print(f'std-{loss_name}: {iteration_losses.std():.4f}')
        if self.use_mlflow:
            self.logger.log_metric(f'avg-{loss_name}', iteration_losses.mean(), self.current_episode)
            self.logger.log_metric(f'std-{loss_name}', iteration_losses.std(), self.current_episode)

    @abstractmethod
    def _finetune_and_calc_loss(self, collection_path: str, seed: int) -> tuple[float, int]:
        pass


@dataclass(kw_only=True)
class RowfollowMamlLossCalculator(RowfollowLossCalculator):
    base_model: maml_api.MamlModel
    inner_buffers: maml_api.InnerBuffers
    inner_lrs: maml_api.InnerLrs
    k: int
    inner_steps: int
    use_anil: bool
    sigma: int

    def _finetune_and_calc_loss(self, collection_path: str, seed: int):
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

    def _get_task(self, collection_path: str, seed: int) -> maml_api.MamlTask:
        return rowfollow_task.RowfollowTaskOldDataset(self.annotations_file_path,
                                                      collection_path,
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


@dataclass(kw_only=True)
class RowfollowNonMamlLossCalculator(RowfollowLossCalculator):
    base_model: rowfollow_model.RowfollowModel
    do_finetune: bool = False
    # these are only needed if do_finetune=True
    k: Optional[int] = None
    inner_steps: Optional[int] = None
    alpha: Optional[float] = None
    sigma: Optional[int] = None
    # TODO comapre sigma of MAML (mostly 10) and non-MAML (???)

    def _calc_loss(self, loss_name: str, collection_paths: list[str], num_iterations: int):
        # only if finetuning happens, do we need multiple iterations
        if not self.do_finetune:
            num_iterations = 1
        super()._calc_loss(loss_name, collection_paths, num_iterations)

    def _finetune_and_calc_loss(self, collection_path: str, seed: int) -> tuple[float, int]:
        # copying, just be safe
        model = copy.deepcopy(self.base_model)
        model.eval()

        # we only need this task for its loss function. Therefore, k, sigma and seed are irrelevant. Not very pretty
        task_for_loss_calc = rowfollow_task.RowfollowTaskOldDataset(annotations_file_path=self.annotations_file_path,
                                                                    support_data_path=collection_path,
                                                                    k=-1,
                                                                    device=self.device,
                                                                    sigma=-1,
                                                                    seed=-1
                                                                    )

        # finetune the model if necessary
        if self.do_finetune:
            self._finetune_model(model, collection_path, task_for_loss_calc, seed)

        collection_loss, collection_n_batches = calc_loss_for_one_collection(collection_path,
                                                                             self.annotations_file_path,
                                                                             model,
                                                                             task_for_loss_calc,
                                                                             self.device)

        return collection_loss, collection_n_batches

    def _finetune_model(self,
                        model: rowfollow_model.RowfollowModel,
                        collection_path: str,
                        task_for_loss_calc: maml_api.MamlTask,
                        seed: int):
        print('Finetuning non-MAML model...')
        model.train()

        x, y = self._sample(collection_path, seed)

        optimizer = torch.optim.SGD(model.parameters(), lr=self.alpha)
        for i in range(self.inner_steps):
            optimizer.zero_grad()
            y_hat = model(x)
            loss = task_for_loss_calc.calc_loss(y_hat, y, maml_api.Stage.VAL, maml_api.SetToSetType.SUPPORT)
            print(f'Loss at step {i}: {loss:.4f}')
            loss.backward()
            optimizer.step()

        model.eval()

    def _sample(self, collection_path: str, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Pretty much copied from the .sample(...) of RowfollowTaskOldDataset
        # TODO integrate into MamlFinetuner, or use DataLoader instead or ...
        annotations = pd.read_csv(self.annotations_file_path)

        all_img_names = [f for f in os.listdir(collection_path) if f.endswith('.jpg')]

        # set deterministic seeding
        random.seed(seed)
        all_img_names = sorted(all_img_names)
        img_names = random.sample(all_img_names, k=self.k)

        x = []
        y = []

        for img_name in img_names:
            image_path = os.path.join(collection_path, img_name)

            original_size = (320, 224) if is_very_late(collection_path) else (1280, 720)
            vp, ll, lr = rowfollow_task.RowfollowTaskOldDataset.get_kps_for_image(img_name, annotations,
                                                                                  original_size=original_size)

            pre_processed_image, _ = rowfollow_utils.pre_process_image_old_data(image_path, new_size=(320, 224))
            pre_processed_image = torch.from_numpy(pre_processed_image)
            x.append(pre_processed_image)

            # vp, ll, lr are coordinates, but we need distributions
            vp_gt = rowfollow_utils.dist_from_keypoint(vp, sig=self.sigma, downscale=4)
            ll_gt = rowfollow_utils.dist_from_keypoint(ll, sig=self.sigma, downscale=4)
            lr_gt = rowfollow_utils.dist_from_keypoint(lr, sig=self.sigma, downscale=4)
            y.append(torch.stack([vp_gt, ll_gt, lr_gt]))

        x = torch.stack(x)
        y = torch.stack(y)

        return x.to(self.device), y.to(self.device)


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
    if config.use_from_pth:
        state_dict = torch.load(config.path_to_pth, map_location=config.device)
        model = rowfollow_model.RowfollowModel()
        model.load_state_dict(state_dict)

        loss_calculator = RowfollowNonMamlLossCalculator(
            current_episode=-1,
            loss_calc_info=config.loss_calc_info,
            annotations_file_path=config.annotations_file_path,
            device=torch.device(config.device),
            base_seed=config.seed,
            use_mlflow=False,
            logger=None,
            base_model=model,
            do_finetune=config.do_finetune_non_maml,
            k=config.k,
            inner_steps=config.inner_steps,
            alpha=config.non_maml_lr,
            sigma=config.sigma
        )
    else:
        # model = load_model(config.run_id, config.episode)
        model = load_model2(config.run_id, config.episode)
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


def main():
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


if __name__ == '__main__':
    main()
