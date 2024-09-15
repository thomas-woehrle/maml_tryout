import os
from typing import Callable, Optional

import cv2
import mlflow
import numpy as np
import torch
from torch import autograd

import maml_api
import maml_config


class MlflowArtifactManager:
    def __init__(self, run_id: str, episode: int,  download_lrs: bool, download_buffers: bool,
                 lib_directory: str = '~/Library/Application Support/torchmaml/runs',
                 model_name: str = 'model', override_downloads: bool = False):
        self.run_id: str = run_id
        self.episode_str: str = 'ep{}'.format(episode)
        self.download_lrs: bool = download_lrs
        self.download_buffers: bool = download_buffers
        self.model_name = model_name
        self.lib_directory = lib_directory
        os.makedirs(self.lib_directory, exist_ok=True)
        self.check_and_download_artifacts(override_downloads)

    def check_and_download_artifacts(self, override_downloads: bool):
        # Download files if needed
        if not os.path.exists(os.path.join(self.lib_directory, self.run_id)) or override_downloads:
            os.makedirs(os.path.join(self.lib_directory, self.run_id))
            self.download_artifact('hparams.json')
        if not os.path.exists(os.path.join(self.lib_directory, self.run_id, self.episode_str)) or override_downloads:
            os.makedirs(os.path.join(self.lib_directory, self.run_id, self.episode_str))
            if self.download_lrs:
                self.download_artifact(os.path.join(self.episode_str, 'inner_lrs.json'))
            if self.download_buffers:
                self.download_artifact(os.path.join(self.episode_str, 'inner_buffers.json'))
            self.download_model()

    def download_artifact(self, artifact_path: str):
        print(f'Downloading artifact {artifact_path}')
        mlflow.artifacts.download_artifacts(run_id=self.run_id, artifact_path=artifact_path,
                                            dst_path=os.path.join(self.lib_directory, self.run_id),
                                            tracking_uri='databricks')

    def download_model(self):
        dst_path = os.path.join(self.lib_directory, self.run_id, self.episode_str)
        os.makedirs(dst_path, exist_ok=True)

        try:
            model_uri = 'runs:/{}/{}/{}'.format(self.run_id, self.episode_str, self.model_name)

            print(f'Downloading model from {model_uri}')
            mlflow.pytorch.load_model(model_uri, dst_path)
        except mlflow.exceptions.MlflowException as e:
            print(f"MLflow error: {e.error_code}. The model probably didn't exist under the given path. ")
            model_uri = 'runs:/{}/{}/{}'.format(self.run_id, 'models', self.episode_str)
            print(f'Downloading model from {model_uri}')
            mlflow.pytorch.load_model(model_uri, dst_path)

    def load_model(self) -> maml_api.MamlModel:
        return mlflow.pytorch.load_model(os.path.join(self.lib_directory, self.run_id,
                                                      self.episode_str, self.model_name))

    def load_hparams(self) -> maml_config.MamlHyperParameters:
        hparams_dict = mlflow.artifacts.load_dict(os.path.join(self.lib_directory, self.run_id, 'hparams.json'))
        return maml_config.MamlHyperParameters(**hparams_dict)

    def load_inner_lrs(self) -> dict[str, list[float]]:
        # in the case of the lrs, they don't have to be transformed to tensors
        return mlflow.artifacts.load_dict(os.path.join(self.lib_directory, self.run_id,
                                                       self.episode_str, 'inner_lrs.json'))

    def load_inner_buffers(self) -> dict[str, dict[str, torch.Tensor]]:
        inner_buffers = mlflow.artifacts.load_dict(os.path.join(self.lib_directory, self.run_id,
                                                                self.episode_str, 'inner_buffers.json'))
        for i, named_buffers in inner_buffers.items():
            for n, b in named_buffers.items():
                inner_buffers[i][n] = torch.tensor(b)
        return inner_buffers


class MamlEvaluator:
    def __init__(self, run_id: str, episode: int, sample_task: Callable[[maml_api.Stage], maml_api.MamlTask],
                 hparams: Optional[maml_config.MamlHyperParameters] = None,
                 inner_steps: Optional[int] = None,
                 lib_directory: str = '~/Library/Application Support/torchmaml/runs',  # TODO use better default path
                 model_name: str = 'model',
                 override_downloads: bool = False,
                 ):
        self.artifact_manager = MlflowArtifactManager(run_id, episode, hparams.use_lslr, hparams.use_bnrs,
                                                      lib_directory, model_name, override_downloads)

        self.sample_task = sample_task
        self.hparams = hparams or self.artifact_manager.load_hparams()
        if inner_steps is not None:
            self.hparams.inner_steps = inner_steps

        self.model = self.artifact_manager.load_model()
        self.inner_lrs = self.artifact_manager.load_inner_lrs() if self.hparams.use_lslr else None
        self.inner_buffers = self.artifact_manager.load_inner_buffers() if self.hparams.use_bnrs else None

        self.already_finetuned: bool = False

    def inner_step(self, x_support: torch.Tensor, y_support: torch.Tensor,
                   params: maml_api.NamedParams, task: maml_api.MamlTask,
                   num_step: int, stage: maml_api.Stage) -> maml_api.NamedParams:
        # TODO add anil back
        # str(num_step), because keys are strings like "0" through saving in json
        # and it would be useless computation to transform to int
        if self.inner_buffers is not None:
            eff_num_step = min(num_step, len(self.inner_buffers.keys()) - 1)
            inner_buffers_to_use = self.inner_buffers[str(eff_num_step)]
        else:
            eff_num_step = num_step
            inner_buffers_to_use = self.model.get_state()[1]

        y_hat = self.model.func_forward(x_support, params, inner_buffers_to_use)
        train_loss = task.calc_loss(y_hat, y_support, stage, maml_api.SetToSetType.SUPPORT)
        print('train_loss at step {}: {}'.format(num_step, train_loss.item()))

        grads = autograd.grad(train_loss, params.values(),
                              create_graph=False)

        factor = 1 if eff_num_step >= num_step else 0.5  # TODO evaluate if this actually brings an improvement
        return {n: p - factor *
                (self.inner_lrs[n.replace('.', '-')][eff_num_step] if self.inner_lrs is not None else self.hparams.alpha)
                * g for (n, p), g in zip(params.items(), grads)}

    def finetune(self):
        if self.already_finetuned:
            print('WARNING: Already finetuned...')
            return False
        print('Finetuning...')
        params, _ = self.model.get_state()
        task = self.sample_task(maml_api.Stage.VAL)
        x_support, y_support = task.sample(maml_api.SetToSetType.SUPPORT)

        params_i = {n: p for n, p in params.items()}
        # i symbolizes the i-th step
        for i in range(self.hparams.inner_steps):
            # finetune params
            params_i = self.inner_step(x_support, y_support, params_i, task, i, maml_api.Stage.VAL)

        self.already_finetuned = True
        self.model.load_state_dict(params_i, strict=False)
        print('Finished finetuning. \n')
        return True

    def predict(self, seed: Optional[int] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Runs one inference on the task at hand and returns y_hat and the loss

        Returns:
            A tuple of 1. the raw prediction ie y_hat and 2. the calculated target loss
        """
        # .eval() ?
        if not self.already_finetuned:
            print('WARNING: Not finetuned yet')
        kwargs = {} if seed is None else {'seed': seed}
        task = self.sample_task(maml_api.Stage.VAL, **kwargs)
        x_target, y_target = task.sample(maml_api.SetToSetType.TARGET)

        y_hat = self.model(x_target)

        return y_hat, task.calc_loss(y_hat, y_target, maml_api.Stage.VAL, maml_api.SetToSetType.TARGET)


# TODO This does NOT belong here, but will live with it for now:
def reverse_preprocessing(pre_processed_img: torch.Tensor) -> np.ndarray:
    # Move the tensor to CPU if it's not
    pre_processed_img = pre_processed_img.cpu() if pre_processed_img.is_cuda else pre_processed_img

    # Convert to NumPy array
    image = pre_processed_img.numpy()

    # Transpose to HWC (Height, Width, Channels)
    image = np.transpose(image, (1, 2, 0))

    # Reverse normalization
    image[:, :, 0] = image[:, :, 0] * 0.229 + 0.485  # Red channel
    image[:, :, 1] = image[:, :, 1] * 0.224 + 0.456  # Green channel
    image[:, :, 2] = image[:, :, 2] * 0.225 + 0.406  # Blue channel

    # Convert range back to [0, 255]
    image = (image * 255).astype(np.uint8)

    # Convert RGB to BGR (for OpenCV)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image


class MamlFinetuner:
    def __init__(self,
                 model: maml_api.MamlModel,
                 inner_lrs: maml_api.InnerLrs,
                 inner_buffers: maml_api.InnerBuffers,
                 inner_steps: int,
                 task: maml_api.MamlTask,
                 use_mlflow: bool = False
                 ):
        self.model: maml_api.MamlModel = model
        self.inner_lrs: maml_api.InnerLrs = inner_lrs
        self.inner_buffers: maml_api.InnerBuffers = inner_buffers
        self.inner_steps: int = inner_steps
        self.task: maml_api.MamlTask = task
        self.use_mlflow: bool = use_mlflow

    def inner_step(self, x_support: torch.Tensor, y_support: torch.Tensor,
                   params: maml_api.NamedParams,
                   num_step: int) -> maml_api.NamedParams:
        capped_num_step = min(num_step, len(self.inner_buffers.keys()) - 1)

        y_hat = self.model.func_forward(x_support, params, self.inner_buffers[str(capped_num_step)])
        train_loss = self.task.calc_loss(y_hat, y_support, maml_api.Stage.VAL, maml_api.SetToSetType.SUPPORT)
        print('train_loss at step {}: {}'.format(num_step, train_loss.item()))

        if self.use_mlflow:
            mlflow.log_metric('train_loss', train_loss.item(), num_step)

        grads = autograd.grad(train_loss, params.values(), create_graph=False)

        # TODO implement different learning rate strategies
        # after the number of learned steps, the learning rate of the last step is halved
        factor = 1 if capped_num_step == num_step else 0.5
        return {n: p - factor * self.inner_lrs[n.replace('.', '-')][capped_num_step] * g
                for (n, p), g in zip(params.items(), grads)}

    def finetune(self):
        """Finetunes self.model using the given (x, y)

        x_support: Input image batch in shape BxCxHxW
        y_support: Target in shape Bx[...]
        """
        print('Finetuning...')
        params, _ = self.model.get_state()
        params_i = {n: p for n, p in params.items()}

        x_support, y_support = self.task.sample(maml_api.SetToSetType.SUPPORT)

        if self.use_mlflow:
            # NOTE: currently this is only applicable to the rowfollow usecase
            for idx, img in enumerate(x_support):
                img = reverse_preprocessing(img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mlflow.log_image(img, f'support_imgs/{idx}.png')

        # i symbolizes the i-th step
        for i in range(self.inner_steps):
            # finetune params
            params_i = self.inner_step(x_support, y_support, params_i, i)

        self.model.load_state_dict(params_i, strict=False)
        print('Finished finetuning. \n')
