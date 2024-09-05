import os
from typing import Callable, Optional

import mlflow
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
