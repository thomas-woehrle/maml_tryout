import os
from typing import Callable, Optional

import mlflow
import torch
from torch import autograd

import maml_api
import maml_config


class MlflowArtifactManager:
    def __init__(self, run_id: str, episode: int, lib_directory: str = '~/Library/Application Support/torchmaml/runs',
                 model_name: str = 'model', override_downloads: bool = False):
        self.run_id: str = run_id
        self.episode_str: str = 'ep{}'.format(episode)
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
            self.download_artifact(os.path.join(self.episode_str, 'inner_lrs.json'))
            self.download_artifact(os.path.join(self.episode_str, 'inner_buffers.json'))
            self.download_model()

    def download_artifact(self, artifact_path: str):
        mlflow.artifacts.download_artifacts(run_id=self.run_id, artifact_path=artifact_path,
                                            dst_path=os.path.join(self.lib_directory, self.run_id),
                                            tracking_uri='databricks')

    def download_model(self):
        dst_path = os.path.join(self.lib_directory, self.run_id, self.episode_str)
        os.makedirs(dst_path, exist_ok=True)

        model_uri = 'runs:/{}/{}/{}'.format(self.run_id, self.episode_str, self.model_name)

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
                 lib_directory: str = '~/Library/Application Support/torchmaml/runs',
                 model_name: str = 'model',
                 override_downloads: bool = False
                 ):
        self.artifact_manager = MlflowArtifactManager(run_id, episode, lib_directory, model_name, override_downloads)

        self.sample_task = sample_task
        self.hparams = hparams or self.artifact_manager.load_hparams()
        self.model = self.artifact_manager.load_model()
        self.inner_lrs = self.artifact_manager.load_inner_lrs()
        self.inner_buffers = self.artifact_manager.load_inner_buffers()

        self.already_finetuned: bool = False

    def inner_step(self, x_support: torch.Tensor, y_support: torch.Tensor,
                   params: maml_api.NamedParams, task: maml_api.MamlTask,
                   num_step: int, stage: maml_api.Stage) -> maml_api.NamedParams:
        # TODO add anil back
        # str(num_step), because keys are strings like "0" through saving in json
        # and it would be useless computation to transform to int
        y_hat = self.model.func_forward(x_support, params, self.inner_buffers[str(num_step)])
        train_loss = task.calc_loss(y_hat, y_support, stage, maml_api.SetToSetType.SUPPORT)
        print('train_loss at step {}: {}'.format(num_step, train_loss.item()))

        grads = autograd.grad(train_loss, params.values(),
                              create_graph=False)

        # return self.inner_optimizer.update_params(params, names_grads_dict, num_step)
        return {n: p - self.inner_lrs[n.replace('.', '-')][num_step] *
                       g for (n, p), g in zip(params.items(), grads)}

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
