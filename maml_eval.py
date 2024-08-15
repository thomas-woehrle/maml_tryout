from typing import Callable

import torch
from torch import autograd

import maml_api
import maml_config
import maml_logging


class MamlEvaluator:
    def __init__(self, hparams: maml_config.MamlHyperParameters,
                 sample_task: Callable[[maml_api.Stage], maml_api.MamlTask], model: maml_api.MamlModel,
                 inner_lrs: dict[str, torch.Tensor], inner_buffers: dict[int, dict[str, torch.Tensor]],
                 device: torch.device):
        self.hparams: maml_config.MamlHyperParameters = hparams
        self.sample_task: Callable[[maml_api.Stage], maml_api.MamlTask] = sample_task
        self.model: maml_api.MamlModel = model
        self.inner_lrs: dict[str, torch.Tensor] = inner_lrs
        self.inner_buffers: dict[int, dict[str, torch.Tensor]] = inner_buffers
        self.device: torch.device = device
        self.already_finetuned: bool = False
        self.logger = maml_logging.Logger()

    def inner_step(self, x_support: torch.Tensor, y_support: torch.Tensor,
                   params: maml_api.NamedParams, task: maml_api.MamlTask,
                   num_step: int, stage: maml_api.Stage) -> maml_api.NamedParams:
        # TODO add anil back
        y_hat = self.model.func_forward(x_support, params, self.inner_buffers[num_step])
        train_loss = task.calc_loss(y_hat, y_support, stage, maml_api.SetToSetType.SUPPORT)

        grads = autograd.grad(train_loss, params.values(),
                              create_graph=False)

        # return self.inner_optimizer.update_params(params, names_grads_dict, num_step)
        return {n: p - self.inner_lrs[n.replace('.', '-')][num_step] *
                       g for (n, p), g in zip(params.items(), grads)}

    def train(self):
        if self.already_finetuned:
            print('WARNING: Already finetuned...')
            return False
        params, _ = self.model.get_state()
        task = self.sample_task(maml_api.Stage.VAL)
        x_support, y_support = task.sample()

        params_i = {n: p for n, p in params.items()}
        # i symbolizes the i-th step
        for i in range(self.hparams.inner_steps):
            # finetune params
            params_i = self.inner_step(x_support, y_support, params_i, task, i, maml_api.Stage.VAL)

        self.already_finetuned = True
        # TODO load parameters into Model
        return True

    def predict(self):
        """
        Runs one inference on the task at hand and returns y_hat and the loss
        """
        if not self.already_finetuned:
            print('WARNING: Not finetuned yet')
        task = self.sample_task(maml_api.Stage.VAL)
        x_target, y_target = task.sample()

        y_hat = self.model(x_target)

        return y_hat, task.calc_loss(y_hat, y_target, maml_api.Stage.VAL, maml_api.SetToSetType.TARGET)
