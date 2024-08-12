import copy
from typing import Callable

import mlflow
import torch
import torch.optim as optim
from torch import autograd

import maml_config
import maml_api

NamedParams = dict[str, torch.nn.parameter.Parameter]
NamedBuffers = dict[str, torch.Tensor]


def default_end_of_ep_fct(params, buffers: torch.tensor,
                          episode: int, acc_loss: float, val_loss: float):
    # TODO offer possibility for custom end_of_ep_fct
    """Default function used at the end of an episode. See usage below."""
    print('acc_loss :', acc_loss)
    print('val_loss :', val_loss)


class MamlTrainer:
    def __init__(self, hparams: maml_config.MamlHyperParameters,
                 sample_task: Callable[[maml_api.Stage], maml_api.MamlTask], model: maml_api.MamlModel,
                 device: torch.device,
                 do_use_mlflow: bool = False,
                 n_val_iters: int = 10,
                 log_val_loss_every_n_episodes: int = 100,
                 log_model_every_n_episodes: int = 1000):
        """
        Args:
            hparams: Hyperparameters relevant for MAML.
            sample_task: Function used to sample tasks i.e. x and y in the supervised case
            model: Model to train
            device: Device to work on
            do_use_mlflow: Inidactes whether mlflow should be used
            n_val_iters: Number of iterations for each validation loss calculation
            log_val_loss_every_n_episodes: Frequency of eval loss logging.
                                            First and last will always be logged. (Default: 100)
            log_model_every_n_episodes: Frequency of model logging. First and last will always be logged. (Default: 1000)
        """
        self.hparams: maml_config.MamlHyperParameters = hparams
        self.sample_task: Callable[[maml_api.Stage], maml_api.MamlTask] = sample_task
        self.model: maml_api.MamlModel = model
        self.device: torch.device = device
        self.do_use_mlflow: bool = do_use_mlflow
        self.n_val_iters: int = n_val_iters
        self.log_val_loss_every_n_episodes: int = log_val_loss_every_n_episodes
        self.log_model_every_n_episodes: int = log_model_every_n_episodes

    def inner_step(self, x: torch.Tensor, y: torch.Tensor,
                   params: NamedParams, buffers: NamedBuffers, task: maml_api.MamlTask,
                   stage: maml_api.Stage) -> NamedParams:
        # TODO add anil back
        y_hat = self.model.func_forward(x, params, buffers)
        train_loss = task.calc_loss(y_hat, y)

        grads = autograd.grad(
            train_loss, params.values(), create_graph=(stage == maml_api.Stage.TRAIN))

        return {n: p - self.hparams.alpha *
                       g for (n, p), g in zip(params.items(), grads)}

    def meta_forward(self, params: NamedParams, buffers: NamedBuffers, stage: maml_api.Stage):
        """Does a meta forward pass

        It first samples a task, then finetunes the model parameters on its support set, then calculates a loss
        using its target set. This loss has to be backpropagateable if the stage is TRAIN.
        """
        task = self.sample_task(stage)
        x_support, y_support = task.sample()
        x_target, y_target = task.sample()

        params_i = {n: p for n, p in params.items()}
        loss = torch.tensor(0.0, device=self.device)
        # i symbolizes the i-th step
        for i in range(self.hparams.inner_steps):
            # finetune params
            params_i = self.inner_step(x_support, y_support, params_i, buffers, task, stage)

            # calculate target loss using new params
            target_loss = task.calc_loss(self.model.func_forward(x_target, params_i, buffers), y_target)
            target_loss *= 1 if (i == self.hparams.inner_steps-1) else 0  # TODO MSL addition
            loss += target_loss

        return loss

    def train(self):
        optimizer = optim.SGD(self.model.parameters(), lr=self.hparams.beta)
        _, buffers = self.model.get_state()

        for episode in range(self.hparams.n_episodes):
            optimizer.zero_grad()
            params, _ = self.model.get_state()

            batch_loss = torch.tensor(0.0, device=self.device)
            for i in range(self.hparams.meta_batch_size):
                batch_loss += self.meta_forward(params, buffers, maml_api.Stage.TRAIN)

            batch_loss.backward()
            optimizer.step()

            if self.do_use_mlflow:
                # log acc_loss
                mlflow.log_metric("batch_loss", batch_loss.item(), step=episode)

                # log eval_loss under condition
                if episode % self.log_val_loss_every_n_episodes == 0 or episode == self.hparams.n_episodes - 1:
                    end_params, _ = self.model.get_state()
                    val_loss = self.meta_forward(end_params, buffers, maml_api.Stage.VAL)
                    mlflow.log_metric("val_loss", val_loss.item(), step=episode)

                # log model under condition
                if episode % self.log_model_every_n_episodes == 0 or episode == self.hparams.n_episodes - 1:
                    # TrainingStage passed to sample_task shouldn't play a role here
                    example_x = self.sample_task(maml_api.Stage.TRAIN).sample()[0].cpu().numpy()
                    mlflow.pytorch.log_model(copy.deepcopy(self.model).cpu(), f'models/ep{episode}', input_example=example_x)

            # '0' temporary. ideal: val_loss.item())
            default_end_of_ep_fct(params, buffers, episode, batch_loss.item(), 0)

        # TODO put buffers back into model?
