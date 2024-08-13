import copy
from typing import Callable

import numpy as np
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
        # the first 10 percent of episodes will use strong multi-step loss updates, afterward weak
        self.multi_step_loss_n_episodes: int = int(hparams.n_episodes * 0.1) or 1
        # the first 30 percent of episodes will use first order updates
        self.first_order_updates_n_episodes: int = int(hparams.n_episodes * 0.3)

        self.current_episode: int = 0

    def get_per_step_loss_importance_vector(self, stage: maml_api.Stage) -> torch.Tensor:
        # Adapted from: https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch
        """
        Generates a tensor of dimensionality (num_inner_loop_steps) indicating the importance of each step's target
        loss towards the optimization loss.

        Args:
            stage: The current stage of MAML.

        Returns:
            The loss importance vector.
        """
        # If Val stage, then only the last step loss is important TODO is this actually the case?
        if stage == maml_api.Stage.VAL:
            loss_weights = np.zeros(self.hparams.inner_steps, dtype=np.float32)
            loss_weights[-1] = 1.0
        else:
            loss_weights = np.ones(self.hparams.inner_steps, dtype=np.float32) / self.hparams.inner_steps
            decay_rate = 1.0 / self.hparams.inner_steps / self.multi_step_loss_n_episodes
            min_value_for_non_final_losses = 0.03 / self.hparams.inner_steps
            for i in range(len(loss_weights) - 1):
                loss_weights[i] = np.maximum(loss_weights[i] - (self.current_episode * decay_rate),
                                             min_value_for_non_final_losses)
            loss_weights[-1] = np.minimum(
                loss_weights[-1] + (self.current_episode * (self.hparams.inner_steps - 1) * decay_rate),
                1.0 - ((self.hparams.inner_steps - 1) * min_value_for_non_final_losses))
            loss_weights[-1] = 1 - loss_weights[0] * (len(loss_weights)-1)

        loss_weights = torch.Tensor(loss_weights).to(device=self.device)
        return loss_weights

    def inner_step(self, x_support: torch.Tensor, y_support: torch.Tensor,
                   params: NamedParams, buffers: NamedBuffers, task: maml_api.MamlTask,
                   stage: maml_api.Stage) -> NamedParams:
        # TODO add anil back
        y_hat = self.model.func_forward(x_support, params, buffers)
        train_loss = task.calc_loss(y_hat, y_support, stage, maml_api.SetToSetType.SUPPORT)

        mlflow.log_metric("second_order_true", int(self.current_episode > self.first_order_updates_n_episodes),
                          self.current_episode)
        grads = autograd.grad(
            train_loss, params.values(), create_graph=(stage == maml_api.Stage.TRAIN
                                                       and self.current_episode > self.first_order_updates_n_episodes))

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

        per_step_loss_importance_vector = self.get_per_step_loss_importance_vector(stage)
        params_i = {n: p for n, p in params.items()}
        loss = torch.tensor(0.0, device=self.device)

        # i symbolizes the i-th step
        for i in range(self.hparams.inner_steps):
            # finetune params
            params_i = self.inner_step(x_support, y_support, params_i, buffers, task, stage)

            # calculate target loss using new params
            target_loss = task.calc_loss(self.model.func_forward(x_target, params_i, buffers), y_target,
                                         stage, maml_api.SetToSetType.TARGET)
            target_loss *= per_step_loss_importance_vector[i]
            loss += target_loss

        return loss

    def train(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.hparams.beta)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.hparams.n_episodes,
                                                         eta_min=self.hparams.min_beta)
        _, buffers = self.model.get_state()

        for episode in range(self.hparams.n_episodes):
            self.current_episode = episode
            optimizer.zero_grad()
            params, _ = self.model.get_state()

            batch_loss = torch.tensor(0.0, device=self.device)
            for i in range(self.hparams.meta_batch_size):
                batch_loss += self.meta_forward(params, buffers, maml_api.Stage.TRAIN)

            batch_loss.backward()
            optimizer.step()

            # log current lr then step
            mlflow.log_metric("lr", lr_scheduler.get_last_lr()[0], episode)
            lr_scheduler.step()

            if self.do_use_mlflow:
                # log acc_loss
                mlflow.log_metric("batch_loss", batch_loss.item(), step=episode)

                # log eval_loss under condition
                # TODO take mean across multiple runs?
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
