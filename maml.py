import copy
from typing import Callable

import numpy as np
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from torch import autograd

import maml_api
import maml_config
import maml_inner_optimizers
import maml_logging


def default_end_of_ep_fct(params, buffers: torch.tensor,
                          episode: int, acc_loss: float, val_loss: float):
    # TODO offer possibility for custom end_of_ep_fct
    """Default function used at the end of an episode. See usage below."""
    print('acc_loss :', acc_loss)
    print('val_loss :', val_loss)


class MamlTrainer(nn.Module):
    def __init__(self, hparams: maml_config.MamlHyperParameters,
                 sample_task: Callable[[maml_api.Stage], maml_api.MamlTask], model: maml_api.MamlModel,
                 device: torch.device, do_use_mlflow: bool = False,
                 n_val_iters: int = 10,
                 log_val_loss_every_n_episodes: int = 100,
                 log_model_every_n_episodes: int = 1000,
                 *args, **kwargs):
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
        super().__init__(*args, **kwargs)
        self.hparams: maml_config.MamlHyperParameters = hparams
        self.sample_task: Callable[[maml_api.Stage], maml_api.MamlTask] = sample_task
        self.model: maml_api.MamlModel = model
        self.device: torch.device = device
        self.do_use_mlflow: bool = do_use_mlflow  # TODO remove
        self.n_val_iters: int = n_val_iters
        self.log_val_loss_every_n_episodes: int = log_val_loss_every_n_episodes
        self.log_model_every_n_episodes: int = log_model_every_n_episodes
        # the first 10 percent of episodes will use strong multi-step loss updates, afterward weak
        self.multi_step_loss_n_episodes: int = int(hparams.n_episodes * 0.1) or 1
        # the first 30 percent of episodes will use first order updates
        self.first_order_updates_n_episodes: int = int(hparams.n_episodes * 0.3)
        example_params, example_buffers = self.model.get_state()
        self.inner_optimizer = maml_inner_optimizers.LSLRGradientDescentLearningRule(
            example_params=example_params, inner_steps=self.hparams.inner_steps, init_lr=self.hparams.alpha,
            use_learnable_learning_rates=True, device=self.device
        )
        self.inner_buffers = self.initialize_buffers(example_buffers)
        self.current_episode: int = 0
        self.logger = maml_logging.Logger()

    def log_buffers(self, episode: int):
        for i in range(self.hparams.inner_steps):
            for n, b in self.inner_buffers[i].items():
                self.logger.log_metric('step{}--'.format(i) + n, b.sum().item(), episode)

    def initialize_buffers(self, example_buffers: maml_api.NamedBuffers) -> dict[int, dict[str, torch.Tensor]]:
        zeroed_buffers = dict()
        for n, b in example_buffers.items():
            zeroed_buffers[n] = torch.zeros_like(b, dtype=b.dtype, device=b.device)

        # create a copy of the zeroed_buffers dict for each inner step
        return {i: {n: b.clone().detach() for n, b in zeroed_buffers.items()}
                for i in range(self.hparams.inner_steps)}

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
            loss_weights[-1] = 1 - loss_weights[0] * (len(loss_weights)-1)

        loss_weights = torch.Tensor(loss_weights).to(device=self.device)
        return loss_weights

    def inner_step(self, x_support: torch.Tensor, y_support: torch.Tensor,
                   params: maml_api.NamedParams, task: maml_api.MamlTask,
                   num_step: int, stage: maml_api.Stage) -> maml_api.NamedParams:
        # TODO add anil back
        y_hat = self.model.func_forward(x_support, params, self.inner_buffers[num_step])
        train_loss = task.calc_loss(y_hat, y_support, stage, maml_api.SetToSetType.SUPPORT)

        self.logger.log_metric("second_order_true", int(self.current_episode > self.first_order_updates_n_episodes),
                               self.current_episode)
        grads = autograd.grad(train_loss, params.values(),
                              create_graph=(stage == maml_api.Stage.TRAIN
                                            and self.current_episode > self.first_order_updates_n_episodes))
        names_grads_dict = dict(zip(params.keys(), grads))

        return self.inner_optimizer.update_params(params, names_grads_dict, num_step)
        # return {n: p - self.hparams.alpha *
        #               g for (n, p), g in zip(params.items(), grads)}

    def meta_forward(self, params: maml_api.NamedParams, buffers: maml_api.NamedBuffers, stage: maml_api.Stage):
        """Does a meta forward pass

        It first samples a task, then finetunes the model parameters on its support set, then calculates a loss
        using its target set. This loss has to be backpropagateable if the stage is TRAIN.
        """
        # TODO the buffers are used only during target loss calculation.
        #  This is not a problem, but they should be saved as attribute of the class instead?
        #  In any case, they should be saved, as should the inner_buffers
        task = self.sample_task(stage)
        x_support, y_support = task.sample()
        x_target, y_target = task.sample()

        per_step_loss_importance_vector = self.get_per_step_loss_importance_vector(stage)
        params_i = {n: p for n, p in params.items()}
        loss = torch.tensor(0.0, device=self.device)

        # i symbolizes the i-th step
        for i in range(self.hparams.inner_steps):
            # finetune params
            params_i = self.inner_step(x_support, y_support, params_i, task, i, stage)

            # calculate target loss using new params
            target_loss = task.calc_loss(self.model.func_forward(x_target, params_i, buffers), y_target,
                                         stage, maml_api.SetToSetType.TARGET)
            target_loss *= per_step_loss_importance_vector[i]
            loss += target_loss

        return loss

    def run_training(self):
        # self.parameters() also includes the per layer per step learning rates if they are learnable
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.beta)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.hparams.n_episodes,
                                                         eta_min=self.hparams.min_beta)
        _, buffers = self.model.get_state()

        for episode in range(self.hparams.n_episodes):
            self.log_buffers(episode)
            self.current_episode = episode
            optimizer.zero_grad()
            params, _ = self.model.get_state()

            batch_loss = torch.tensor(0.0, device=self.device)
            for i in range(self.hparams.meta_batch_size):
                batch_loss += self.meta_forward(params, buffers, maml_api.Stage.TRAIN)

            batch_loss.backward()
            optimizer.step()

            # log current lr then step
            self.logger.log_metric("lr", lr_scheduler.get_last_lr()[0], episode)
            lr_scheduler.step()

            if self.do_use_mlflow:
                # log acc_loss
                self.logger.log_metric("batch_loss", batch_loss.item(), step=episode)

                # log eval_loss under condition
                if episode % self.log_val_loss_every_n_episodes == 0 or episode == self.hparams.n_episodes - 1:
                    end_params, _ = self.model.get_state()
                    # TODO take mean across multiple runs?
                    # TODO the val_loss shouldn't just be the meta_forward loss ?
                    val_loss = self.meta_forward(end_params, buffers, maml_api.Stage.VAL)
                    self.logger.log_metric("val_loss", val_loss.item(), step=episode)

                # log model under condition
                if episode % self.log_model_every_n_episodes == 0 or episode == self.hparams.n_episodes - 1:
                    # TrainingStage passed to sample_task shouldn't play a role here
                    example_x = self.sample_task(maml_api.Stage.TRAIN).sample()[0].cpu().numpy()
                    mlflow.pytorch.log_model(copy.deepcopy(self.model).cpu(), f'models/ep{episode}', input_example=example_x)

                self.logger.log_buffer_to_mlflow(episode)

            # '0' temporary. ideal: val_loss.item())
            default_end_of_ep_fct(params, buffers, episode, batch_loss.item(), 0)

        # TODO put buffers back into model?


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
