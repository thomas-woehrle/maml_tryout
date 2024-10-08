import copy
import os
import random
import time
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


def set_seed(seed: int):
    # Python's built-in random module
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # PyTorch Backends
    torch.backends.cudnn.deterministic = True  # Forces deterministic behavior
    torch.backends.cudnn.benchmark = False


class MamlTrainer(nn.Module):
    def __init__(self, hparams: maml_config.MamlHyperParameters,
                 sample_task: Callable[[maml_api.Stage], maml_api.MamlTask], model: maml_api.MamlModel,
                 device: torch.device, do_use_mlflow: bool,
                 train_config: maml_config.TrainConfig,
                 calc_val_loss_fct: Callable[[int, maml_api.MamlModel,
                                              maml_api.InnerBuffers, maml_api.InnerLrs, maml_logging.Logger], None],
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
            calc_val_loss_fct: Function which can be used to calculate the val loss at a given episode.
        """
        super().__init__(*args, **kwargs)
        self.hparams: maml_config.MamlHyperParameters = hparams
        self.sample_task: Callable[[maml_api.Stage], maml_api.MamlTask] = sample_task
        self.model: maml_api.MamlModel = model
        self.device: torch.device = device
        self.do_use_mlflow: bool = do_use_mlflow  # TODO remove

        self.log_val_loss_every_n_episodes: int = train_config.log_val_loss_every_n_episodes
        self.log_model_every_n_episodes: int = train_config.log_model_every_n_episodes

        # number of episodes that will use strong multi-step loss updates, afterward weak
        self.multi_step_loss_n_episodes: int = int(hparams.n_episodes * hparams.msl_percentage_of_episodes) or 1
        # number of episodes that will use first order updates
        self.first_order_updates_n_episodes: int = int(hparams.n_episodes *
                                                       hparams.first_order_percentage_of_episodes) or 1

        example_params, example_buffers = self.model.get_state()
        # inner_optimizer is created in any case, but only used when hparams.use_lslr is True
        self.inner_optimizer = maml_inner_optimizers.LSLRGradientDescentLearningRule(
            example_params=example_params, inner_steps=self.hparams.inner_steps, init_lr=self.hparams.alpha,
            use_learnable_learning_rates=True, device=self.device
        )
        self.inner_buffers = self.initialize_inner_buffers(example_buffers)

        self.current_episode: int = 0
        self.logger = maml_logging.Logger()
        self.calc_val_loss_fct = calc_val_loss_fct

    def log_buffers(self, episode: int):
        for i in range(self.hparams.inner_steps):
            for n, b in self.inner_buffers[i].items():
                self.logger.log_metric('step{}--'.format(i) + n, b.sum().item(), episode)

        _, meta_buffers = self.model.get_state()
        for n, b in meta_buffers.items():
            self.logger.log_metric('metabuffer--' + n, b.sum().item(), episode)

    def initialize_inner_buffers(self, example_buffers: maml_api.NamedBuffers) -> dict[int, dict[str, torch.Tensor]]:
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
        # If Val stage or msl shouldn't be used, then only the last step loss is important
        if not self.hparams.use_msl or stage == maml_api.Stage.VAL:
            loss_weights = np.zeros(self.hparams.inner_steps, dtype=np.float32)
            loss_weights[-1] = 1.0
        else:
            # TODO currently steps 0 to N - 2 (second to last) have the same importance -> should this be the case?
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
        y_hat = self.model.func_forward(x_support, params, self.inner_buffers[num_step])
        train_loss = task.calc_loss(y_hat, y_support, stage, maml_api.SetToSetType.SUPPORT)

        # determine whether to use second order
        use_second_order = maml_api.Stage.TRAIN and (not self.hparams.use_da
                                                     or self.current_episode > self.first_order_updates_n_episodes)
        self.logger.log_metric("second_order_true", int(use_second_order), self.current_episode)

        if self.hparams.use_anil:
            # assumes that head is assigned via self.head = ...
            head = {n: p for n, p in params.items() if n.startswith('head')}

            # get grads only wrt head
            head_grads = autograd.grad(
                train_loss, head.values(), create_graph=use_second_order)

            # do update only for head
            if self.hparams.use_lslr:
                names_head_grads_dict = dict(zip(head.keys(), head_grads))
                # inner_optimizer.update_params does not care about anil or no-anil
                head_i = self.inner_optimizer.update_params(head, names_head_grads_dict, num_step)
                return {**params, **head_i}  # old params, with only head replaced
            else:
                head_i = {n: p - self.hparams.alpha * g for (n, p), g in zip(head.items(), head_grads)}
                return {**params, **head_i}
        else:
            grads = autograd.grad(train_loss, params.values(),
                                  create_graph=use_second_order)

            # update params
            if self.hparams.use_lslr:
                names_grads_dict = dict(zip(params.keys(), grads))
                return self.inner_optimizer.update_params(params, names_grads_dict, num_step)
            else:
                return {n: p - self.hparams.alpha * g for (n, p), g in zip(params.items(), grads)}

    def meta_forward(self, params: maml_api.NamedParams, buffers: maml_api.NamedBuffers, stage: maml_api.Stage):
        """Does a meta forward pass

        It first samples a task, then finetunes the model parameters on its support set, then calculates a loss
        using its target set. This loss has to be backpropagateable if the stage is TRAIN.

        The buffers will only be updated during the pass used for calculation of the target loss, ie after finetuning
        has happened. This is a feature not a bug, since they represent the distribution of the data passing through a
        model after finetuning has happened. These buffers can be used for evaluating later on.
        """
        task = self.sample_task(stage)
        x_support, y_support = task.sample(maml_api.SetToSetType.SUPPORT)
        x_target, y_target = task.sample(maml_api.SetToSetType.TARGET)

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
        if self.hparams.use_ca:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.hparams.n_episodes,
                                                                eta_min=self.hparams.min_beta)
        for episode in range(self.hparams.n_episodes):
            # self.log_buffers(episode)
            # self.inner_optimizer.log_lrs(episode, self.logger)
            self.current_episode = episode

            if self.do_use_mlflow:
                # TODO switch to self.logger instead to buffer log this? does this actually cost time?
                mlflow.log_metric('current_episode', episode)

            optimizer.zero_grad()
            params, buffers = self.model.get_state()

            batch_loss = torch.tensor(0.0, device=self.device)
            for i in range(self.hparams.meta_batch_size):
                batch_loss += self.meta_forward(params, buffers, maml_api.Stage.TRAIN)

            batch_loss.backward()
            optimizer.step()

            # log current lr then step
            if self.hparams.use_ca:
                self.logger.log_metric("lr", lr_scheduler.get_last_lr()[0], episode)
                lr_scheduler.step()

            if self.do_use_mlflow:
                # log batch_loss
                self.logger.log_metric("batch_loss", batch_loss.item(), step=episode)

                # log eval_loss under condition
                if episode % self.log_val_loss_every_n_episodes == 0 or episode == self.hparams.n_episodes - 1:
                    self.calc_val_loss_fct(episode, self.model,
                                           # turning int to str, because maml_eval expects this
                                           # TODO prettify this
                                           {str(n): b for n, b in self.inner_buffers.items()},
                                           {n: t for n, t in self.inner_optimizer.names_lrs_dict.items()}, self.logger)

                # log model under condition
                if episode % self.log_model_every_n_episodes == 0 or episode == self.hparams.n_episodes - 1:
                    print('\n ###### Saving model checkpoint #######')
                    # TrainingStage and SetToSetType passed to sample_task shouldn't play a role here
                    # example_x = (self.sample_task(maml_api.Stage.TRAIN).sample(maml_api.SetToSetType.SUPPORT)[0].
                    #             cpu().numpy())

                    timestamp = time.time_ns()
                    file_name = f'{timestamp}.pth'

                    print(f'Creating model checkpoint {file_name} ...')
                    torch.save(copy.deepcopy(self.model).cpu(), file_name)
                    # this makes problems when running on the cluster:
                    # mlflow.pytorch.log_model(copy.deepcopy(self.model).cpu(), f'ep{episode}/model',
                    #                         input_example=example_x)
                    print(f'Uploading model checkpoint {file_name} ...')
                    mlflow.log_artifact(file_name, f'ep{episode}')
                    print(f'Removing model checkpoint {file_name} locally...')
                    os.remove(file_name)

                    print('###### Model checkpoint saved ####### \n')

                    # TODO should this be logged even if hparams.use_bnrs and/or hparams.use_lslr is false?
                    self.logger.log_dict(self.inner_buffers, f'ep{episode}/inner_buffers.json')
                    #  {...} to turn nn.ParameterDict into normal dict
                    self.logger.log_dict({n: t for n, t in self.inner_optimizer.names_lrs_dict.items()},
                                         f'ep{episode}/inner_lrs.json')

                self.logger.log_metrics_buffer_to_mlflow(episode)

            print('batch_loss:', batch_loss.item())
