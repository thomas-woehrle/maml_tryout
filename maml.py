import copy
from typing import Callable, Optional

import mlflow
import torch.optim as optim
from torch import autograd

import maml_config
import maml_api


def inner_loop_update_for_testing(anil, model: maml_api.MamlModel, params, buffers, task: maml_api.MamlTask, alpha,
                                  inner_gradient_steps):
    # NOTE ONLY TEMPORARY
    x_support, y_support = task.sample()
    params_i = {n: p for n, p in params.items()}
    for i in range(inner_gradient_steps):
        x_hat = model.func_forward(x_support, params_i, buffers)
        train_loss = task.calc_loss(
            x_hat, y_support)
        if anil:
            head = {n: p for n, p in params_i.items() if n.startswith('head')
                    }  # NOTE assumes that head is assigned via self.head = ...
            head_grads = autograd.grad(
                train_loss, head.values(), create_graph=False)
            # create_graph=True should enable second order, also leads to slower execution
            head_i = {n: p - alpha *
                      g for (n, p), g in zip(head.items(), head_grads)}
            params_i = {**params_i, **head_i}
        else:
            grads = autograd.grad(
                train_loss, params_i.values(), create_graph=False)
            # create_graph=True should enable second order, also leads to slower execution
            params_i = {n: p - alpha *
                        g for (n, p), g in zip(params_i.items(), grads)}
    return params_i


def inner_loop_update(use_anil: bool, model: maml_api.MamlModel, params, buffers, task: maml_api.MamlTask, alpha: float,
                      inner_gradient_steps: int):
    inner_gradient_steps = 1  # NOTE assumption for now
    x_support, y_support = task.sample()
    x_hat = model.func_forward(x_support, params, buffers)
    train_loss = task.calc_loss(x_hat, y_support)
    if use_anil:
        head = {n: p for n, p in params.items() if n.startswith('head')
                }  # NOTE assumes that head is assigned via self.head = ...
        head_grads = autograd.grad(
            train_loss, head.values(), create_graph=True)
        # create_graph=True should enable second order, also leads to slower execution
        head_i = {n: p - alpha *
                  g for (n, p), g in zip(head.items(), head_grads)}
        params_i = {**params, **head_i}
    else:
        grads = autograd.grad(train_loss, params.values(), create_graph=True)
        params_i = {n: p - alpha *
                    g for (n, p), g in zip(params.items(), grads)}
    return params_i


def train(hparams: maml_config.MamlHyperParameters,
          sample_task: Callable[[maml_api.TrainingStage], maml_api.MamlTask], model: maml_api.MamlModel,
          end_of_episode_fct: Optional[Callable] = None,
          do_use_mlflow: bool = False, log_model_every_n_episodes: int = 1000):
    """Executes the MAML training loop

    Args:
        hparams: Hyperparameters relevant for MAML.
        sample_task: Function used to sample tasks i.e. x and y in the supervised case
        model: Model to train
        end_of_episode_fct: Function called at the end of an episode.
                            Gets passed the parameters, buffers, episode, acc_loss, eval_loss.
        do_use_mlflow: Inidactes whether mlflow should be used
        log_model_every_n_episodes: Frequency of model logging. First and last will always be logged. (Default: 1000)
    """
    optimizer = optim.SGD(model.parameters(), lr=hparams.beta)
    _, buffers = model.get_state()

    for episode in range(hparams.n_episodes):
        optimizer.zero_grad()
        params, _ = model.get_state()
        # Accumulated loss. Will become tensor and this way device will be inferred instead of specified.
        acc_loss = 0.0

        for i in range(hparams.meta_batch_size):
            task = sample_task(maml_api.TrainingStage.TRAIN)

            params_i = inner_loop_update(hparams.use_anil, model,
                                         params, buffers, task, hparams.alpha, hparams.inner_gradient_steps)

            # Meta update
            x_query, y_query = task.sample()
            query_loss = task.calc_loss(
                model.func_forward(x_query, params_i, buffers), y_query)
            acc_loss += query_loss

        acc_loss.backward()
        optimizer.step()

        # calculate evaluation loss
        eval_task = sample_task(maml_api.TrainingStage.EVAL)
        episode_end_params, _ = model.get_state()
        eval_params = inner_loop_update_for_testing(hparams.use_anil, model, episode_end_params, buffers, eval_task,
                                                    hparams.alpha, hparams.inner_gradient_steps)
        eval_x, eval_y = eval_task.sample()
        eval_loss = eval_task.calc_loss(model.func_forward(eval_x, eval_params, buffers), eval_y)

        if do_use_mlflow:
            mlflow.log_metric("acc_loss", acc_loss.item(), step=episode)
            mlflow.log_metric("eval_loss", eval_loss.item(), step=episode)
            if episode % log_model_every_n_episodes == 0 or episode == hparams.n_episodes - 1:
                # TrainingStage passed to sample_task shouldn't play a role here
                example_x = sample_task(maml_api.TrainingStage.TRAIN).sample()[0].cpu().numpy()
                mlflow.pytorch.log_model(copy.deepcopy(model).cpu(), f'models/ep{episode}', input_example=example_x)

        if end_of_episode_fct is not None:
            end_of_episode_fct(params, buffers, episode, acc_loss, eval_loss)
