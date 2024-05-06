from typing import Callable

import torch.optim as optim
from torch import autograd

import maml_config
import maml_api


def std_log(episode: int, loss: float):
    if episode % 1 == 0:
        print(f'{episode}: {loss}')


def inner_loop_update_for_testing(anil, model: maml_api.MamlModel, params, buffers, task: maml_api.MamlTask, alpha, inner_gradient_steps, current_ep=-1):
    # NOTE ONLY TEMPORARY
    x_support, y_support = task.sample(maml_api.SampleMode.SUPPORT, current_ep)
    params_i = {n: p for n, p in params.items()}
    for i in range(inner_gradient_steps):
        x_hat = model.func_forward(x_support, params_i, buffers)
        train_loss = task.calc_loss(
            x_hat, y_support, maml_api.SampleMode.SUPPORT, current_ep)
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


def inner_loop_update(use_anil: bool, current_ep: int, model: maml_api.MamlModel, params, buffers, task: maml_api.MamlTask, alpha: float, inner_gradient_steps: int):
    inner_gradient_steps = 1  # NOTE assumption for now
    mode = maml_api.SampleMode.SUPPORT
    x_support, y_support = task.sample(mode, current_ep)
    x_hat = model.func_forward(x_support, params, buffers)
    train_loss = task.calc_loss(x_hat, y_support, mode, current_ep)
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
          sample_task: Callable[[], maml_api.MamlTask], model: maml_api.MamlModel, checkpoint_fct,
          episode_logger: Callable[[int, float], None] = std_log):
    """Executes the MAML training loop

    Args:
        hparams: Hyperparameters relevant to MAML. 
        sample_task: Function used to sample tasks i.e. x and y in the supervised case
        model: Model to train
        checkpoint_fct: Checkpoint function called after every episode
        episode_logger: Logging function called after every episode. Defaults to std_log.
    """
    optimizer = optim.SGD(model.parameters(), lr=hparams.beta)
    _, buffers = model.get_state()

    for episode in range(hparams.n_episodes):
        optimizer.zero_grad()
        params, _ = model.get_state()
        acc_loss = 0.0

        for i in range(hparams.meta_batch_size):
            mode = maml_api.SampleMode.QUERY
            task = sample_task()

            params_i = inner_loop_update(hparams.use_anil, episode, model,
                                         params, buffers, task, hparams.alpha, hparams.inner_gradient_steps)

            # Meta update
            x_query, y_query = task.sample(mode, episode)
            test_loss = task.calc_loss(
                model.func_forward(x_query, params_i, buffers), y_query, 'query', episode)
            acc_loss += test_loss

        acc_loss.backward()
        optimizer.step()
        checkpoint_fct(params, buffers, episode, acc_loss)
        episode_logger(episode, acc_loss)
