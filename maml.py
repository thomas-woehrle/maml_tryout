import torch
import torch.optim as optim
from torch.autograd import grad
from interfaces import MamlModel, MamlTask
from typing import Callable


def std_log(episode, loss):
    if episode % 1 == 0:
        print(f'{episode}: {loss}')


def inner_loop_update_for_testing(anil, model: MamlModel, params, buffers, task: MamlTask, alpha, inner_gradient_steps, current_ep=-1):
    x_support, y_support = task.sample('support', current_ep)
    params_i = {n: p for n, p in params.items()}
    for i in range(inner_gradient_steps):
        x_hat = model.func_forward(x_support, params_i, buffers)
        train_loss = task.calc_loss(x_hat, y_support, 'support')
        if anil:
            head = {n: p for n, p in params_i.items() if n.startswith('head')
                    }  # NOTE assumes that head is assigned via self.head = ...
            head_grads = grad(train_loss, head.values(), create_graph=False)
            # create_graph=True should enable second order, also leads to slower execution
            head_i = {n: p - alpha *
                      g for (n, p), g in zip(head.items(), head_grads)}
            params_i = {**params_i, **head_i}
        else:
            grads = grad(train_loss, params_i.values(), create_graph=False)
            # create_graph=True should enable second order, also leads to slower execution
            params_i = {n: p - alpha *
                        g for (n, p), g in zip(params_i.items(), grads)}
    return params_i


def inner_loop_update(anil, current_ep, model: MamlModel, params, buffers, task: MamlTask, alpha, inner_gradient_steps):
    inner_gradient_steps = 1  # NOTE assumption for now
    x_support, y_support = task.sample('support', current_ep)
    x_hat = model.func_forward(x_support, params, buffers)
    train_loss = task.calc_loss(x_hat, y_support, 'support')
    if anil:
        head = {n: p for n, p in params.items() if n.startswith('head')
                }  # NOTE assumes that head is assigned via self.head = ...
        head_grads = grad(train_loss, head.values(), create_graph=True)
        # create_graph=True should enable second order, also leads to slower execution
        head_i = {n: p - alpha *
                  g for (n, p), g in zip(head.items(), head_grads)}
        params_i = {**params, **head_i}
    else:
        grads = grad(train_loss, params.values(), create_graph=True)
        # create_graph=True should enable second order, also leads to slower execution
        params_i = {n: p - alpha *
                    g for (n, p), g in zip(params.items(), grads)}
    return params_i


def maml_learn(anil, num_episodes: int, meta_batch_size: int, inner_gradient_steps: int, alpha: float, beta: float,
               sample_task: Callable[[], MamlTask], model: MamlModel, checkpoint_fct,
               episode_logger: Callable[[int, int], any] = std_log):
    optimizer = optim.Adam(model.parameters(), lr=beta)

    _, buffers = model.get_initial_state()

    for episode in range(num_episodes):
        params, _ = model.get_initial_state()
        # acc_meta_update = {n: torch.zeros_like(p) for n, p in params.items()}
        acc_loss = 0
        for i in range(meta_batch_size):
            task = sample_task()
            params_i = inner_loop_update(anil, episode, model,
                                         params, buffers, task, alpha, inner_gradient_steps)
            x_query, y_query = task.sample('query', episode)
            test_loss = task.calc_loss(
                model.func_forward(x_query, params_i, buffers), y_query, 'query')
            acc_loss += test_loss
            # grads = grad(test_loss, params.values())
            # acc_meta_update = {n: current_update +
            #                   g for (n, current_update), g in zip(acc_meta_update.items(), grads)}

        # params = {n: p - beta *
        #           upd for (n, p), (_, upd) in zip(params.items(), acc_meta_update.items())}
        acc_loss.backward()
        optimizer.step()
        checkpoint_fct(params, buffers, episode, acc_loss)
        episode_logger(episode, acc_loss)


# TODO meaningful typing
# TODO meaningful inheritance
# TODO I should think about also passing a MetaOptimizer
# TODO create uniform config parameters type across train and test
# TODO for rowfollow rename boolean variables
