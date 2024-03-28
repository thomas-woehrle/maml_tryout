import torch
from torch.autograd import grad


def compute_adapted_theta(model: MamlModule, theta, task: MamlTask, alpha, inner_gradient_steps):
    inner_gradient_steps = 1  # NOTE assumption for now
    x, y = task.sample()
    # get same device as model parameters, assuming all parameters are on same device
    x_hat = model.forward(x, theta)
    train_loss = task.calc_loss(x_hat, y)
    grads = grad(train_loss, theta, create_graph=True)
    # create_graph=True should enable second order, also leads to slower execution
    theta_task = [p - alpha * g for p, g in zip(theta, grads)]
    return theta_task


# NOTE
# I think no device needs to be passed, cause the callee can determine the device
# inner_gradient_steps will be fixed at 1 for now
# I should think about also passing a MetaOptimizer
def maml_learn(num_episodes: int, meta_batch_size: int, inner_gradient_steps: int, alpha: float, beta: float,
               sample_task: callable[[], MamlTask], model: MamlModel, checkpoint_fct: callable[[int, int], ...]):
    theta = model.init_params()  # should be list of tensors

    for episode in range(num_episodes):
        acc_meta_update = (torch.zeros_like(p) for p in theta)
        acc_loss = 0
        for i in range(meta_batch_size):
            task = sample_task()
            theta_i = compute_adapted_theta(model,
                                            theta, task, alpha, inner_gradient_steps)
            x_test, y_test = task.sample()
            test_loss = task.calc_loss(model.forward(x_test, theta_i), y_test)
            acc_loss += test_loss.item()
            grads = grad(test_loss, theta)
            acc_meta_update = [current_update +
                               g for current_update, g in zip(acc_meta_update, grads)]

        theta = [p - beta * upd for p, upd in zip(theta, acc_meta_update)]
        checkpoint_fct(episode, acc_loss)
