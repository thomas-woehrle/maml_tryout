import datetime
import os
import torch
import torch.nn as nn
from models import OmniglotModel
from tasks import OmniglotTask, test_chars
from torch.autograd import grad


def get_checkpoint_dir():
    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    checkpoint_dir = './checkpoints/' + formatted_time
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir


# TODO rework checkpointing into model
def save_ckpt(theta, model, ckpt_path):
    state_dict = {n: p for (n, _), p in zip(model.named_parameters(), theta)}
    # this makes it possible to not have to use the named_parameters everywhere
    # relies on the order structure .named_parameters() returns
    state_dict['net.1.running_mean'] = model.running_mean_1
    state_dict['net.1.running_var'] = model.running_var_1
    state_dict['net.5.running_mean'] = model.running_mean_5
    state_dict['net.5.running_var'] = model.running_var_5
    state_dict['net.9.running_mean'] = model.running_mean_9
    state_dict['net.9.running_var'] = model.running_var_9
    torch.save(
        {
            'model_state_dict': state_dict,
            # TODO rework saving of test_chars vs train_chars at train time -> see also comment in tasks.py
            'test_chars': test_chars
        }, ckpt_path
    )


def compute_adapted_theta(model: nn.Module, theta, task: OmniglotTask, k: int, alpha, inner_gradient_steps):
    inner_gradient_steps = 1  # NOTE assumption for now
    x, y = task.sample(k, device=next(model.parameters()).device)
    # get same device as model parameters, assuming all parameters are on same device
    x_hat = model.forward(x, theta)
    train_loss = task.loss_fct(x_hat, y)
    grads = grad(train_loss, theta, create_graph=True)
    # create_graph=True should enable second order, also leads to slower execution
    theta_task = [p - alpha * g for p, g in zip(theta, grads)]
    return theta_task


# Hyperparameters
device = 'cpu'
ckpt_saving_freq = 1000  # specifies an amount of episodes
checkpoint_dir = get_checkpoint_dir()  # directory where .pt files are saved to
num_episodes = 60000
meta_batch_size = 32  # number of tasks sampled each episode
n = 5  # n-way
k = 1  # k-shot
inner_gradient_steps = 1  # gradient steps done in inner loop during training
alpha, beta = 0.4, 0.001  # learning rates

model = OmniglotModel(n)
model.to(device)

# randomly_initialize parameters
theta = [p for p in model.parameters()]  # should be list of tensors
# TODO let this be done by model.init_params() or similar

for episode in range(num_episodes):
    acc_meta_update = (torch.zeros_like(p) for p in theta)
    acc_loss = 0
    for i in range(meta_batch_size):
        task = OmniglotTask('train', n)
        theta_i = compute_adapted_theta(model,
                                        theta, task, k, alpha, inner_gradient_steps)
        x_test, y_test = task.sample(k, device)
        test_loss = task.loss_fct(model.forward(x_test, theta_i), y_test)
        acc_loss += test_loss.item()
        # I think no create_graph=True is needed here
        grads = grad(test_loss, theta)
        acc_meta_update = [current_update +
                           g for current_update, g in zip(acc_meta_update, grads)]

    theta = [p - beta * upd for p, upd in zip(theta, acc_meta_update)]
    if episode % 100 == 0:
        print(episode, ':', acc_loss)
    if episode % ckpt_saving_freq == 0 or episode == num_episodes - 1:
        ckpt_path = os.path.join(checkpoint_dir, f'ep{
                                 episode}_loss{acc_loss}.pt')
        save_ckpt(theta, model, ckpt_path)
