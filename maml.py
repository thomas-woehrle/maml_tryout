import torch
import torch.nn as nn
import torch.nn.functional as F
from task import OmniglotTask
from torch.autograd import grad


def forward_old(net: nn.Sequential, theta, x):
    for layer_idx, param in enumerate(net.parameters()):
        # NOTE does this break backpropagation ? -> no should work
        # NOTE THIS BREAKS THE COMPUTATION GRAPH, I WAS FOOLED
        param.data = theta[layer_idx]

    return net.forward(x)


def forward(x, theta):
    # Convolutional Layers with Batch Normalization
    x = F.conv2d(x, theta[0], theta[1])
    x = F.batch_norm(
        x, None, None, theta[2], theta[3], momentum=1, training=True)
    x = F.relu(x)
    x = F.max_pool2d(x, 2, 2)

    x = F.conv2d(x, theta[4], theta[5])
    x = F.batch_norm(
        x, None, None, theta[6], theta[7], momentum=1, training=True)
    x = F.relu(x)
    x = F.max_pool2d(x, 2, 2)

    x = F.conv2d(x, theta[8], theta[9])
    x = F.batch_norm(
        x, None, None, theta[10], theta[11], momentum=1, training=True)
    x = F.relu(x)
    x = F.max_pool2d(x, 2, 2)

    # Flatten and Linear Layer
    x = torch.flatten(x, start_dim=1)
    x = F.linear(x, theta[12], theta[13])

    return x


def compute_adapted_theta(net: nn.Sequential, theta, task: OmniglotTask, k: int, alpha, inner_gradient_steps):
    # theta = [p.clone().detach().requires_grad_(True) for p in theta]
    inner_gradient_steps = 1  # NOTE assumption for now
    x, y = task.sample(k)
    x_hat = forward(x, theta)
    train_loss = task.loss_fct(x_hat, y)
    grads = grad(train_loss, theta, create_graph=False)
    # create_graph=True should enable second order
    theta_task = [p - alpha * g for p, g in zip(theta, grads)]
    return theta_task


# Hyperparameters
num_episodes = 100
meta_batch_size = 32  # number of tasks sampled each episode
n = 5  # n-way
k = 1  # k-shot
# size of D_i. number of data points to train on per task. Will be batch processed (?)
# the paper does not mention whether the size of D_i' has to be the same, but anything else wouldnt make sense imo
inner_gradient_steps = 1  # gradient steps done in inner loop during training
# NOTE not entirely sure how multiple gradient steps are handled
alpha, beta = 0.4, 0.001

net = nn.Sequential(
    nn.Conv2d(3, 64, 3),
    nn.BatchNorm2d(64, momentum=1, affine=True, track_running_stats=False),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(64, 64, 3),
    nn.BatchNorm2d(64, momentum=1, affine=True, track_running_stats=False),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(64, 64, 3),
    nn.BatchNorm2d(64, momentum=1, affine=True, track_running_stats=False),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2, 2),
    nn.Flatten(),
    nn.Linear(64, n)
)
# NOTE not sure how to handle train vs eval -> batchnorm

# randomly_initialize parameters
theta = [p for p in net.parameters()]  # should be list of tensors

for _ in range(num_episodes):
    acc_meta_update = (torch.zeros_like(p) for p in theta)
    acc_loss = 0
    for i in range(meta_batch_size):
        task = OmniglotTask('train', n)
        theta_i = compute_adapted_theta(net,
                                        theta, task, k, alpha, inner_gradient_steps)
        # this function has to sample k datapoints, batch process them with the given theta,
        # calculate the loss, calculate a gradient of the params wrt this loss and
        # update and return this theta. In addition, it has to be possible to backpropagate through this theta
        x_test, y_test = task.sample(k)
        test_loss = task.loss_fct(forward(x_test, theta_i), y_test)
        acc_loss += test_loss.item()
        # I think no create_graph=True is needed
        grads = grad(test_loss, theta)
        acc_meta_update = [current_update +
                           g for current_update, g in zip(acc_meta_update, grads)]

    theta = [p - beta * upd for p, upd in zip(theta, acc_meta_update)]
    print(acc_loss)
