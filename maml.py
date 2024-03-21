import copy
import cv2
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from alt_omniglot_net import OmniglotNet as AltOmniglotNet
from collections import OrderedDict
from omniglot_helper import get_all_chars, viz_logit
from torchvision import transforms
from typing import List

omniglot_chars = get_all_chars()
random.shuffle(omniglot_chars)  # in-place shuffle
train_chars = omniglot_chars[:1200]
test_chars = omniglot_chars[1200:]


def get_task(dataset: str, n: int) -> List[str]:
    if dataset == 'train':
        return random.sample(train_chars, k=n)
    elif dataset == 'test':
        return random.sample(test_chars, k=n)
    else:
        raise ValueError('dataset argument invalid: ' + dataset)


transform = transforms.Compose([
    transforms.ToTensor(),  # also changes to C x H x W
    transforms.Resize((28, 28), antialias=True)
    # antialias=True, because warning message
])


def generate_k_samples_from_task(task: List[str], k):
    x = []  # will be transformed to a tensor later
    y = []  # same here
    for i, char in enumerate(task):
        file_names = random.sample(os.listdir(char), k=k)
        for fn in file_names:
            img = cv2.imread(os.path.join(char, fn))
            img = transform(img)
            x.append(img)
            y.append(i)

    x = torch.stack(x)
    y = torch.tensor(y)

    return x, y


def load_named_parameters_in_model(model: nn.Module, named_parameters):
    """
    Works in-place.
    The two parts have to be compatible.
    """
    model_dict = model.state_dict()

    # Update with parameters from the original model
    for name, param in named_parameters:
        model_dict[name].copy_(param.data)

    # Load the updated state dictionary
    model.load_state_dict(model_dict)


n_episodes = 100
meta_batch_size = 32
n = 5
k = 1
alpha, beta = 0.4, 0.01  # learning rates during training
# TODO find out real beta

criterion = nn.CrossEntropyLoss(reduction='sum')  # same for every task
meta_model = AltOmniglotNet(n)

for episode in range(n_episodes):
    meta_loss = 0
    # initialize accumulated gradient to be a dictionary with keys corresponding to weights and values of 0 everywhere
    acc_grad = {name: 0 for name, param in meta_model.named_parameters()}

    for i in range(meta_batch_size):
        task_model = copy.deepcopy(meta_model)
        inner_optimizer = optim.SGD(task_model.parameters(), lr=alpha)

        task = get_task('train', n)
        x, y = generate_k_samples_from_task(task, k)
        train_loss = criterion(task_model(x), y)
        # x and y have batch_size of n*k
        # technically, get_task and generate_k_samples_from_task could easily be put into one function. However,
        # this approach sticks closer to the original concept of a task that generates samples

        # Inner loop update, currently only one step
        inner_optimizer.zero_grad()
        train_loss.backward()
        inner_optimizer.step()

        # Update meta loss
        x_test, y_test = generate_k_samples_from_task(task, k)
        logit = task_model(x_test)
        if episode == 20:
            viz_logit(x_test, y_test, torch.round(logit * 100))

        test_loss = criterion(logit, y_test)
        meta_loss += test_loss

        # Update grad accumulation used for meta update
        inner_optimizer.zero_grad()  # needed
        test_loss.backward()
        acc_grad = {name: acc_grad[name] + param.grad for name,
                    param in task_model.named_parameters()}

    print(episode)
    print(meta_loss.item())
    theta = {name: param - beta * acc_grad[name]
             for name, param in meta_model.named_parameters()}
    meta_model.load_state_dict(theta)
