import cv2
import matplotlib.pyplot as plt
import os
import random
import sys
import torch
import torch.nn as nn
from learn2learn import clone_module
from omniglot_helper import get_all_chars
from omniglot_net import OmniglotNet
from torchvision import transforms
from typing import List

omniglot_chars = get_all_chars()
random.shuffle(omniglot_chars)  # in-place shuffle
train_chars = omniglot_chars[:1200]
val_chars = omniglot_chars[1200:]


def get_task(dataset, n) -> List[str]:
    if dataset == 'train':
        return random.sample(train_chars, k=n)
    elif dataset == 'val':
        return random.sample(val_chars, k=n)
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


n_episodes = ...
meta_batch_size = 32
n = 5
k = 1
alpha, beta = 0.4, 0.001  # learning rates during training
# TODO find out real beta

criterion = nn.CrossEntropyLoss(reduction='sum')  # same for every task
optim = ...
meta_model = OmniglotNet(n)
meta_optimizer = optim.SGD(meta_model.parameters(), beta)

for i in range(n_episodes):
    meta_loss = 0
    for j in range(meta_batch_size):
        train_task = get_task('train', n)
        x, y = generate_k_samples_from_task(train_task, k)
        # x and y have batch_size of n*k
        # technically, get_task and generate_k_samples_from_task could easily be put into one function. However,
        # this approach sticks closer to the original concept of a task that generates samples

        task_model = clone_module(meta_model)  # pseuodcode
        task_optimizer = optim.SGD(task_model.parameters(), alpha)
        loss = criterion(task_model(x), y)
        loss.backward()  # this should update gradients in task_model, but realisitically will only update it in meta_model?
        task_optimizer.step()
        test_task = get_task('val', n)
        x, y = generate_k_samples_from_task(test_task, k)
        meta_loss += criterion(task_model(x), y)

    meta_optimizer.zero_grad()
    meta_loss.backward()
    meta_optimizer.step()
