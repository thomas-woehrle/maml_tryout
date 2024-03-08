import cv2
import matplotlib.pyplot as plt
import os
import random
import sys
import torch
from omniglot_helper import get_all_chars
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


task = get_task('train', 5)
x, y = generate_k_samples_from_task(task, 5)
print(x.shape)
print(y.shape)

sys.exit(0)

n_episodes = ...
meta_batch_size = 32
n = ...
k = ...
a, b = ...  # learning rates during training

criterion = ...  # same for every task
optim = ...
meta_model = ...
meta_optimizer = optim.SGD(meta_model.parameters, b)

task_specific_params = dict()
task_meta_samples = dict()

for i in range(n_episodes):
    meta_loss = 0
    for j in range(meta_batch_size):
        train_task = get_task('train', n)  # determine the characters ?
        # x and y have to have batch_size of nxk
        x, y = generate_k_samples_from_task(train_task, k)
        loss = criterion(meta_model(x), y)
        # the gradients of meta_model are utilized in the inner loop as well
        meta_optimizer.zero_grad()
        loss.backward()
        theta_prime = ...
        # just clone the old model via the learn2learn clone function ?
        for parameter in meta_model:
            theta_prime[parameter.name] = parameter - a * parameter.grad
        task_model = ...  # create from theta_prime
        test_task = get_task(test_dataset)
        x, y = generate_k_samples_from_task(test_task, k)
        meta_loss += criterion(task_model(x), y)

    meta_optimizer.zero_grad()
    meta_loss.backward()
    meta_optimizer.step()
