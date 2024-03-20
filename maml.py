import cv2
import matplotlib.pyplot as plt
import os
import random
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from alt_omniglot_net import OmniglotNet as AltOmniglotNet
from collections import OrderedDict
from omniglot_helper import get_all_chars, test_sync
from omniglot_net import OmniglotNet
from torchvision import transforms
from typing import List

omniglot_chars = get_all_chars()
random.shuffle(omniglot_chars)  # in-place shuffle
train_chars = omniglot_chars[:1200]
val_chars = omniglot_chars[1200:]


def get_task(dataset: str, n: int) -> List[str]:
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


n_episodes = 1000
meta_batch_size = 32
n = 5
k = 1
alpha, beta = 0.4, 0.1  # learning rates during training
# TODO find out real beta

criterion = nn.CrossEntropyLoss(reduction='sum')  # same for every task
meta_model = AltOmniglotNet(n)
meta_optimizer = optim.SGD(meta_model.parameters(), lr=beta)

for i in range(n_episodes):
    meta_loss = 0
    for j in range(meta_batch_size):
        train_task = get_task('train', n)
        x, y = generate_k_samples_from_task(train_task, k)
        # x and y have batch_size of n*k
        # technically, get_task and generate_k_samples_from_task could easily be put into one function. However,
        # this approach sticks closer to the original concept of a task that generates samples

        # inspiration taken from https://github.com/katerakelly/pytorch-maml/blob/master/src/inner_loop.py
        meta_optimizer.zero_grad()
        loss = criterion(meta_model.forward(x), y)
        loss.backward()
        # NOTE slightly different architecture needed if more than one update is made
        task_theta = OrderedDict((name, param - alpha*param.grad)
                                 for (name, param) in meta_model.named_parameters())
        val_task = get_task('val', n)
        x_val, y_val = generate_k_samples_from_task(val_task, k)
        logits = meta_model.forward(x_val, weights=task_theta)

        x_hat = torch.argmax(logits, dim=1)
        if i == 80:
            for img, (idx, cls_hat), cls in zip(x_val, enumerate(x_hat), y_val):
                img = img.permute(1, 2, 0)
                img = img.numpy()
                plt.imshow(img)
                plt.title('logits:' + str(logits[idx]) + ' \nargmax:' +
                          str(cls_hat.item()) + '\nreal:' + str(cls.item()))
                plt.tight_layout()
                plt.show()

        task_val_loss = criterion(logits, y_val)
        meta_loss += task_val_loss

    print(i)
    print(meta_loss.item())
    meta_optimizer.zero_grad()
    meta_loss.backward()
    meta_optimizer.step()
