import cv2
import os
import random
import torch
import torch.nn as nn
from torchvision import transforms
from omniglot_helper import get_all_chars
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


TRANSFORM = transforms.Compose([
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
            img = TRANSFORM(img)
            x.append(img)
            y.append(i)

    x = torch.stack(x)
    y = torch.tensor(y)

    return x, y


class OmniglotTask():
    def __init__(self, train_or_test: str, n: int):
        self.loss_fct = nn.CrossEntropyLoss(reduction='sum')
        # use reduction='mean' instead ?

        if train_or_test == 'train':
            self.chars = random.sample(train_chars, k=n)
        elif train_or_test == 'test':
            self.chars = random.sample(test_chars, k=n)
        else:
            raise ValueError('Argument invalid: ' + train_or_test)

    def sample(self, k: int, device='cpu') -> tuple[torch.Tensor, torch.Tensor]:
        x = []  # will be transformed to a tensor later
        y = []  # same here
        for i, char in enumerate(self.chars):
            file_names = random.sample(os.listdir(char), k=k)
            for fn in file_names:
                img = cv2.imread(os.path.join(char, fn))
                img = TRANSFORM(img)
                x.append(img)
                y.append(i)

        x = torch.stack(x)
        y = torch.tensor(y)

        return x.to(device), y.to(device)
