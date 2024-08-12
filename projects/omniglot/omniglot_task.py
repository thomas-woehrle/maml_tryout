import os
import random

import cv2
import torch
import torch.nn as nn
from torchvision import transforms

import maml_api

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),  # also changes to C x H x W
    transforms.Resize((28, 28), antialias=True)
    # antialias=True, because warning message
])


class OmniglotTask(maml_api.MamlTask):
    def __init__(self, chars: list[str], k: int, device):
        self.chars = chars
        self.k = k
        self.device = device
        self._loss_fct = nn.CrossEntropyLoss(
            reduction='sum')  # NOTE reduction=... ?

    def sample(self):
        x = []  # will be transformed to a tensor later
        y = []  # same here
        for i, char in enumerate(self.chars):
            file_names = random.sample(os.listdir(char), k=self.k)
            for fn in file_names:
                img = cv2.imread(os.path.join(char, fn))
                img = TRANSFORM(img)
                x.append(img)
                y.append(i)

        x = torch.stack(x)
        y = torch.tensor(y)

        return x.to(self.device), y.to(self.device)

    def calc_loss(self, y_hat: torch.Tensor, y: torch.Tensor,
                  stage: maml_api.Stage, sts_type: maml_api.SetToSetType):
        return self._loss_fct(y_hat, y)
