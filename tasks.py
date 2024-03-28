import cv2
import os
import random
import torch
import torch.nn as nn
from interfaces import MamlTask
from torchvision import transforms

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),  # also changes to C x H x W
    transforms.Resize((28, 28), antialias=True)
    # antialias=True, because warning message
])


class OmniglotTask(MamlTask):
    def __init__(self, chars: list[str], k: int, device):
        self.chars = chars
        self.k = k
        self.device = device
        self._loss_fct = nn.CrossEntropyLoss()  # NOTE reduction=... ?

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

    def calc_loss(self, x_hat: torch.Tensor, y: torch.Tensor):
        return self._loss_fct(x_hat, y)
