import torch.nn as nn

from maml_train import maml_api


class OmniglotModel(maml_api.MamlModel):
    # adapted from https://github.com/katerakelly/pytorch-maml/blob/master/src/omniglot_net.py
    def __init__(self, num_classes):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
        )
        self.head = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.net(x)
        x = self.head(x)
        return x
