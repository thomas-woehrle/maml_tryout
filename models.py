import torch
import torch.nn as nn
import torch.nn.functional as F

# adapted from https://github.com/katerakelly/pytorch-maml/blob/master/src/omniglot_net.py


class OmniglotModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, theta=None):
        if theta is None:
            return self.net(x)
        else:
            # NOTE I have to rework the train=True and running_mean and running_var part
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
