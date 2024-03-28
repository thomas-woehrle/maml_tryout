import torch
import torch.nn as nn
import torch.nn.functional as F

# adapted from https://github.com/katerakelly/pytorch-maml/blob/master/src/omniglot_net.py


class OmniglotModel(nn.Module):
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
            nn.Linear(64, num_classes)
        )
        # NOTE in theory this net might only be used to define the structure and get the initial params.
        # One thing to note is that the Batchnorms don't track any running stats, since as mentioned these layers
        # are not actually used for training. The running stats are tracked in the attributes below

        self.running_mean_1 = torch.zeros(64)
        self.running_var_1 = torch.ones(64)
        self.running_mean_5 = torch.zeros(64)
        self.running_var_5 = torch.ones(64)
        self.running_mean_9 = torch.zeros(64)
        self.running_var_9 = torch.ones(64)

    def forward(self, x, theta=None, is_train=True):
        if theta is None:
            return self.net(x)
        else:
            # Convolutional Layers with Batch Normalization
            x = F.conv2d(x, theta[0], theta[1])
            x = F.batch_norm(
                x, self.running_mean_1, self.running_var_1, theta[2], theta[3], training=is_train)
            x = F.relu(x)
            x = F.max_pool2d(x, 2, 2)

            x = F.conv2d(x, theta[4], theta[5])
            x = F.batch_norm(
                x, self.running_mean_5, self.running_var_5, theta[6], theta[7], training=is_train)
            x = F.relu(x)
            x = F.max_pool2d(x, 2, 2)

            x = F.conv2d(x, theta[8], theta[9])
            x = F.batch_norm(
                x, self.running_mean_9, self.running_var_9, theta[10], theta[11], training=is_train)
            x = F.relu(x)
            x = F.max_pool2d(x, 2, 2)

            # Flatten and Linear Layer
            x = torch.flatten(x, start_dim=1)
            x = F.linear(x, theta[12], theta[13])

            return x
