import torch
import torch.nn as nn
import torch.nn.functional as F
from interfaces import MamlModel
from torchvision import models


class OmniglotModel(MamlModel):
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

    def forward(self, x, theta, is_train=True):
        # TODO if is_train=False but the evaluation takes place on a new instance,
        # then the running_mean has to be taken from somewhere the savedtheta
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

    def get_init_params(self):
        return [p for p in self.parameters()]


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


class RowfollowModel(MamlModel):
    def __init__(self):
        super().__init__()

        resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        encoder = nn.Sequential(*list(resnet18.children())[:8])

        self.conv1 = encoder[0]
        self.bn1 = encoder[1]
        self.relu = encoder[2]
        self.maxpool = encoder[3]

        self.layer1 = encoder[4]
        self.layer2 = encoder[5]
        self.layer3 = encoder[6]
        self.layer4 = encoder[7]

        self.up1 = Up(512+256, 256, scale_factor=2)
        self.up2 = Up(256+128, 128, scale_factor=2)
        self.up3 = Up(128+64, 64)

        self.head = nn.Conv2d(64, 3, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x1 = self.maxpool(x)

        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        x_up = self.up1(x5, x4)
        x_up = self.up2(x_up, x3)
        x_up = self.up3(x_up, x2)
        x = self.head(x_up)

        return x

    def func_forward(self, x: torch.Tensor, params, buffers) -> torch.Tensor:
        return torch.func.functional_call(self, (params, buffers), x)

    def get_initial_state(self):
        params = {n: p for n, p in self.named_parameters()}
        buffers = {n: b for n, b in self.named_buffers()}
        return params, buffers
