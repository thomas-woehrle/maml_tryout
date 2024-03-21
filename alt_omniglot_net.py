import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# Adapted from https://github.com/katerakelly/pytorch-maml


class OmniglotNet(nn.Module):
    '''
    The base model for few-shot learning on Omniglot
    '''

    def __init__(self, num_classes):
        super(OmniglotNet, self).__init__()
        # Define the network
        self.net = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 64, 3)),
            ('bn1', nn.BatchNorm2d(64, momentum=1, affine=True)),
            ('relu1', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool2d(2, 2)),
            ('conv2', nn.Conv2d(64, 64, 3)),
            ('bn2', nn.BatchNorm2d(64, momentum=1, affine=True)),
            ('relu2', nn.ReLU(inplace=True)),
            ('pool2', nn.MaxPool2d(2, 2)),
            ('conv3', nn.Conv2d(64, 64, 3)),
            ('bn3', nn.BatchNorm2d(64, momentum=1, affine=True)),
            ('relu3', nn.ReLU(inplace=True)),
            ('pool3', nn.MaxPool2d(2, 2)),
            ('flatten', nn.Flatten()),
            ('fc', nn.Linear(64, num_classes))
        ]))

        # Initialize weights
        # self._init_weights()

    def forward(self, x, weights=None):
        ''' Define what happens to data in the net '''
        if weights == None:
            return self.net(x)
        else:
            # print(weights['net.fc.weight'].shape)
            # print(weights['net.fc.bias'].shape)
            # TODO add meaningful running mean and var?
            running_mean = torch.zeros(64)
            running_var = torch.ones(64)
            x = F.conv2d(x, weights['net.conv1.weight'],
                         weights['net.conv1.bias'])
            x = F.batch_norm(x, running_mean=running_mean, training=True, running_var=running_var, weight=weights['net.bn1.weight'],
                             bias=weights['net.bn1.bias'])  # , momentum=1)
            x = F.relu(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)
            x = F.conv2d(x, weights['net.conv2.weight'],
                         weights['net.conv2.bias'])
            x = F.batch_norm(x, running_mean=running_mean, training=True, running_var=running_var, weight=weights['net.bn2.weight'],
                             bias=weights['net.bn2.bias'], momentum=1)
            x = F.relu(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)
            x = F.conv2d(x, weights['net.conv3.weight'],
                         weights['net.conv3.bias'])
            x = F.batch_norm(x, running_mean=running_mean, training=True, running_var=running_var, weight=weights['net.bn3.weight'],
                             bias=weights['net.bn3.bias'], momentum=1)
            x = F.relu(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)
            x = torch.squeeze(x)
            x = F.linear(x, weights['net.fc.weight'], weights['net.fc.bias'])
            return x
