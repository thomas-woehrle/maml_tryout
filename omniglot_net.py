import torch.nn as nn

# adapted from https://github.com/katerakelly/pytorch-maml/blob/master/src/omniglot_net.py


class OmniglotNet(nn.Module):
    def __init__(self, num_classes):
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
        # TODO softmax (?), CrossEntropyLoss epxects unnormalized logits as input.
        # It turns logits into probabilites if target is probabilities afaik

    def forward(self, x):
        return self.net(x)
