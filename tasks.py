import ast
import cv2
import os
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from interfaces import MamlTask
from rowfollow_utils import pre_process_image, gaussian_heatmap
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
        self._loss_fct = nn.CrossEntropyLoss(
            reduction='sum')  # NOTE reduction=... ?

    def sample(self, mode):
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

    def calc_loss(self, x_hat: torch.Tensor, y: torch.Tensor, mode):
        return self._loss_fct(x_hat, y)


class RowfollowTask(MamlTask):
    def __init__(self, bag_path: str, k: int, device: torch.device, sigma: int = 10, sigma_scheduling: bool = False, num_episodes: int = -1, seed: int | None = None):
        # NOTE in self-supervised version, the image names should be a property as well
        # TODO make sigma_scheduling yes/no a parameter
        self.bag_path = bag_path
        self.k = k
        df_left = pd.read_csv(
            os.path.join(bag_path, 'left_cam', 'labels.csv'))
        df_right = pd.read_csv(
            os.path.join(bag_path, 'right_cam', 'labels.csv'))
        self.labels = pd.concat([df_left, df_right], keys=[
                                'left_cam', 'right_cam'], names=['cam_side']).reset_index(level=0).reset_index(drop=True)
        self.seed = seed
        self._loss_fct = nn.KLDivLoss(reduction='batchmean')
        self.device = device
        self._num_episodes = num_episodes
        self.sigma = sigma
        self.sigma_scheduling = sigma_scheduling
        self.START_SIGMA = 30
        self.END_SIGMA = 1

    def sample(self, mode, current_ep: int = -1) -> tuple[torch.Tensor, torch.Tensor]:
        # NOTE for supervised version, the mode does not play a role
        """
        current_ep : only neede if self.sigma_scheduling is True
        """
        if self.sigma_scheduling:
            sig = self.START_SIGMA \
                - (self.START_SIGMA - self.END_SIGMA) \
                * (current_ep / (0.9 * self._num_episodes))
            sig = self.END_SIGMA if sig < self.END_SIGMA else sig
        else:
            sig = self.sigma
        samples = self.labels.sample(self.k, random_state=self.seed)
        x = []
        y = []

        for idx, sample in samples.iterrows():
            image_path = os.path.join(
                self.bag_path, sample.cam_side, sample.image_name)
            vp, ll, lr = ast.literal_eval(sample.vp), ast.literal_eval(
                sample.ll), ast.literal_eval(sample.lr)
            pre_processed_image, _ = pre_process_image(image_path)
            pre_processed_image = torch.from_numpy(
                pre_processed_image)
            x.append(pre_processed_image)
            # this can be passed as is to the model as input x
            vp_gt = gaussian_heatmap(vp, sig=sig)
            ll_gt = gaussian_heatmap(ll, sig=sig)
            lr_gt = gaussian_heatmap(lr, sig=sig)
            y.append(torch.stack([vp_gt, ll_gt, lr_gt]))

        x = torch.stack(x)
        y = torch.stack(y)

        return x.to(self.device), y.to(self.device)

    def calc_loss(self, x_hat: torch.Tensor, y: torch.Tensor, mode):
        """
        x_hat: Should be passed as logits, as the model outputs it, i.e. no softmax applied
        y: Is expected as distribution. sample() returns it as such atm
        """
        # NOTE for supervised version, the mode does not play a role
        x_hat = F.log_softmax(x_hat.view(
            *x_hat.size()[:2], -1), 2).view_as(x_hat)
        return self._loss_fct(x_hat[:, 0], y[:, 0]) + self._loss_fct(x_hat[:, 1], y[:, 1]) + self._loss_fct(x_hat[:, 2], y[:, 2])
