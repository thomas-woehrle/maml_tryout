import ast
import os
import random

import cv2
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import maml_api
import rowfollow_utils as utils

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

    def sample(self, mode: maml_api.SampleMode, current_ep: int):
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

    def calc_loss(self, y_hat: torch.Tensor, y: torch.Tensor, mode: maml_api.SampleMode, current_ep: int):
        return self._loss_fct(y_hat, y)


class RowfollowTask(maml_api.MamlTask):
    """Task used in the case of rowfollow. One task represents one run"""

    def __init__(self, bag_path: str, k: int, device: torch.device, sigma: int = 10, seed=None):
        """
        Args:
            bag_path: Path to the bag/run for this task.
            k: Sample batch size.
            device: Device to be used.
            sigma: Sigma used to create the labels i.e. distributions from the keypoints. Defaults to 10.
            seed: Seed to be used for the sampling. No seed used if this is None. Defaults to None.
        """
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
        self.sigma = sigma

    def sample(self, mode: maml_api.SampleMode, current_ep: int) -> tuple[torch.Tensor, torch.Tensor]:
        samples = self.labels.sample(self.k, random_state=self.seed)
        x = []
        y = []

        for idx, sample in samples.iterrows():
            image_path = os.path.join(
                self.bag_path, sample.cam_side, sample.image_name)
            pre_processed_image, _ = utils.pre_process_image(image_path)
            pre_processed_image = torch.from_numpy(
                pre_processed_image)
            x.append(pre_processed_image)
            # this can be passed as is to the model as input x
            vp, ll, lr = ast.literal_eval(sample.vp), ast.literal_eval(
                sample.ll), ast.literal_eval(sample.lr)
            # vp, ll, lr are coordinates, but we need distributions
            vp_gt = utils.dist_from_keypoint(vp, sig=self.sigma, downscale=4)
            ll_gt = utils.dist_from_keypoint(ll, sig=self.sigma, downscale=4)
            lr_gt = utils.dist_from_keypoint(lr, sig=self.sigma, downscale=4)
            y.append(torch.stack([vp_gt, ll_gt, lr_gt]))

        x = torch.stack(x)
        y = torch.stack(y)

        return x.to(self.device), y.to(self.device)

    def calc_loss(self, y_hat: torch.Tensor, y: torch.Tensor, mode: maml_api.SampleMode, current_ep: int):
        """See also description of MamlTask

        Args:
            y_hat: Should be passed as raw model output 
            y: Should be the target as probabilities  

        Returns:
            KL-divergence loss of y_hat and y
        """
        y_hat = F.log_softmax(y_hat.view(
            *y_hat.size()[:2], -1), 2).view_as(y_hat)
        return self._loss_fct(y_hat[:, 0], y[:, 0]) + self._loss_fct(y_hat[:, 1], y[:, 1]) + self._loss_fct(y_hat[:, 2], y[:, 2])
