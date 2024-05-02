import ast
import cv2
import os
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from maml_api import MamlTask
from rowfollow_utils import pre_process_image, gaussian_heatmap
from torchvision import transforms
from maml_api import SampleMode

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

    def sample(self, mode: SampleMode, current_ep: int):
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

    def calc_loss(self, y_hat: torch.Tensor, y: torch.Tensor, mode: SampleMode, current_ep: int):
        return self._loss_fct(y_hat, y)


class RowfollowTask(MamlTask):
    def __init__(self, bag_path: str, k: int, device: torch.device, sigma: int = 10, seed=None):
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

    def sample(self, mode: SampleMode, current_ep: int) -> tuple[torch.Tensor, torch.Tensor]:
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
            vp_gt = gaussian_heatmap(vp, sig=self.sigma)
            ll_gt = gaussian_heatmap(ll, sig=self.sigma)
            lr_gt = gaussian_heatmap(lr, sig=self.sigma)
            y.append(torch.stack([vp_gt, ll_gt, lr_gt]))

        x = torch.stack(x)
        y = torch.stack(y)

        return x.to(self.device), y.to(self.device)

    def calc_loss(self, y_hat: torch.Tensor, y: torch.Tensor, mode: SampleMode, current_ep: int):
        """
        x_hat: Should be passed as logits, as the model outputs it, i.e. no softmax applied
        y: Is expected as distribution. sample() returns it as such atm
        """
        # NOTE for supervised version, the mode does not play a role
        y_hat = F.log_softmax(y_hat.view(
            *y_hat.size()[:2], -1), 2).view_as(y_hat)
        return self._loss_fct(y_hat[:, 0], y[:, 0]) + self._loss_fct(y_hat[:, 1], y[:, 1]) + self._loss_fct(y_hat[:, 2], y[:, 2])
