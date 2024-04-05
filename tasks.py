import ast
import cv2
import os
import pandas as pd
import random
import torch
import torch.nn as nn
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


class RowfollowTask(MamlTask):
    def __init__(self, bag_path: str, k: int):
        # NOTE in self-supervised version, the image names should be a property as well
        self.bag_path = bag_path
        self.k = k
        df_left = pd.read_csv(
            os.path.join(bag_path, 'left_cam', 'labels.csv'))
        df_right = pd.read_csv(
            os.path.join(bag_path, 'right_cam', 'labels.csv'))
        self.labels = pd.concat([df_left, df_right], keys=[
                                'left_cam', 'right_cam'], names=['cam_side']).reset_index(level=0).reset_index(drop=True)
        self._loss_fct = nn.KLDivLoss(reduction='batchmean')

    def sample(self, mode) -> tuple[torch.Tensor, torch.Tensor]:
        import matplotlib.pyplot as plt
        # NOTE for supervised version, the mode does not play a role
        samples = self.labels.sample(self.k)
        x = []
        y = []

        for idx, sample in samples.iterrows():
            image_path = os.path.join(
                self.bag_path, sample.cam_side, sample.image_name)
            vp, ll, lr = ast.literal_eval(sample.vp), ast.literal_eval(
                sample.ll), ast.literal_eval(sample.lr)
            pre_processed_image, _ = pre_process_image(image_path)
            pre_processed_image = torch.from_numpy(
                pre_processed_image)  # TODO add device
            x.append(pre_processed_image)
            # this can be passed as is to the model as input x
            vp_gt = gaussian_heatmap(vp)
            ll_gt = gaussian_heatmap(ll)
            lr_gt = gaussian_heatmap(lr)
            y.append(torch.stack([vp_gt, ll_gt, lr_gt]))

        x = torch.stack(x)
        y = torch.stack(y)

        return x, y  # TODO add device handling

    def calc_loss(self, x_hat: torch.Tensor, y: torch.Tensor, mode):
        return super().calc_loss(x_hat, y, mode)
