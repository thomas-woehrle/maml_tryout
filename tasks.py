import ast
import os
import random

import cv2
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import io
from torchvision import transforms
from torchvision import tv_tensors
from torchvision.transforms import v2

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


def turn_bounding_boxes_into_y(bounding_boxes_list: list[tv_tensors.BoundingBoxes], sig: int):
    y = []

    for bounding_boxes in bounding_boxes_list:
        vp = bounding_boxes[0][:2].detach().numpy()
        ll = bounding_boxes[1][:2].detach().numpy()
        lr = bounding_boxes[2][:2].detach().numpy()
        vp_gt = utils.dist_from_keypoint(vp, sig=sig, downscale=4)
        ll_gt = utils.dist_from_keypoint(ll, sig=sig, downscale=4)
        lr_gt = utils.dist_from_keypoint(lr, sig=sig, downscale=4)
        y.append(torch.stack([vp_gt, ll_gt, lr_gt]))

    return torch.stack(y)


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

        img_transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),  # divides by 255
            v2.Resize((224, 320)),
        ])

        transform = v2.Compose([
            # Geometric
            # v2.RandomRotation(degrees=(0, 45)),
            # v2.RandomPerspective(distortion_scale=0.5, p=0.5),
            # v2.RandomApply([v2.ElasticTransform(alpha=100.0)], p=0.5),
            # v2.RandomVerticalFlip(),

            # Photometric
            # v2.RandomApply([v2.Grayscale()], p=0.5),
            # v2.RandomApply([v2.ColorJitter(0.5, 0.25, 0.25, 0.1)], 0.5),
            # v2.RandomInvert(),
            # v2.RandomPosterize(bits=3),
            # v2.RandomSolarize(threshold=0.5),
            # v2.RandomEqualize(),

            v2.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            v2.ClampBoundingBoxes()
        ])

        for idx, sample in samples.iterrows():
            image_path = os.path.join(
                self.bag_path, sample.cam_side, sample.image_name)
            img = io.read_image(image_path)  # C x H x W
            img = img[:, 10:448+10, 96:640+96]
            x.append(img)
            vp, ll, lr = ast.literal_eval(sample.vp), ast.literal_eval(
                sample.ll), ast.literal_eval(sample.lr)
            y.append(
                tv_tensors.BoundingBoxes([
                    [*vp, *vp],
                    [*ll, *ll],
                    [*lr, *lr]
                ],
                    format="XYXY", canvas_size=(224, 320))
            )

        x = torch.stack(x)
        x = img_transform(x)

        x, y = transform(x, y)
        # y at this point is still a list of bounding boxes
        y = turn_bounding_boxes_into_y(y, sig=self.sigma)

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
