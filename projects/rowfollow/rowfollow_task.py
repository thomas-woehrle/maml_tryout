import ast
import os

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import maml_api
from projects.rowfollow import rowfollow_utils as utils


def l1_distance_argmax_sum(y_hat, y, reduction='mean'):
    # Ensure y_hat and y have the same batch size
    assert y_hat.shape == y.shape, "y_hat and y must have the same shape"

    # Reshape y and y_hat to (N, 3, h*w) for argmax calculation
    N, C, H, W = y.shape
    y = y.view(N, C, -1)
    y_hat = y_hat.view(N, C, -1)

    # Get the indices of the max values in each heatmap
    y_argmax = torch.argmax(y, dim=2)
    y_hat_argmax = torch.argmax(y_hat, dim=2)

    # Convert the flat indices back to 2D coordinates
    y_coords = torch.stack([y_argmax // W, y_argmax % W], dim=2)
    y_hat_coords = torch.stack([y_hat_argmax // W, y_hat_argmax % W], dim=2)

    # Calculate the L1 distance (Manhattan distance) between the coordinates
    l1_distances = torch.abs(y_coords - y_hat_coords).sum(dim=2)

    # Sum the L1 distances across all heatmaps for each item in the batch
    batch_l1_distances = l1_distances.sum(dim=1).to(dtype=torch.float32)

    # Apply reduction (mean or sum) across the batch
    if reduction == 'mean':
        return batch_l1_distances.mean()
    elif reduction == 'sum':
        return batch_l1_distances.sum()
    else:
        raise ValueError("Invalid reduction method. Use 'mean' or 'sum'.")


class RowfollowTask(maml_api.MamlTask):
    """Task used in the case of rowfollow. One task represents one day (as of 07/25/24)"""

    def __init__(self, bag_path: str, k: int, device: torch.device, sigma: int = 10, seed=None):
        """
        Args:
            bag_path: Path to the bag/run_configs for this task.
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

    def sample(self) -> tuple[torch.Tensor, torch.Tensor]:
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

    def calc_loss(self, y_hat: torch.Tensor, y: torch.Tensor,
                  stage: maml_api.Stage, sts_type: maml_api.SetToSetType) -> torch.Tensor:
        """See also description of MamlTask

        Args:
            y_hat: Should be passed as raw model output 
            y: Should be the target as probabilities
            stage: ...
            sts_type: ...

        Returns:
            KL-divergence loss of y_hat and y
        """
        if stage == maml_api.Stage.VAL and sts_type == maml_api.SetToSetType.TARGET:
            return l1_distance_argmax_sum(y_hat, y)
        else:
            y_hat = F.log_softmax(y_hat.view(*y_hat.size()[:2], -1), 2).view_as(y_hat)

            return (self._loss_fct(y_hat[:, 0], y[:, 0])
                    + self._loss_fct(y_hat[:, 1], y[:, 1])
                    + self._loss_fct(y_hat[:, 2], y[:, 2]))


