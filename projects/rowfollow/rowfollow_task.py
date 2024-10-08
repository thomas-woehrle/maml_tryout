import ast
import os
import random
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import maml_api
import rowfollow_utils as utils
from rf_utils import vision


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


def get_labels_df(bag_path: str, cam_side: str) -> pd.DataFrame:
    if cam_side == 'left_cam' or cam_side == 'right_cam':
        labels = pd.read_csv(os.path.join(bag_path, cam_side, 'labels.csv'))
        labels['cam_side'] = cam_side
    elif cam_side == 'both':
        df_left = pd.read_csv(os.path.join(bag_path, 'left_cam', 'labels.csv'))
        df_right = pd.read_csv(os.path.join(bag_path, 'right_cam', 'labels.csv'))
        labels = pd.concat([df_left, df_right],
                           keys=['left_cam', 'right_cam'],
                           names=['cam_side']).reset_index(level=0).reset_index(drop=True)
    else:
        raise ValueError('cam_side must be either "left_cam", "right_cam" or "both".')
    return labels


class RowfollowTask(maml_api.MamlTask):
    """Task used in the case of rowfollow. One task represents one day (as of 07/25/24)"""

    def __init__(self, support_bag_path: str,
                 k: int,
                 device: torch.device,
                 sigma: int = 10, seed=None,
                 support_cam_side: str = 'both',
                 target_bag_path: str = None,
                 target_cam_side: str = 'both'):
        """
        Args:
            support_bag_path: Path to the bag/run_configs for this task.
            k: Sample batch size.
            device: Device to be used.
            sigma: Sigma used to create the labels i.e. distributions from the keypoints. Defaults to 10.
            seed: Seed to be used for the sampling. No seed used if this is None. Defaults to None.
        """
        self.support_bag_path = support_bag_path
        self.support_labels = get_labels_df(self.support_bag_path, support_cam_side)
        # assign target_bag_path, which will be the same as support_bag_path if not supplied
        self.target_bag_path = target_bag_path or self.support_bag_path
        self.target_labels = get_labels_df(self.target_bag_path, target_cam_side)
        self.k = k

        self.seed = seed
        self._loss_fct = nn.KLDivLoss(reduction='batchmean')
        self.device = device
        self.sigma = sigma

    def sample(self, sts_type: maml_api.SetToSetType) -> tuple[torch.Tensor, torch.Tensor]:
        if sts_type == maml_api.SetToSetType.SUPPORT:
            labels = self.support_labels
            bag_path = self.support_bag_path
        elif sts_type == maml_api.SetToSetType.TARGET:
            labels = self.target_labels
            bag_path = self.target_bag_path
        else:
            raise ValueError('sts_type {} could not be matched'.format(str(sts_type)))

        samples = labels.sample(self.k, random_state=self.seed, replace=False)

        x = []
        y = []

        for idx, sample in samples.iterrows():
            image_path = os.path.join(bag_path, sample.cam_side, sample.image_name)
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


class RowfollowTaskOldDataset(maml_api.MamlTask):
    def __init__(self, annotations_file_path: str, support_data_path: str,
                 k: int, device: torch.device, sigma: int = 10,
                 target_data_path: Optional[str] = None, seed: Optional[int] = None):
        self.annotations: pd.DataFrame = pd.read_csv(annotations_file_path)
        self.support_data_path: str = support_data_path
        self.target_data_path: str = target_data_path or self.support_data_path
        self.k: int = k
        self.device: torch.device = device
        self.sigma: int = sigma
        self.seed: Optional[int] = seed
        if self.seed is not None:
            random.seed(self.seed)

        self.dataset_info_df = pd.read_csv(
            os.path.join('/'.join(self.support_data_path.split('/')[:-2]), 'dataset_info.csv')
        )
        self._loss_fct = nn.KLDivLoss(reduction='batchmean')

    @staticmethod
    def get_kps_for_image(image_name: str, annotations_df: Optional[pd.DataFrame] = None,
                          annotation_row: Optional[pd.Series] = None,
                          original_size: tuple[int, int] = (1280, 720),
                          new_size: tuple[int, int] = (320, 224),):
        if annotation_row is None:
            annotation_row = annotations_df[annotations_df['image_name'] == image_name].iloc[0]

        vp = np.array([annotation_row['X_VAN_Cords'], annotation_row['Y_VAN_Cords']], dtype=np.float32)
        ll = np.array([annotation_row['X_line_Left'], annotation_row['Y_line_Left']], dtype=np.float32)
        lr = np.array([annotation_row['X_line_Right'], annotation_row['Y_line_Right']], dtype=np.float32)
        ll = vision.get_coordinates_on_frame(vp, ll, original_size)
        lr = vision.get_coordinates_on_frame(vp, lr, original_size)

        downscale_x = new_size[0] / original_size[0]
        downscale_y = new_size[1] / original_size[1]

        vp *= [downscale_x, downscale_y]
        ll *= [downscale_x, downscale_y]
        lr *= [downscale_x, downscale_y]

        return vp, ll, lr

    def sample(self, sts_type: maml_api.SetToSetType) -> tuple[torch.Tensor, torch.Tensor]:
        data_path = self.support_data_path if sts_type == maml_api.SetToSetType.SUPPORT else self.target_data_path
        all_img_names = [f for f in os.listdir(data_path) if f.endswith('.jpg')]
        if self.seed is not None:
            all_img_names = sorted(all_img_names)
        img_names = random.sample(all_img_names, k=self.k)

        x = []
        y = []

        for img_name in img_names:
            image_path = os.path.join(data_path, img_name)

            if utils.get_collection_growth_stage(self.dataset_info_df, data_path.split('/')[-1]) == 'very late':
                original_size = (320, 224)
            else:
                original_size = (1280, 720)
            vp, ll, lr = self.get_kps_for_image(img_name, self.annotations, original_size=original_size)

            pre_processed_image, _ = utils.pre_process_image_old_data(image_path, new_size=(320, 224))
            pre_processed_image = torch.from_numpy(pre_processed_image)
            x.append(pre_processed_image)

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
