import cv2
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F


def pre_process_image(path_to_image):
    # Image is loaded in BGR format as np array
    image = cv2.imread(path_to_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # clipping to 320/224 ratio
    # cutting of 96 from left, right and 10 from top, bottom -> 640x448 dimension*
    image = image[10:448+10, 96:640+96]
    # then downsample
    image = cv2.resize(image, (320, 224), interpolation=cv2.INTER_AREA)

    data = image/255
    # ImageNet Normalization because, ResNet trained on it
    data[:, :, 0] = (data[:, :, 0]-0.485)/0.229
    data[:, :, 1] = (data[:, :, 1]-0.456)/0.224
    data[:, :, 2] = (data[:, :, 2]-0.406)/0.225
    data = np.transpose(data, axes=[2, 0, 1]).astype(np.float32)

    return data, image


def pre_process_image_old_data(path_to_image, new_size=(320, 224)):
    # Image is loaded in BGR format as np array
    image = cv2.imread(path_to_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # resize image -> we don't worry about distortion here
    image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    data = image/255
    # ImageNet Normalization because, ResNet trained on it
    data[:, :, 0] = (data[:, :, 0]-0.485)/0.229
    data[:, :, 1] = (data[:, :, 1]-0.456)/0.224
    data[:, :, 2] = (data[:, :, 2]-0.406)/0.225
    data = np.transpose(data, axes=[2, 0, 1]).astype(np.float32)

    return data, image


def dist_from_keypoint(center: tuple[int, int], image_size: tuple[int, int] = (80, 56), sig: float = 10, downscale: float = 1):
    # adapted from https://stackoverflow.com/a/58621239
    """Creates a probability distribution from a keypoint, by applying a gaussian, then a softmax.

    Args:
        center: The keypoint and center of the gaussian.
        image_size: The image size. Defaults to (80, 56).
        sig: The sigma used for the gaussian. Defaults to 10.
        downscale: How the center should be downscaled. Defaults to 1.

    Returns:
        The distribution in the needed shape.
    """
    x_axis = np.linspace(0, image_size[0]-1,
                         image_size[0]) - center[0] / downscale
    y_axis = np.linspace(0, image_size[1]-1,
                         image_size[1]) - center[1] / downscale
    xx, yy = np.meshgrid(x_axis, y_axis)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
    kernel = torch.from_numpy(kernel)
    # TODO probably should split this function into two (because of SoC), but not now
    kernel = F.softmax(kernel.view(-1), dim=0).view_as(kernel)
    return kernel


def get_train_and_test_bags(directory, exclude_first_x, exclude_last_y):
    days = glob.glob(os.path.join(directory, '*/'))
    days = sorted(days)

    train_days = days[exclude_first_x:-exclude_last_y if exclude_last_y > 0 else None]
    test_days = days[:exclude_first_x] + days[-exclude_last_y:] if exclude_last_y > 0 else []

    train_days_bags = []
    test_days_bags = []

    for subdir in train_days:
        sub_subdirs = glob.glob(os.path.join(subdir, '*/'))
        train_days_bags.extend(sub_subdirs)

    # Step 4: List immediate sub-subdirectories for excluded subdirectories
    for subdir in test_days:
        sub_subdirs = glob.glob(os.path.join(subdir, '*/'))
        test_days_bags.extend(sub_subdirs)

    # Step 5: Return the lists of selected and excluded sub-subdirectories
    return train_days_bags, test_days_bags

