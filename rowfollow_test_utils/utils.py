import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import List


# NOTE does not work with subfolders
def get_filenames(dir_path):
    file_names = []

    # Iterate over all items in the folder
    for item in os.listdir(dir_path):
        if item.lower().endswith(('.jpg', '.png')):
            file_names.append(item)

    return file_names


"""
This is function for generic use that brings the pic to the approximately right ratio, without touching the height. 
Since all pictures in the current case have the same height and width, we can apply a static cropping and don't need this function.
"""


def cut_image_into_ratio(image, ratio):
    needed_width = (image.shape[0] * ratio[1]) // ratio[0]
    cut_width = (image.shape[1] - needed_width) // 2
    return image[:, cut_width:cut_width + needed_width]


def process_image(path_to_image):
    # Image is loaded in BGR format as np array
    image = cv2.imread(path_to_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # clipping to 320/224 ratio
    # cutting of 96 from left, right and 10 from top, bottom -> 640x448 dimension*
    image = image[10:448+10, 96:640+96]
    # then downsample
    image = cv2.resize(image, (320, 224), interpolation=cv2.INTER_AREA)

    data = image/255
    # NOTE: why do we use ImageNet normalization values?
    data[:, :, 0] = (data[:, :, 0]-0.485)/0.229
    data[:, :, 1] = (data[:, :, 1]-0.456)/0.224
    data[:, :, 2] = (data[:, :, 2]-0.406)/0.225
    data = np.expand_dims(data, axis=0)
    data = np.transpose(data, axes=[0, 3, 1, 2]).astype(np.float32)

    return data, image


class PlotInfo:
    # supply None to vmin and vmax to use the min and max of the array
    def __init__(self, array, isImage, vmin=None, vmax=None, gradientColor="gray", title=None):
        self.array = array
        self.isImage = isImage
        self.gradientColor = gradientColor
        self.title = title

        self.vmin = vmin
        if self.vmin is None:
            self.vmin = array.min()
        self.vmax = vmax
        if self.vmax is None:
            self.vmax = array.max()


def plot_multiple_color_maps(*arrays_of_plotinfos: List[PlotInfo], full_title="title", directory="output"):
    n_columns = len(arrays_of_plotinfos)  # one column for each array
    n_rows = len(arrays_of_plotinfos[0])  # assuming each array has same length
    fig, axes = plt.subplots(
        n_rows, n_columns, figsize=(5 * n_columns, 3 * n_rows))

    # Plot arrays for each color
    for row in range(n_rows):
        for column in range(n_columns):
            plotinfo = arrays_of_plotinfos[column][row]
            if plotinfo.isImage:
                axes[row, column].imshow(plotinfo.array)
            else:
                axes[row, column].imshow(plotinfo.array, cmap=mcolors.LinearSegmentedColormap.from_list(
                    "", ["white", plotinfo.gradientColor]), vmin=plotinfo.vmin, vmax=plotinfo.vmax)
            title = plotinfo.title or f"{row}, {column}"
            axes[row, column].set_title(title)

    plt.tight_layout(pad=3.0)
    # Adjust the top to make space for the suptitle
    plt.subplots_adjust(top=0.97)
    fig.suptitle(full_title, fontsize=16)

    i = 0
    base_filename = "plot"
    ext = ".pdf"
    filename = f"{base_filename}{ext}"
    os.makedirs(directory, exist_ok=True)

    filepath = os.path.join(directory, filename)

    # Check if the file exists and create a new filename if necessary
    while os.path.exists(filepath):
        i += 1
        filename = f"{base_filename}_{i}{ext}"
        filepath = os.path.join(directory, filename)

    plt.savefig(filepath)


def find_intersection(line1, line2):

    x1, y1 = line1[0]
    x2, y2 = line1[1]
    x3, y3 = line2[0]
    x4, y4 = line2[1]

    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if denominator == 0:
        # Lines are parallel, no intersection point
        return None

    intersection_x = ((x1 * y2 - y1 * x2) * (x3 - x4) -
                      (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
    intersection_y = ((x1 * y2 - y1 * x2) * (y3 - y4) -
                      (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator

    return (int(intersection_x), int(intersection_y))


def get_coordinates_on_frame(vp, kp, left_or_right, dim=(319, 223)):
    line1 = (vp, kp)
    line2 = ((0, dim[1]), (dim[0], dim[1]))
    intersect = find_intersection(line1, line2)
    if intersect == None:
        return kp

    if intersect[0] > dim[0] or intersect[0] < 0:
        x = 0 if left_or_right == 'left' else (
            dim[0] if left_or_right == 'right' else None)
        line2 = ((x, 0), (x, dim[1]))
        intersect = find_intersection(line1, line2)

    if intersect == None:
        return kp

    return intersect
