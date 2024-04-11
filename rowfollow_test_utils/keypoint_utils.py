import cv2
import numpy as np
from scipy.special import softmax
from typing import List


class Line:
    # from_coords and to_coords are tuples of (x, y)
    def __init__(self, from_coords, to_coords, color, thickness):
        self.x_from = from_coords[0]
        self.y_from = from_coords[1]
        self.x_to = to_coords[0]
        self.y_to = to_coords[1]
        self.color = color
        self.thickness = thickness


def normalize(input):
    # input is c,h,w which we change to w,h,c
    # isnt it h,w,c?
    input = input.transpose((1, 2, 0))

    complete_dist = input.copy()

    r = input[:, :, 0]
    g = input[:, :, 1]
    b = input[:, :, 2]

    r_min = r.min()
    g_min = g.min()
    b_min = b.min()

    r_max = r.max()
    g_max = g.max()
    b_max = b.max()

    # why different ordering ?
    complete_dist[:, :, 0] = (input[:, :, 2]-b_min)/(b_max-b_min)
    complete_dist[:, :, 1] = (input[:, :, 1]-g_min)/(g_max-g_min)
    complete_dist[:, :, 2] = (input[:, :, 0]-r_min)/(r_max-r_min)

    # output = np.where(output > 0.1, output, np.zeros_like(output))

    # TODO: refactor out of here and to wherever is called, but at the end to not break something.
    lr_dist = complete_dist[:, :, 0]
    ll_dist = complete_dist[:, :, 1]
    vp_dist = complete_dist[:, :, 2]

    return complete_dist, lr_dist, ll_dist, vp_dist


def get_dists(pred, resize_to=False):
    pred[0] = softmax(pred[0])
    pred[1] = softmax(pred[1])
    pred[2] = softmax(pred[2])

    complete_dist, lr_dist, ll_dist, vp_dist = normalize(
        pred)

    if resize_to:
        complete_dist = cv2.resize(complete_dist, resize_to)
        vp_dist = cv2.resize(vp_dist, resize_to)
        ll_dist = cv2.resize(ll_dist, resize_to)
        lr_dist = cv2.resize(lr_dist, resize_to)

    return complete_dist, vp_dist, ll_dist, lr_dist


def get_keypoints(pred, factor=1):
    vp_y, vp_x = np.unravel_index(pred[0].argmax(), pred[0].shape)
    ll_y, ll_x = np.unravel_index(pred[1].argmax(), pred[1].shape)
    lr_y, lr_x = np.unravel_index(pred[2].argmax(), pred[2].shape)

    return ((vp_x * factor, vp_y * factor), (ll_x * factor, ll_y * factor), (lr_x * factor, lr_y * factor))


def get_vp_conf(self, vp_pred):
    def sigmoid(x):
        sig = 1 / (1 + np.exp(-x))
        return sig

    vp_sig = np.sqrt(np.var(vp_pred))
    sigma = 1.0e-5

    vp_conf = sigmoid(3*np.tanh(5*(1-((sigma)/(5*vp_sig)))))
    return vp_conf


def img_with_heatmap(image, heatmap):
    # image and heatmap have to be same size
    pred_img = 0.5*cv2.cvtColor(image, cv2.COLOR_BGR2RGB)+0.5 * \
        (heatmap)*255

    return pred_img


def img_with_lines(image, lines: List[Line]):
    return_img = image.copy()
    for line in lines:
        return_img = cv2.line(return_img, (line.x_from, line.y_from),
                              (line.x_to, line.y_to), line.color, line.thickness)

    return return_img


def image_with_lines_from_pred(image, pred):
    vp_coords, ll_coords, lr_coords = get_keypoints(pred, factor=4)
    image_with_lines = cv2.circle(
        image, vp_coords, 5, (255, 0, 0), -1)

    lines = [Line(vp_coords, ll_coords, (0, 255, 0), 2),
             Line(vp_coords, lr_coords, (0, 0, 255), 2)]

    image_with_lines = img_with_lines(
        image_with_lines, lines)

    return image_with_lines


# factor has to be calculated cautiously
# finds the point that is
# - most down and right, left of the middle for left prediction
# - most down and left, right of the middle for right prediction
# - and at the same time above the specified theshold
def heuristic1(array, left_or_right, factor=4, threshold=0.08):
    row_range = range(array.shape[1] // 2 - 50, -1, -1) if left_or_right == 'left' else range(
        array.shape[1] // 2, array.shape[1], 1)

    # Iterate over the array in reverse order
    for y in range(array.shape[0] - 1, -1, -1):
        for x in row_range:
            if array[y, x] > threshold:
                return (x * factor, y * factor)

    return None


def heuristic1_ythenx(array, y_start, y_stop, y_step, x_start, x_stop, x_step, factor=4, threshold=0.08):
    for y in range(y_start, y_stop, y_step):
        for x in range(x_start, x_stop, x_step):
            if array[y, x] > threshold:
                return (x * factor, y * factor)

    return None


def heuristic1_xtheny(array, y_start, y_stop, y_step, x_start, x_stop, x_step, factor=4, threshold=0.08):
    for x in range(x_start, x_stop, x_step):
        for y in range(y_start, y_stop, y_step):
            if array[y, x] > threshold:
                return (x * factor, y * factor)

    return None


def create_gaussian_array(dimensions, middle_point, mean, sigma):
    height, width = dimensions
    y_middle, x_middle = middle_point

    # Create a grid of (x,y) coordinates
    x = np.arange(0, width, 1)
    y = np.arange(0, height, 1)
    x_grid, y_grid = np.meshgrid(x, y)

    # Calculate the 2D Gaussian
    gaussian = np.exp(-(((x_grid - x_middle) ** 2) / (2 * sigma **
                      2) + ((y_grid - y_middle) ** 2) / (2 * sigma ** 2)))

    # Adjust the Gaussian so that its peak is at 'mean'
    gaussian *= mean / gaussian.max()

    return gaussian


# createss a gaussian array with the mean located at
# - the highest value in the pixels 5 - 45 of the 56th row for left
# - the highest value in the pixels 35 - 60 of the 56th row for right
# assumes dimension (56, 80) for the array
def heuristic2(array, left_or_right, factor=0.95, sigma=4):
    mu = (array.max() - array.min()) * factor
    # highest value in the pixels 5 - 45 of the 56th row for left and 75 - 60 for right
    if left_or_right == 'left':
        mean_x = 5 + array[55][5:45].argmax()
    elif left_or_right == 'right':
        mean_x = 35 + array[55][35:75].argmax()

    gaussian = create_gaussian_array(
        array.shape, (55, mean_x), mu, sigma)

    return gaussian


# given distribution returns ll_heur, lr_heur, thresholds
# and the idx of the lowest threshold where < prctg*total_number_of_pixels of pixels are above this threshold
def heuristic3(dist, left_or_right, prctg=0.5):
    thresholds1 = [i/1000 for i in range(100, 40, -5)]
    thresholds2 = [i/1000 for i in range(40, 0, -1)]
    thresholds = thresholds1 + thresholds2

    limit = dist.shape[0] * dist.shape[1] * prctg

    last_map = dist
    last_th_idx = 0
    for idx, th in enumerate(thresholds):
        current_map = np.where(dist > th, 1, 0)
        if current_map.sum() > limit:
            break
        last_map = current_map
        last_th_idx = idx

    side = left_or_right  # has to be either 'left' or 'right'
    # y_start, y_stop, y_step = 223, 210, -1
    y_start, y_stop, y_step = 210, 223, 1
    if side == 'left':
        x_start, x_stop, x_step = 110, -1, -1
    elif side == 'right':
        x_start, x_stop, x_step = 210, 320, 1

    # the following lines would be an approach if the lines are too far inside
    # if last_th_idx > 0:
    #    last_map = np.where(dist > thresholds[last_th_idx - 1], 1, 0)

    heur = heuristic1_xtheny(
        last_map, y_start, y_stop, y_step, x_start, x_stop, x_step, factor=1, threshold=0.5)

    return heur, thresholds, last_th_idx


def heuristic4(dist, top_y, bottom_y, left_x, right_x):

    search_area = dist[top_y:bottom_y, left_x:right_x]
    # gives coordinates relative to search area and in y, x format
    heur = np.unravel_index(search_area.argmax(), search_area.shape)
    # turns into absolute values and x, y format
    heur = heur[1] + left_x, heur[0] + top_y
    return heur
