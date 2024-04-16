import argparse
import ast
import os

import torch
import torch.nn as nn

from maml import inner_loop_update_for_testing
from models import RowfollowModel
from rowfollow_utils import pre_process_image
from shared_utils import get_indices_from_pred, get_coordinates_on_frame
from tasks import RowfollowTask


def get_data(base_path, df, device):
    x = []
    y = []

    for idx, row in df.iterrows():
        image_path = os.path.join(
            base_path, row.cam_side, row.image_name)
        pre_processed_image, _ = pre_process_image(image_path)
        pre_processed_image = torch.from_numpy(pre_processed_image)
        x.append(pre_processed_image)
        vp, ll, lr = ast.literal_eval(row.vp), ast.literal_eval(
            row.ll), ast.literal_eval(row.lr)
        y.append(torch.tensor([vp, ll, lr]))

    x = torch.stack(x).to(device)
    y = torch.stack(y).to(device)

    return x, y


def calc_loss(ckpt_path, bag_path, device, seed):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = RowfollowModel()
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)

    k = 5
    num_episodes = ckpt.get('num_episodes', 60_000)
    current_ep = num_episodes - 1  # NOTE ??
    anil = ckpt.get('anil', False)
    inner_gradient_steps = 1

    task = RowfollowTask(bag_path, k, num_episodes, device, seed=seed)
    params, buffers = model.get_initial_state()
    params = inner_loop_update_for_testing(anil, current_ep,
                                           model, params, buffers, task, 0.4, inner_gradient_steps)
    model.load_state_dict(params | buffers)
    # model.eval() # NOTE why not needed/working?

    labels = task.labels
    loss_fct = nn.L1Loss(reduction='none')

    x, y = get_data(bag_path, labels, device)

    x_hat = get_indices_from_pred(
        model(x)) * 4

    x_hat_on_frame = []
    for pred in x_hat:
        vp = pred[0]
        ll = pred[1]
        lr = pred[2]
        vp = tuple(vp)
        ll = get_coordinates_on_frame(vp, ll)
        lr = get_coordinates_on_frame(vp, lr)
        x_hat_on_frame.append(torch.tensor([vp, ll, lr]))
    x_hat_on_frame = torch.stack(x_hat_on_frame)

    loss = loss_fct(x_hat, y)
    loss = torch.sum(loss, dim=2).float()
    loss_on_frame = loss_fct(x_hat_on_frame, y)
    loss_on_frame = torch.sum(loss_on_frame, dim=2).float()
    return loss, loss_on_frame


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt_path', metavar='', type=str,
                        help='Path to the trained model parameters')
    parser.add_argument('bag_path')
    parser.add_argument('--n_runs', default=10, type=int)
    # TODO add k, inner_gradient_steps as arguments
    args = parser.parse_args()
    seed = 0

    losses = []
    losses_on_frame = []
    for i in range(args.n_runs):
        print('Run', i+1, 'out of', args.n_runs, 'starting...')
        loss, loss_on_frame = calc_loss(ckpt_path=args.ckpt_path, bag_path=args.bag_path,
                                        device=torch.device('cpu'), seed=seed+i)
        losses.append(loss)
        losses_on_frame.append(loss_on_frame)

    losses = torch.cat(losses)
    losses_on_frame = torch.cat(losses_on_frame)
    print('Keypoints - Raw:')
    print('Mean:', losses.mean(dim=0))
    print('Variance:', losses.var(dim=0))
    print('Std dev.:', torch.sqrt(losses.var(dim=0)))
    print('Keypoints - On frame:')
    print('Mean:', losses_on_frame.mean(dim=0))
    print('Variance:', losses_on_frame.var(dim=0))
    print('Std dev.:', torch.sqrt(losses_on_frame.var(dim=0)))
