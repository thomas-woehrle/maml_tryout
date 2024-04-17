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


def print_loss_info(losses: list, title):
    print(title)
    print('Mean:', losses.mean(dim=0))
    print('Variance:', losses.var(dim=0))
    print('Std dev.:', torch.sqrt(losses.var(dim=0)))


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


def calc_loss(ckpt_path, bag_path, device, seed, no_finetuning, k, inner_gradient_steps, alpha):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = RowfollowModel()
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)

    anil = ckpt.get('anil', False)
    sigma_scheduling = ckpt.get('sigma_scheduling', False)

    if not no_finetuning:
        params, buffers = model.get_initial_state()
        if sigma_scheduling:
            num_episodes = ckpt.get('num_episodes', 60_000)
            current_ep = num_episodes - 1  # NOTE ??
            task = RowfollowTask(
                bag_path, k, device=device, sigma_scheduling=sigma_scheduling, num_episodes=num_episodes, seed=seed)
            params = inner_loop_update_for_testing(anil, current_ep,
                                                   model, params, buffers, task, alpha, inner_gradient_steps)
        else:
            task = RowfollowTask(bag_path, k, device, seed=seed)
            params = inner_loop_update_for_testing(anil,
                                                   model, params, buffers, task, alpha, inner_gradient_steps)
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
    return loss, loss_on_frame, torch.mean(loss, dim=0), torch.mean(loss_on_frame, 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt_path', metavar='', type=str,
                        help='Path to the trained model parameters')
    parser.add_argument('bag_path')
    parser.add_argument('--n_runs', default=10, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--k', default=5, type=int)
    parser.add_argument('--inner_gradient_steps', default=1, type=int)
    parser.add_argument('--alpha', default=0.4, type=float)
    parser.add_argument('--no_finetuning', default=False, type=bool)
    args = parser.parse_args()

    losses = []  # n_runsxNx3
    losses_on_frame = []  # n_runsxNx3
    mean_losses = []  # n_runsx3, where the Nx3 were reduced to 3 by taking the mean. means per run are stored here
    mean_losses_on_frame = []  # n_runsx3
    for i in range(args.n_runs):
        print('Run', i+1, 'out of', args.n_runs, 'starting...')
        loss, loss_on_frame, mean_loss, mean_loss_on_frame = calc_loss(ckpt_path=args.ckpt_path,
                                                                       bag_path=args.bag_path,
                                                                       device=torch.device(
                                                                           'cpu'),
                                                                       # maybe device make this variable
                                                                       seed=args.seed+i,
                                                                       no_finetuning=args.no_finetuning,
                                                                       k=args.k,
                                                                       inner_gradient_steps=args.inner_gradient_steps,
                                                                       alpha=args.alpha)
        losses.append(loss)
        losses_on_frame.append(loss_on_frame)
        mean_losses.append(mean_loss)
        mean_losses_on_frame.append(mean_loss_on_frame)

    losses = torch.cat(losses)
    losses_on_frame = torch.cat(losses_on_frame)
    mean_losses = torch.stack(mean_losses)
    mean_losses_on_frame = torch.stack(mean_losses_on_frame)
    print(losses.shape)
    print(losses_on_frame.shape)
    print(mean_losses.shape)
    print(mean_losses_on_frame.shape)
    print_loss_info(losses, 'Keypoints - raw - across runs')
    print_loss_info(losses_on_frame, 'Keypoints - on frame - across runs')
    print_loss_info(mean_losses, 'Keypoints - raw - between runs')
    print_loss_info(mean_losses_on_frame,
                    'Keypoints - on frame - between runs')
