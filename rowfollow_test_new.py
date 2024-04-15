import argparse
import ast
import os

import torch
import torch.nn as nn

from maml import inner_loop_update_for_testing
from models import RowfollowModel
from rowfollow_utils import pre_process_image
from shared_utils import get_indices_from_pred
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


def main(ckpt_path, bag_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = RowfollowModel()
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)

    k = 5
    num_episodes = ckpt.get('num_episodes', 60_000)
    current_ep = num_episodes - 1  # NOTE ??
    anil = ckpt.get('anil', False)
    inner_gradient_steps = 1
    seed = 0

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
    loss = loss_fct(x_hat, y)
    loss = torch.sum(loss, dim=2).float()
    loss = torch.mean(loss, dim=0)
    print(loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt_path', metavar='', type=str,
                        help='Path to the trained model parameters')
    parser.add_argument('bag_path')
    # TODO add k, inner_gradient_steps as arguments
    args = parser.parse_args()

    main(ckpt_path=args.ckpt_path, bag_path=args.bag_path,
         device=torch.device('cpu'))