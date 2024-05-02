import datetime
import os
import torch
import argparse

import hparams


def std_checkpoint_fct(current_episode,
                       current_loss,
                       params,
                       buffers,
                       train_data,
                       test_data,
                       maml_hparams: hparams.MamlHyperParameters,
                       env_config: hparams.EnvConfig,
                       other_config: dict):
    if not current_episode % 1000 == 0 and not current_episode == maml_hparams.n_episodes - 1:
        return
    ckpt_dir = get_ckpt_dir(env_config.ckpt_base,
                            maml_hparams.use_anil, env_config.run_name)
    ckpt_name = os.path.join(
        ckpt_dir, f'ep{current_episode}_loss{current_loss}.pt')

    state_dict = params | buffers
    torch.save(
        {
            'current_date': datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S"),
            'current_episode': current_episode,
            'current_loss': current_loss,
            'model_state_dict': state_dict,
            'train_data': train_data,
            'test_data': test_data,
            **vars(maml_hparams),
            **vars(env_config),
            **other_config
        }, ckpt_name
    )


def get_base_parser():
    """
    Returns a arg parser, which can be used and expanded by any maml implementation
    """
    base_parser = argparse.ArgumentParser()
    base_parser.add_argument('--num_episodes', type=int,
                             help='The number of episodes. f.e 60k')
    base_parser.add_argument('--meta_batch_size', type=int,
                             help='The number of tasks sampled per episode')
    base_parser.add_argument(
        '--k', type=int, help='The batch_size of samples for each task')
    base_parser.add_argument('--inner_gradient_steps', type=int,
                             help='The number of gradient steps taken in the inner loop')
    base_parser.add_argument(
        '--alpha', type=float, help='The learning rate of the inner loop')
    base_parser.add_argument(
        '--beta', type=float, help='The learning rate of the outer loop')
    base_parser.add_argument(
        '--device', type=str, help='The device used. Should be passable to torch.device(...)')
    base_parser.add_argument('--ckpt_base_dir', type=str,
                             help='In this directory/run_name the checkpoints will be saved')
    base_parser.add_argument('--run_name', type=str, default=None,
                             help='See description of ckpt_base_dir. Will be the current date and time if nothing is supplied. ')
    base_parser.add_argument('--anil', action='store_true',
                             help='Enable ANIL mode. If turned on, the model needs to have a self.header child')

    return base_parser


def get_ckpt_dir(base_dir, anil, run_name):
    if run_name is None:
        run_name = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

    ckpt_dir = os.path.join(base_dir, run_name) if not anil else os.path.join(
        base_dir, 'anil', run_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    return ckpt_dir


def get_indices_from_pred(pred):
    """
    pred will have shape (batch_sizex3x56x80)  
    """
    batch_size = pred.shape[0]
    flattened_indices = torch.argmax(pred.view(batch_size, 3, -1), dim=2)

    height = 56
    width = 80
    y_coords = flattened_indices // width
    x_coords = flattened_indices % width

    indices = torch.stack((x_coords, y_coords), dim=2)
    return indices


def get_coordinates_on_frame(vp, kp, dim=(319, 223)):
    line1 = (vp, kp)
    line2 = ((0, dim[1]), (dim[0], dim[1]))
    intersect = find_intersection(line1, line2)
    if intersect == None:
        if torch.is_tensor(kp):
            kp = kp[0].item(), kp[1].item()
        return tuple(kp)

    if intersect[0] > dim[0] or intersect[0] < 0:
        x = dim[0] if intersect[0] > dim[0] else 0
        line2 = ((x, 0), (x, dim[1]))
        intersect = find_intersection(line1, line2)

    if intersect == None:
        if torch.is_tensor(kp):
            kp = kp[0].item(), kp[1].item()
        return kp

    return intersect


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
