import argparse
import datetime
import os

import torch

import maml_config


def std_checkpoint_fct(ckpt_dir,
                       current_episode,
                       current_loss,
                       params,
                       buffers,
                       train_data,
                       test_data,
                       maml_hparams: maml_config.MamlHyperParameters,
                       env_config: maml_config.EnvConfig,
                       other_config: dict):
    if not current_episode % 1000 == 0 and not current_episode == maml_hparams.n_episodes - 1:
        return

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
