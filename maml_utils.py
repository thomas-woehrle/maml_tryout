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


def get_ckpt_dir(base_dir, anil, run_name):
    if run_name is None:
        run_name = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

    ckpt_dir = os.path.join(base_dir, run_name) if not anil else os.path.join(
        base_dir, 'anil', run_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    return ckpt_dir
