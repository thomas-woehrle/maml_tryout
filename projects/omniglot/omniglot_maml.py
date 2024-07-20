import argparse
import datetime
import os
import random
from typing import Any

import maml
import maml_config
import maml_utils

import omniglot_model
import omniglot_task
import omniglot_utils


def get_checkpoint_dir():
    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    checkpoint_dir = './checkpoints/omniglot_maml/' + formatted_time
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir


def main(maml_hparams: maml_config.MamlHyperParameters, env_config: maml_config.EnvConfig, other_config: dict[str, Any]):
    omniglot_chars = omniglot_utils.get_all_chars()
    random.shuffle(omniglot_chars)  # in-place shuffle
    train_chars = omniglot_chars[:1200]
    test_chars = omniglot_chars[1200:]
    # randomly determine 1200 chars for training, rest for testing

    def sample_task():
        return omniglot_task.OmniglotTask(random.sample(train_chars, k=other_config['n']), maml_hparams.k, env_config.device)

    omniglotModel = omniglot_model.OmniglotModel(other_config['n'])
    omniglotModel.to(env_config.device)

    ckpt_dir = maml_utils.get_ckpt_dir(env_config.ckpt_base,
                                         maml_hparams.use_anil, env_config.run_name)

    def checkpoint_fct(params, buffers, episode, loss):
        maml_utils.std_checkpoint_fct(ckpt_dir=ckpt_dir,
                                        current_episode=episode, current_loss=loss,
                                        params=params, buffers=buffers,
                                        train_data=train_chars, test_data=test_chars,
                                        maml_hparams=maml_hparams, env_config=env_config, other_config=other_config)

    maml.train(maml_hparams, sample_task, omniglotModel, checkpoint_fct, do_use_mlflow=True)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("config_filepath", type=str)
    config_filepath: str = arg_parser.parse_args().config_filepath

    maml_hparams, env_config, other_config = maml_config.load_configuration(
        config_filepath)

    main(maml_hparams, env_config, other_config)
