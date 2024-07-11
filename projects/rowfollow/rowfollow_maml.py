import argparse
import random
from typing import Any

import maml
import maml_config
import maml_utils

import rowfollow_model
import rowfollow_task
import rowfollow_utils


def main(maml_hparams: maml_config.MamlHyperParameters, env_config: maml_config.EnvConfig, other_config: dict[str, Any]):
    train_bags, test_bags = rowfollow_utils.get_train_and_test_bags(
        env_config.data_dir, 4, 5)

    def sample_task():
        return rowfollow_task.RowfollowTask(random.choice(train_bags), maml_hparams.k, env_config.device, sigma=other_config['sigma'])

    model = rowfollow_model.RowfollowModel()
    model.to(env_config.device)

    ckpt_dir = maml_utils.get_ckpt_dir(env_config.ckpt_base,
                                         maml_hparams.use_anil, env_config.run_name)

    def checkpoint_fct(params, buffers, episode, loss):
        maml_utils.std_checkpoint_fct(ckpt_dir=ckpt_dir,
                                        current_episode=episode, current_loss=loss,
                                        params=params, buffers=buffers,
                                        train_data=train_bags, test_data=test_bags,
                                        maml_hparams=maml_hparams, env_config=env_config, other_config=other_config)

    maml.train(maml_hparams, sample_task, model, checkpoint_fct)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("config_filepath", type=str)
    config_filepath: str = arg_parser.parse_args().config_filepath
    maml_hparams, env_config, other_config = maml_config.load_configuration(
        config_filepath)

    main(maml_hparams, env_config, other_config)
