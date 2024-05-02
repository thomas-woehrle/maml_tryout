import argparse
import random
from typing import Any

import maml
import maml_config
import models
import rowfollow_utils as utils
import shared_utils
import tasks


def main(maml_hparams: maml_config.MamlHyperParameters, env_config: maml_config.EnvConfig, other_config: dict[str, Any]):
    train_bags, test_bags = utils.get_train_and_test_bags(
        env_config.data_dir, 4, 5)

    def sample_task():
        return tasks.RowfollowTask(random.choice(train_bags), maml_hparams.k, env_config.device, sigma=other_config['sigma'])

    model = models.RowfollowModel()
    model.to(env_config.device)

    def checkpoint_fct(params, buffers, episode, loss):
        shared_utils.std_checkpoint_fct(current_episode=episode, current_loss=loss, params=params, buffers=buffers, train_data=train_bags,
                                        test_data=test_bags, maml_hparams=maml_hparams, env_config=env_config, other_config=other_config)

    maml.train(maml_hparams, sample_task, model, checkpoint_fct)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("config_filepath", type=str)
    config_filepath: str = arg_parser.parse_args().config_filepath
    maml_hparams, env_config, other_config = maml_config.load_configuration(
        config_filepath)

    main(maml_hparams, env_config, other_config)
