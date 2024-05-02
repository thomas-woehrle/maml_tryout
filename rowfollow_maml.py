import argparse
import random
from typing import Any

import maml_config
from maml import maml_learn
from models import RowfollowModel
from rowfollow_utils import get_train_and_test_bags
from shared_utils import std_checkpoint_fct
from tasks import RowfollowTask


def main(maml_hparams: maml_config.MamlHyperParameters, env_config: maml_config.EnvConfig, other_config: dict[str, Any]):
    train_bags, test_bags = get_train_and_test_bags(env_config.data_dir, 4, 5)

    def sample_task():
        return RowfollowTask(random.choice(train_bags), maml_hparams.k, env_config.device, sigma=other_config['sigma'])

    model = RowfollowModel()
    model.to(env_config.device)

    def checkpoint_fct(params, buffers, episode, loss):
        std_checkpoint_fct(current_episode=episode, current_loss=loss, params=params, buffers=buffers, train_data=train_bags,
                           test_data=test_bags, maml_hparams=maml_hparams, env_config=env_config, other_config=other_config)

    maml_learn(maml_hparams, sample_task, model, checkpoint_fct)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("config_filepath", type=str)
    config_filepath: str = arg_parser.parse_args().config_filepath
    maml_hparams, env_config, other_config = maml_config.load_configuration(
        config_filepath)

    main(maml_hparams, env_config, other_config)
