import argparse
import random
from typing import Any

import maml
import maml_config

import omniglot_model
import omniglot_task
import omniglot_utils


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

    def end_of_ep_fct(parameters, buffers, episode, acc_loss):
        print(episode, ':', acc_loss.item())

    maml.train(maml_hparams, sample_task, omniglotModel, end_of_ep_fct, do_use_mlflow=True)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("config_filepath", type=str)
    config_filepath: str = arg_parser.parse_args().config_filepath

    maml_hparams, env_config, other_config = maml_config.load_configuration(
        config_filepath)

    main(maml_hparams, env_config, other_config)
