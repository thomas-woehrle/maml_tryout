import random
from typing import Any

import mlflow

import maml_train
import maml_api
import maml_config

import omniglot_model
import omniglot_task
import omniglot_utils


def main(maml_hparams: maml_config.MamlHyperParameters, env_config: maml_config.EnvConfig,
         other_config: dict[str, Any]):
    # randomly determine 1200 chars for training, rest for testing
    omniglot_chars = omniglot_utils.get_all_chars(env_config.data_dir)
    random.shuffle(omniglot_chars)  # in-place shuffle
    train_chars = omniglot_chars[:1200]
    val_chars = omniglot_chars[1200:]

    # save train and eval chars
    if env_config.do_use_mlflow:
        # this in effect means that the chars are not saved if do_use_mlflow=False. This is not perfect but okay for now
        chars_dict = {
            "train_chars": train_chars,
            "val_chars": val_chars
        }
        mlflow.log_dict(chars_dict, 'chars.json')

    def sample_task(training_stage: maml_api.Stage):

        if training_stage == maml_api.Stage.TRAIN:
            return omniglot_task.OmniglotTask(random.sample(train_chars, k=other_config['n']),
                                              maml_hparams.k, env_config.device)
        if training_stage == maml_api.Stage.VAL:
            return omniglot_task.OmniglotTask(random.sample(val_chars, k=other_config['n']),
                                              maml_hparams.k, env_config.device)

    omniglotModel = omniglot_model.OmniglotModel(other_config['n'])
    omniglotModel.to(env_config.device)

    trainer = maml.MamlTrainer(hparams=maml_hparams,
                               sample_task=sample_task,
                               model=omniglotModel,
                               device=env_config.device,
                               do_use_mlflow=env_config.do_use_mlflow)
    trainer.train()


if __name__ == '__main__':
    main(*maml_config.parse_maml_args())
