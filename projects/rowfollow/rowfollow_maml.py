import random
from typing import Any

import mlflow

import maml
import maml_api
import maml_config

import rowfollow_model
import rowfollow_task
import rowfollow_utils


def main(maml_hparams: maml_config.MamlHyperParameters, env_config: maml_config.EnvConfig,
         other_config: dict[str, Any]):
    # add test_bags ?
    train_bags, val_bags = rowfollow_utils.get_train_and_test_bags(
        env_config.data_dir, 4, 5)
    # TODO log into mlflow

    # save train and val bags
    if env_config.do_use_mlflow:
        bags_dict = {
            "train_bags": train_bags,
            "val_bags": val_bags
        }
        mlflow.log_dict(bags_dict, 'bags.json')

    def sample_task(training_stage: maml_api.Stage):
        if training_stage == maml_api.Stage.TRAIN:
            return rowfollow_task.RowfollowTask(random.choice(train_bags),
                                                maml_hparams.k, env_config.device, sigma=other_config['sigma'])
        if training_stage == maml_api.Stage.VAL:
            return rowfollow_task.RowfollowTask(random.choice(val_bags),
                                                maml_hparams.k, env_config.device, sigma=other_config['sigma'])

    model = rowfollow_model.RowfollowModel()
    model.to(env_config.device)

    trainer = maml.MamlTrainer(hparams=maml_hparams,
                               sample_task=sample_task,
                               model=model,
                               device=env_config.device,
                               do_use_mlflow=env_config.do_use_mlflow)
    trainer.train()


if __name__ == '__main__':
    main(*maml_config.parse_maml_args())
