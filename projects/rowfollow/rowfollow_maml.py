import random
from typing import Any

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

    def sample_task(training_stage: maml_api.TrainingStage):
        if training_stage == maml_api.TrainingStage.TRAIN:
            return rowfollow_task.RowfollowTask(random.choice(train_bags),
                                                maml_hparams.k, env_config.device, sigma=other_config['sigma'])
        if training_stage == maml_api.TrainingStage.VAL:
            return rowfollow_task.RowfollowTask(random.choice(val_bags),
                                                maml_hparams.k, env_config.device, sigma=other_config['sigma'])

    model = rowfollow_model.RowfollowModel()
    model.to(env_config.device)

    maml.train(maml_hparams, sample_task, model, do_use_mlflow=True)


if __name__ == '__main__':
    main(*maml_config.parse_maml_args())
