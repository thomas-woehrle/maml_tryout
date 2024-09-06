import os
import random
from typing import Any

import mlflow

import maml_train
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

    trainer = maml_train.MamlTrainer(hparams=maml_hparams,
                                     sample_task=sample_task,
                                     model=model,
                                     device=env_config.device,
                                     do_use_mlflow=env_config.do_use_mlflow)
    trainer.run_training()


def main_old_data(maml_hparams: maml_config.MamlHyperParameters, env_config: maml_config.EnvConfig,
                  other_config: dict[str, Any]):
    train_data_path = os.path.join(env_config.data_dir, 'train')
    train_annotations = os.path.join(train_data_path, 'v2_annotations_train.csv')
    train_collections = [os.path.join(train_data_path, d) for d in os.listdir(train_data_path)
                         if os.path.isdir(os.path.join(train_data_path, d))]

    val_data_path = os.path.join(env_config.data_dir, 'val')
    val_annotations = os.path.join(val_data_path, 'v2_annotations_val.csv')
    val_collections = [os.path.join(val_data_path, d) for d in os.listdir(val_data_path)
                         if os.path.isdir(os.path.join(val_data_path, d))]

    def sample_task(training_stage: maml_api.Stage):
        if training_stage == maml_api.Stage.TRAIN:
            support_data_path = random.choice(train_collections)
            return rowfollow_task.RowfollowTaskOldDataset(train_annotations,
                                                          support_data_path,
                                                          maml_hparams.k,
                                                          env_config.device,
                                                          sigma=other_config['sigma'])
        if training_stage == maml_api.Stage.VAL:
            support_data_path = random.choice(val_collections)
            return rowfollow_task.RowfollowTaskOldDataset(val_annotations,
                                                          support_data_path,
                                                          maml_hparams.k,
                                                          env_config.device,
                                                          sigma=other_config['sigma'])

    model = rowfollow_model.RowfollowModel()
    model.to(env_config.device)

    trainer = maml_train.MamlTrainer(hparams=maml_hparams,
                                     sample_task=sample_task,
                                     model=model,
                                     device=env_config.device,
                                     do_use_mlflow=env_config.do_use_mlflow)
    trainer.run_training()


if __name__ == '__main__':
    main_old_data(*maml_config.parse_maml_args())
