import copy
import os
import random
from typing import Any

import mlflow

import maml_logging
import maml_train
import maml_api
import maml_config

import rowfollow_model
import rowfollow_task
import rowfollow_utils
import rowfollow_test


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


def main_old_data(maml_hparams: maml_config.MamlHyperParameters, train_config: maml_config.TrainConfig,
                  env_config: maml_config.EnvConfig, other_config: dict[str, Any]):
    dataset_info_path = os.path.join(env_config.data_dir, 'dataset_info.csv')

    train_data_path = os.path.join(env_config.data_dir, 'train')
    train_annotations = os.path.join(train_data_path, 'v2_annotations_train.csv')
    train_collections = rowfollow_utils.get_train_data_paths(train_data_path, other_config['train_dataset_name'],
                                                             dataset_info_path)

    val_data_path = os.path.join(env_config.data_dir, 'val')
    val_annotations = os.path.join(val_data_path, 'v2_annotations_val.csv')
    val_collections = [os.path.join(val_data_path, d) for d in other_config['val_collections']
                       if os.path.isdir(os.path.join(val_data_path, d))]

    # save train and val bags
    if env_config.do_use_mlflow:
        data_info_dict = {
            "train_collections": train_collections,
            "val_collections": val_collections
        }
        mlflow.log_dict(data_info_dict, 'data_info.json')

    if env_config.seed is not None:
        maml_train.set_seed(env_config.seed)

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

    def calc_val_losses(current_episode: int, current_model: maml_api.MamlModel,
                        current_inner_buffers: maml_api.InnerBuffers, current_lrs: maml_api.InnerLrs,
                        logger: maml_logging.Logger):
        # get state and reassign it later. Done, because seed is set inside val_loss calculation
        random_prior_state = random.getstate()

        # see the documentation of maml_config.EnvConfig
        if env_config.use_same_val_seed_for_all_episodes:
            val_seed = env_config.seed or 0
        else:
            val_seed = current_episode + (env_config.seed or 0)

        for val_coll in val_collections:
            val_coll_model = copy.deepcopy(current_model).to(env_config.device)
            val_coll_inner_lrs = copy.deepcopy(current_lrs)
            val_coll_inner_buffers = copy.deepcopy(current_inner_buffers)
            # TODO is there another way instead of deepcopying ?
            rowfollow_test.calc_val_loss_for_train(current_episode, val_coll_model,
                                                   val_coll_inner_buffers, val_coll_inner_lrs,
                                                   maml_hparams.k, maml_hparams.inner_steps,
                                                   val_coll, val_annotations, env_config.device,
                                                   seed=val_seed, use_mlflow=env_config.do_use_mlflow, logger=logger,
                                                   sigma=other_config['sigma'])

        # reassign prior random state
        random.setstate(random_prior_state)

    model = rowfollow_model.RowfollowModel()
    model.to(env_config.device)

    trainer = maml_train.MamlTrainer(hparams=maml_hparams,
                                     sample_task=sample_task,
                                     model=model,
                                     device=env_config.device,
                                     do_use_mlflow=env_config.do_use_mlflow,
                                     train_config=train_config,
                                     calc_val_loss_fct=calc_val_losses)
    trainer.run_training()


if __name__ == '__main__':
    main_old_data(*maml_config.parse_maml_args())
