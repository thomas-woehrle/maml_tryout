import random
from typing import Any

import maml
import maml_api
import maml_config

import omniglot_model
import omniglot_task
import omniglot_utils


def main(maml_hparams: maml_config.MamlHyperParameters, env_config: maml_config.EnvConfig,
         other_config: dict[str, Any]):
    def sample_task(training_stage: maml_api.TrainingStage):
        # randomly determine 1200 chars for training, rest for testing
        omniglot_chars = omniglot_utils.get_all_chars()
        random.shuffle(omniglot_chars)  # in-place shuffle
        train_chars = omniglot_chars[:1200]
        test_chars = omniglot_chars[1200:]

        if training_stage == maml_api.TrainingStage.TRAIN:
            return omniglot_task.OmniglotTask(random.sample(train_chars, k=other_config['n']),
                                              maml_hparams.k, env_config.device)
        if training_stage == maml_api.TrainingStage.EVAL:
            return omniglot_task.OmniglotTask(random.sample(test_chars, k=other_config['n']),
                                              maml_hparams.k, env_config.device)

    omniglotModel = omniglot_model.OmniglotModel(other_config['n'])
    omniglotModel.to(env_config.device)

    def end_of_ep_fct(parameters, buffers, episode, acc_loss, eval_loss):
        print(episode, ':', acc_loss.item())
        print(episode, ':', eval_loss.item())

    maml.train(maml_hparams, sample_task, omniglotModel,
               end_of_ep_fct, do_use_mlflow=True)


if __name__ == '__main__':
    main(*maml_config.parse_maml_args())
