import datetime
import os
import random

import maml
import models
import omniglot_utils as utils
import shared_utils
import tasks


def get_checkpoint_dir():
    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    checkpoint_dir = './checkpoints/omniglot_maml/' + formatted_time
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir


def main(n, k, num_episodes, meta_batch_size, inner_gradient_steps, alpha, beta, device='cpu'):
    omniglot_chars = utils.get_all_chars()
    random.shuffle(omniglot_chars)  # in-place shuffle
    train_chars = omniglot_chars[:1200]
    test_chars = omniglot_chars[1200:]

    def sample_task():
        return tasks.OmniglotTask(random.sample(train_chars, k=n), k, device)

    omniglotModel = models.OmniglotModel(n)
    omniglotModel.to(device)

    ckpt_dir = get_checkpoint_dir()

    def checkpoint_fct(params, buffers, episode, loss):
        shared_utils.std_checkpoint_fct(episode, loss, params, buffers, train_chars, test_chars, 'OmniglotTask', num_episodes,
                                        meta_batch_size, inner_gradient_steps, alpha, beta, k, ckpt_dir, None)

    maml.train(num_episodes, meta_batch_size, inner_gradient_steps,
               alpha, beta, sample_task, omniglotModel, checkpoint_fct)


if __name__ == '__main__':
    n = 5
    k = 1

    num_episodes = 60000
    meta_batch_size = 32
    inner_gradient_steps = 1
    alpha = 0.4
    beta = 0.001
    # TODO adjust to new variable structure of maml.py
    main(n, k, num_episodes, meta_batch_size,
         inner_gradient_steps, alpha, beta)
