import datetime
import os
import random
from maml import maml_learn
from models import OmniglotModel
from omniglot_utils import get_all_chars
from shared_utils import std_checkpoint_fct
from tasks import OmniglotTask


def get_checkpoint_dir():
    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    checkpoint_dir = './checkpoints/omniglot_maml/' + formatted_time
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir


def main(anil, n, k, num_episodes, meta_batch_size, inner_gradient_steps, alpha, beta, device='cpu'):
    omniglot_chars = get_all_chars()
    random.shuffle(omniglot_chars)  # in-place shuffle
    train_chars = omniglot_chars[:1200]
    test_chars = omniglot_chars[1200:]

    def sample_task():
        return OmniglotTask(random.sample(train_chars, k=n), k, device)

    omniglotModel = OmniglotModel(n)
    omniglotModel.to(device)

    ckpt_dir = get_checkpoint_dir()

    def checkpoint_fct(params, buffers, episode, loss):
        std_checkpoint_fct(episode, loss, params, buffers, train_chars, test_chars, 'OmniglotTask', num_episodes,
                           meta_batch_size, k, inner_gradient_steps, alpha, beta, anil, ckpt_dir)

    maml_learn(anil, num_episodes, meta_batch_size, inner_gradient_steps,
               alpha, beta, sample_task, omniglotModel, checkpoint_fct)

# TODO double-check if train and test chars are the same in checkpoints


if __name__ == '__main__':
    n = 5
    k = 1

    anil = False
    num_episodes = 60000
    meta_batch_size = 32
    inner_gradient_steps = 1
    alpha = 0.4
    beta = 0.001
    # TODO adjust to new variable structure of maml.py
    main(anil, n, k, num_episodes, meta_batch_size,
         inner_gradient_steps, alpha, beta)
