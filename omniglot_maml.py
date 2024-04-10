import datetime
import os
import random
import torch
from maml import maml_learn
from anil_maml import anil_learn
from models import OmniglotModel
from omniglot_helper import get_all_chars
from tasks import OmniglotTask


def get_checkpoint_dir():
    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    checkpoint_dir = './checkpoints/' + formatted_time
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir


def main(n, k, num_episodes, meta_batch_size, inner_gradient_steps, alpha, beta, device='cpu'):
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
        if not episode % 1000 == 0 and not episode == num_episodes - 1:
            return

        state_dict = params | buffers
        torch.save(
            {
                'model_state_dict': state_dict,
                'train_chars': train_chars,
                'test_chars': test_chars
            }, os.path.join(ckpt_dir, f'ep{episode}_loss{loss}.pt')
        )

    maml_learn(num_episodes, meta_batch_size, inner_gradient_steps,
               alpha, beta, sample_task, omniglotModel, checkpoint_fct)

# TODO double-check if train and test chars are the same in checkpoints


if __name__ == '__main__':
    n = 5
    k = 1

    num_episodes = 60000
    meta_batch_size = 32
    inner_gradient_steps = 1
    alpha = 0.4
    beta = 0.001

    main(n, k, num_episodes, meta_batch_size,
         inner_gradient_steps, alpha, beta)
