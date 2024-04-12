import datetime
import os
import random
import torch
from anil_maml import anil_learn
from maml import maml_learn
from models import RowfollowModel
from rowfollow_utils import get_train_and_test_bags
from shared_utils import std_checkpoint_fct
from tasks import RowfollowTask


def get_checkpoint_dir():
    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y_%m_%d-%H_%M_%S")
    checkpoint_dir = './checkpoints/rowfollow_maml/' + formatted_time
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir


def main(data_dir, num_episodes, meta_batch_size, inner_gradient_steps, alpha, beta, k, device=torch.device('cpu')):
    train_bags, test_bags = get_train_and_test_bags(data_dir, 4, 5)

    def sample_task():
        return RowfollowTask(random.choice(train_bags), k, num_episodes, device)

    model = RowfollowModel()
    model.to(device)

    ckpt_dir = get_checkpoint_dir()

    def checkpoint_fct(params, buffers, episode, loss):
        std_checkpoint_fct(episode, loss, params, buffers, train_bags, test_bags, 'RowfollowTask', num_episodes,
                           meta_batch_size, inner_gradient_steps, alpha, beta, k, ckpt_dir, None)

    maml_learn(num_episodes, meta_batch_size, inner_gradient_steps,
               alpha, beta, sample_task, model, checkpoint_fct)


if __name__ == '__main__':
    data_dir = '/Users/tomwoehrle/Documents/research_assistance/cornfield1_labeled_new/'
    device = torch.device('cpu')

    num_episodes = 60000
    meta_batch_size = 4
    inner_gradient_steps = 1
    alpha = 0.4
    beta = 0.001
    k = 4

    main(data_dir, num_episodes, meta_batch_size,
         inner_gradient_steps, alpha, beta, k, device)
