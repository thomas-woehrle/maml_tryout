import datetime
import os
import random
import torch
from maml import maml_learn
from models import RowfollowModel
from tasks import RowfollowTask
from rowfollow_utils import get_train_and_test_bags


def get_checkpoint_dir():
    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y_%m_%d-%H_%M_%S")
    checkpoint_dir = './checkpoints/rowfollow_maml/' + formatted_time
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir


def main(data_dir, num_episodes, meta_batch_size, inner_gradient_steps, alpha, beta, k, device=torch.device('cpu')):
    train_bags, test_bags = get_train_and_test_bags(data_dir, 4, 5)

    def sample_task():
        return RowfollowTask(random.choice(train_bags), k, device)

    model = RowfollowModel()
    model.to(device)

    ckpt_dir = get_checkpoint_dir()

    def checkpoint_fct(params, buffers, episode, loss):
        if not episode % 1000 == 0 and not episode == num_episodes - 1:
            return

        state_dict = params | buffers
        torch.save(
            {
                'model_state_dict': state_dict,
                'train_bags': train_bags,
                'test_bags': test_bags
            }, os.path.join(ckpt_dir, f'ep{episode}_loss{loss}.pt')
        )

    maml_learn(num_episodes, meta_batch_size, inner_gradient_steps,
               alpha, beta, sample_task, model, checkpoint_fct)


if __name__ == '__main__':
    data_dir = '/Users/tomwoehrle/Documents/research_assistance/cornfield1_labeled_new/'
    device = torch.device('cpu')

    num_episodes = 10000
    meta_batch_size = 4
    inner_gradient_steps = 1
    alpha = 0.4
    beta = 0.001
    k = 4

    main(data_dir, num_episodes, meta_batch_size,
         inner_gradient_steps, alpha, beta, k, device)
