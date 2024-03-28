from omniglot_helper import get_all_chars
import random
from tasks import OmniglotTask
from models import OmniglotModel
from maml import maml_learn


def main(n, k, num_episodes, meta_batch_size, inner_gradient_steps, alpha, beta, device='cpu'):
    omniglot_chars = get_all_chars()
    random.shuffle(omniglot_chars)  # in-place shuffle
    train_chars = omniglot_chars[:1200]
    test_chars = omniglot_chars[1200:]

    def sample_task():
        return OmniglotTask(random.sample(train_chars, k=n), k, device)

    omniglotModel = OmniglotModel(n)
    omniglotModel.to(device)

    def checkpoint_fct(episode, loss):
        pass

    theta = maml_learn(num_episodes, meta_batch_size, inner_gradient_steps,
                       alpha, beta, sample_task, omniglotModel, checkpoint_fct)


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
