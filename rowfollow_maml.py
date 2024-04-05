from maml import maml_learn
from models import RowfollowModel


def main():

    def sample_task():
        ...

    model = RowfollowModel()

    maml_learn()


if __name__ == '__main__':
    num_episodes = 100
    meta_batch_size = 4
    inner_gradient_steps = 1
    alpha = 0.4
    beta = 0.001

    main(num_episodes, meta_batch_size, inner_gradient_steps, alpha, beta)
