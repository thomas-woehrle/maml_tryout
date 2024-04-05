import random
from maml import maml_learn
from models import RowfollowModel
from tasks import RowfollowTask
from rowfollow_utils import get_train_and_test_bags


def main(data_dir, k):
    train_bags, test_bags = get_train_and_test_bags(data_dir, 4, 5)

    def sample_task():
        return RowfollowTask(random.sample(train_bags, 1), k)

    model = RowfollowModel()
    sample_task()


if __name__ == '__main__':
    data_dir = '/Users/tomwoehrle/Documents/research_assistance/cornfield1_labeled_new/'

    num_episodes = 100
    meta_batch_size = 4
    inner_gradient_steps = 1
    alpha = 0.4
    beta = 0.001
    k = 4

    main(data_dir)
