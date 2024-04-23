import random
import torch
from maml import maml_learn
from models import RowfollowModel
from rowfollow_utils import get_train_and_test_bags
from shared_utils import std_checkpoint_fct, get_base_parser, get_ckpt_dir
from tasks import RowfollowTask


def main(num_episodes: int, meta_batch_size: int, k: int, inner_gradient_steps: int,
         alpha: float, beta: float, ckpt_dir: str, device: torch.device,
         data_dir: str, anil: bool, sigma: int):
    train_bags, test_bags = get_train_and_test_bags(data_dir, 4, 5)

    def sample_task():
        return RowfollowTask(random.choice(train_bags), k, device, sigma=sigma)

    model = RowfollowModel()
    model.to(device)

    def checkpoint_fct(params, buffers, episode, loss):
        std_checkpoint_fct(episode, loss, params, buffers, train_bags, test_bags, 'RowfollowTask', num_episodes,
                           meta_batch_size, k, inner_gradient_steps, alpha, beta, anil, ckpt_dir, add_info={'sigma': sigma})

    maml_learn(anil, num_episodes, meta_batch_size, inner_gradient_steps,
               alpha, beta, sample_task, model, checkpoint_fct)


if __name__ == '__main__':
    parser = get_base_parser()
    parser.add_argument(
        '--data_dir', type=str, help='The directory where the files in question are stored')
    parser.add_argument(
        '--sigma', default=10, type=int, help='The sigma value applied to create a heatmap out of the labels. (Default: 10)')

    args = parser.parse_args()

    ckpt_dir = get_ckpt_dir(args.ckpt_base_dir, args.anil, args.run_name)
    device = torch.device(args.device)

    main(num_episodes=args.num_episodes,
         meta_batch_size=args.meta_batch_size,
         k=args.k,
         inner_gradient_steps=args.inner_gradient_steps,
         alpha=args.alpha,
         beta=args.beta,
         ckpt_dir=ckpt_dir,
         device=device,
         data_dir=args.data_dir,
         anil=args.anil,
         sigma=args.sigma)
