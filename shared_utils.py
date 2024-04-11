import datetime
import os
import torch


def std_checkpoint_fct(current_episode,
                       current_loss,
                       params,
                       buffers,
                       train_data,
                       test_data,
                       task_name,
                       num_episodes,
                       meta_batch_size,
                       inner_gradient_steps,
                       alpha,
                       beta,
                       k,
                       ckpt_dir,
                       ckpt_name):
    if not current_episode % 1000 == 0 and not current_episode == num_episodes - 1:
        return
    ckpt_name = os.path.join(ckpt_dir, f'ep{current_episode}_loss{
                             current_loss}.pt')  # fixed for now

    state_dict = params | buffers
    torch.save(
        {
            'model_state_dict': state_dict,
            'train_data': train_data,
            'test_data': test_data,
            'task_name': task_name,
            'num_episodes': num_episodes,
            'meta_batch_size': meta_batch_size,
            'inner_gradient_steps': inner_gradient_steps,
            'alpha': alpha,
            'beta': beta,
            'k': k,
            'current_date': datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S"),
            'current_episode': current_episode,
        }, ckpt_name
    )
