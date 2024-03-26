import copy
import datetime
import os
import torch
import torch.nn as nn
import torch.optim as optim
from alt_omniglot_net import OmniglotNet as AltOmniglotNet
from omniglot_helper import viz_logit
from task import get_task, generate_k_samples_from_task


now = datetime.datetime.now()
formatted_time = now.strftime("%Y-%m-%d-%H-%M-%S")
checkpoint_dir = './checkpoints/' + formatted_time
os.makedirs(checkpoint_dir, exist_ok=True)
n_episodes = 60000  # i think 60000 in real world
meta_batch_size = 32
n = 5
k = 1
alpha, beta = 0.4, 0.001  # learning rates during training
# TODO find out real beta -> apparently is 0.001

criterion = nn.CrossEntropyLoss(reduction='sum')  # same for every task
meta_model = AltOmniglotNet(n)

last_losses = []
for episode in range(n_episodes):
    meta_loss = 0
    # initialize accumulated gradient to be a dictionary with keys corresponding to weights and values of 0 everywhere
    acc_grad = {name: 0 for name, param in meta_model.named_parameters()}
    for i in range(meta_batch_size):
        task_model = copy.deepcopy(meta_model)
        old_task_model = {name: param for name,
                          param in task_model.named_parameters()}
        inner_optimizer = optim.SGD(task_model.parameters(), lr=alpha)

        task = get_task('train', n)
        x, y = generate_k_samples_from_task(task, k)
        train_loss = criterion(task_model(x), y)
        # x and y have batch_size of n*k
        # technically, get_task and generate_k_samples_from_task could easily be put into one function. However,
        # this approach sticks closer to the original concept of a task that generates samples

        # Inner loop update, currently only one step
        inner_optimizer.zero_grad()
        train_loss.backward()
        inner_optimizer.step()

        # Update meta loss
        x_test, y_test = generate_k_samples_from_task(task, k)
        logit = task_model(x_test)
        if episode < 0:
            viz_logit(x_test, y_test, torch.round(logit * 100))

        test_loss = criterion(logit, y_test)
        meta_loss += test_loss

        # Update grad accumulation used for meta update
        inner_optimizer.zero_grad()  # needed
        test_loss.backward()
        acc_grad = {name: acc_grad[name] + param.grad for name,
                    param in task_model.named_parameters()}

    last_losses.append(meta_loss.item())
    if len(last_losses) == 100:
        print(episode)
        print(sum(last_losses) // 1)
        last_losses = []
    # print(acc_grad)
    # theta = {name: param.data - beta * acc_grad[name]
    #          # not sure about param vs param.data
    #          for name, param in meta_model.named_parameters()}
    # meta_model.load_state_dict(theta)
    for name, param in meta_model.named_parameters():
        param.data -= beta * acc_grad[name]

    # checkpoint saving
    if (episode % 1000 == 0) or (episode == n_episodes - 1):
        checkpoint_path = os.path.join(
            checkpoint_dir, f'model_episode_{episode}_loss_{meta_loss.item()}.pt')
        torch.save({
            'epoch': episode,
            'model_state_dict': meta_model.state_dict()
        }, checkpoint_path)
