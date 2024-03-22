import argparse
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from alt_omniglot_net import OmniglotNet
from task import get_task, generate_k_samples_from_task

parser = argparse.ArgumentParser()
parser.add_argument('ckpt_path')
ckpt_path = parser.parse_args().ckpt_path

ckpt = torch.load(ckpt_path)

meta_model = OmniglotNet(5)
meta_model.load_state_dict(ckpt['model_state_dict'])
criterion = nn.CrossEntropyLoss(reduction='sum')
# TODO load checkpoint
alpha = 0.4  # NOTE only makes sense to use same as alpha from training, right?
n_evaluations = 2
n, k = 5, 1

for i in range(n_evaluations):

    task_model = copy.deepcopy(meta_model)
    old_task_model = {name: param for name,
                      param in task_model.named_parameters()}
    inner_optimizer = optim.SGD(task_model.parameters(), lr=alpha)

    task = get_task('test', n)
    # Evaluation uses 3 steps
    for i in range(3):
        x, y = generate_k_samples_from_task(task, k)
        train_loss = criterion(task_model(x), y)

        # Inner loop update, currently only one step
        inner_optimizer.zero_grad()
        train_loss.backward()
        inner_optimizer.step()

    # evaluation of capabilities after training
    x, y = generate_k_samples_from_task(task, 1)
    logits = task_model(x)
    probs = nn.Softmax(dim=1)(logits)
    pred = torch.argmax(logits, dim=1)
    print(i)
    print(logits)
    print(probs)
    print(pred)
    print(y)
    test_loss = criterion(task_model(x), y)
    print(test_loss)
