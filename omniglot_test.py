import argparse
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import random
from models import OmniglotModel
from tasks import OmniglotTask

parser = argparse.ArgumentParser()
parser.add_argument('ckpt_path')
ckpt_path = parser.parse_args().ckpt_path

ckpt = torch.load(ckpt_path)

meta_model = OmniglotModel(5)
test_chars = ckpt['test_chars']
meta_model.load_state_dict(ckpt['model_state_dict'])
criterion = nn.CrossEntropyLoss(reduction='sum')
# TODO load checkpoint
alpha = 0.4  # NOTE only makes sense to use same as alpha from training, right?
n_evaluations = 10
n, k = 5, 1

for i in range(n_evaluations):

    task_model = copy.deepcopy(meta_model)
    old_task_model = {name: param for name,
                      param in task_model.named_parameters()}
    inner_optimizer = optim.SGD(task_model.parameters(), lr=alpha)

    task = OmniglotTask(random.sample(test_chars, k=n), k, 'cpu')
    # Evaluation uses 3 steps
    for j in range(3):
        x, y = task.sample()
        theta = [p for p in task_model.parameters()]
        train_loss = criterion(task_model(x, theta), y)

        # Inner loop update, currently only one step
        inner_optimizer.zero_grad()
        train_loss.backward()
        inner_optimizer.step()

    # evaluation of capabilities after training
    theta = [p for p in task_model.parameters()]
    x, y = task.sample()
    logits = task_model(x, theta)
    probs = nn.Softmax(dim=1)(logits)
    pred = torch.argmax(logits, dim=1)
    print(i)
    print(pred)
    print(y)
    test_loss = criterion(logits, y)
