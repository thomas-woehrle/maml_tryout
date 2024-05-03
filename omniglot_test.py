import argparse
import random

import torch
import torch.nn as nn

import maml
import models
import tasks

parser = argparse.ArgumentParser()
parser.add_argument('ckpt_path')
ckpt_path = parser.parse_args().ckpt_path

ckpt = torch.load(ckpt_path)

meta_model = models.OmniglotModel(5)
test_chars = ckpt['test_chars']
meta_model.load_state_dict(ckpt['model_state_dict'])
meta_params, buffers = meta_model.get_initial_state()
meta_model.eval()

criterion = nn.CrossEntropyLoss(reduction='sum')
# TODO load checkpoint
alpha = 0.4  # NOTE only makes sense to use same as alpha from training, right?
n_evaluations = 10
n, k = 5, 1

for i in range(n_evaluations):
    task = tasks.OmniglotTask(random.sample(test_chars, k=n), k, 'cpu')

    params_i = meta_params
    for i in range(3):
        params_i = maml.inner_loop_update(
            meta_model, meta_params, buffers, task, alpha, 'doesnt matter')

    # evaluation of capabilities after training
    x, y = task.sample('query')
    logits = meta_model.func_forward(x, params_i, buffers)
    probs = nn.Softmax(dim=1)(logits)
    pred = torch.argmax(logits, dim=1)
    print(i)
    print(pred)
    print(y)
    test_loss = criterion(logits, y)
