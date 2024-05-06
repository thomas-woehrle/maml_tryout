import argparse
import random

import torch

import maml
import models
import tasks
from torchmetrics import classification

parser = argparse.ArgumentParser()
parser.add_argument('ckpt_path')
ckpt_path = parser.parse_args().ckpt_path

ckpt = torch.load(ckpt_path)

meta_model = models.OmniglotModel(5)
test_chars = ckpt.get('test_chars', ckpt.get('test_data', None))
meta_model.load_state_dict(ckpt['model_state_dict'])
meta_params, buffers = meta_model.get_state()

alpha = 0.4  # NOTE only makes sense to use same as alpha from training, right?
n_evaluations = 100
n, k = 5, 1
inner_gradient_steps = 3

accuracy = classification.MulticlassAccuracy(num_classes=5)

for i in range(n_evaluations):
    task = tasks.OmniglotTask(random.sample(test_chars, k=n), k, 'cpu')

    params_i = meta_params
    params_i = maml.inner_loop_update_for_testing(anil=False,
                                                  model=meta_model,
                                                  params=meta_params,
                                                  buffers=buffers,
                                                  task=task,
                                                  alpha=alpha,
                                                  inner_gradient_steps=inner_gradient_steps)

    # meta_model.eval()  # check how this is needed
    # evaluation of capabilities after training
    x, y = task.sample('query', -1)
    y_hat = meta_model.func_forward(x, params_i, buffers)
    # print(y_hat.argmax(dim=1))
    # print(y)
    accuracy.update(y_hat, y)

print('### Final accuracy ###')
print(accuracy.compute())
