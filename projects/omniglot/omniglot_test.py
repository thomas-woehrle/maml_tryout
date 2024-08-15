import random
from typing import List

import mlflow

import maml_train
import maml_config
import omniglot_model
import omniglot_task
from torchmetrics import classification

# TODO add argparser which takes the different variables in + an entrypoint in MLproject? see comment below this as well

# maybe use eval-experiments and runs instead of not logging anything
mlflow.set_tracking_uri('databricks')

run_id = 'abbfb7205d3b4f618acbd77b01aa1405'
model_path = 'models/ep2'
n_evaluations = 10

run_url = "runs:/{}".format(run_id)

# load model
model: omniglot_model.OmniglotModel = mlflow.pytorch.load_model("{}/{}".format(run_url, model_path))

# load and adjust hparams and configuration
hparams = maml_config.MamlHyperParameters(**mlflow.artifacts.load_dict('{}/{}'.format(run_url, 'hparams.json')))
hparams.inner_steps = 3
n = mlflow.artifacts.load_dict("{}/{}".format(run_url, 'other_config.json'))['n']

val_chars: List[str] = mlflow.artifacts.load_dict("{}/{}".format(run_url, 'chars.json'))['val_chars']

accuracy = classification.MulticlassAccuracy(num_classes=5)

params, buffers = model.get_state()

for i in range(n_evaluations):
    task = omniglot_task.OmniglotTask(random.sample(val_chars, k=n), hparams.k, 'cpu')

    params_i = maml.inner_loop_update_for_testing(anil=False,
                                                  model=model,
                                                  params=params,
                                                  buffers=buffers,
                                                  task=task,
                                                  alpha=hparams.alpha,
                                                  inner_gradient_steps=hparams.inner_steps, )

    # meta_model.eval()  # check how this is needed
    # evaluation of capabilities after training
    x, y = task.sample()
    y_hat = model.func_forward(x, params_i, buffers)
    # print(y_hat.argmax(dim=1))
    # print(y)
    accuracy.update(y_hat, y)

print('### Final accuracy ###')
print(accuracy.compute())
