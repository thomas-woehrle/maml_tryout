import json
import os

import cv2
import mlflow
import torch

import maml_api
import maml_config
from rf_utils import viz
import rowfollow_utils as utils

import rowfollow_task


def download_model(run_id: str, episode: int):
    dst_path = os.path.join('models', run_id)
    os.makedirs(dst_path, exist_ok=True)
    model_uri = 'runs:/{}/models/ep{}'.format(run_id, episode)

    mlflow.pytorch.load_model(model_uri, dst_path)


def load_local_model(run_id: str, episode: int) -> maml_api.MamlModel:
    return mlflow.pytorch.load_model(os.path.join('models', run_id, 'ep{}'.format(episode)))


def load_local_dict(run_id: str, filename: str) -> dict:
    with open(os.path.join('models', run_id, filename), 'r') as f:
        return json.load(f)


def download_config(run_id: str):
    dst_path = os.path.join('models/{}'.format(run_id))
    os.makedirs(dst_path, exist_ok=True)
    mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path='hparams.json',
                                        dst_path=dst_path, tracking_uri='databricks')
    mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path='other_config.json',
                                        dst_path=dst_path, tracking_uri='databricks')
    mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path='bags.json',
                                        dst_path=dst_path, tracking_uri='databricks')


def train_model(model: maml_api.MamlModel, task: maml_api.MamlTask, alpha: float, steps: int):
    x, y = task.sample()
    optimizer = torch.optim.SGD(model.parameters(), alpha)
    for i in range(steps):
        optimizer.zero_grad()
        loss = task.calc_loss(model(x), y, maml_api.Stage.VAL, maml_api.SetToSetType.SUPPORT)
        print(loss.item())
        loss.backward()
        optimizer.step()


def test(run_id: str, episode: int, cf_and_bag: str, hparams_kwargs: dict = None):
    bag_path = os.path.join('/Users/tomwoehrle/Documents/research_assistance/cornfield1_labeled_new', cf_and_bag)

    model_path = os.path.join('models', run_id, 'ep{}'.format(episode))
    if not os.path.exists(model_path):
        download_model(run_id, episode)
    if not os.path.exists(os.path.join('models', run_id, 'hparams.json')):
        download_config(run_id)

    hparams = maml_config.MamlHyperParameters(**load_local_dict(run_id, 'hparams.json'))
    other_config = load_local_dict(run_id, 'other_config.json')

    model = load_local_model(run_id, episode)
    task = rowfollow_task.RowfollowTask(bag_path, hparams.k, torch.device('cpu'), other_config['sigma'])
    train_model(model, task, hparams.alpha / 2, hparams.inner_steps * 2)

    input_path = os.path.join(bag_path, 'left_cam')
    for file_name in os.listdir(input_path):
        if not file_name.endswith('.png'):
            continue
        file_path = os.path.join(input_path, file_name)
        data, img = utils.pre_process_image(file_path)
        data = torch.from_numpy(data)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_with_lines = viz.img_with_lines_from_pred(img, model(data.float().unsqueeze(0)))

        cv2.imshow('', img_with_lines)

        cv2.waitKey()


if __name__ == '__main__':
    bag1 = '20220603_cornfield/ts_2022_06_03_02h54m54s'
    bag2 = '20220622_cornfield/ts_2022_06_22_11h45m58s_one_row'

    test('59d7d133156d450895bd0243aeab2937', 19999, bag1)
