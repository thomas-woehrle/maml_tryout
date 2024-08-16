import ast
import json
import os
from typing import Optional

import cv2
import mlflow
import pandas as pd
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


def get_y_for_img(img_name: str, df: pd.DataFrame, sigma: float) -> Optional[torch.Tensor]:
    try:
        sample = df[df['image_name'] == img_name].iloc[0]
        vp, ll, lr = ast.literal_eval(sample.vp), ast.literal_eval(
            sample.ll), ast.literal_eval(sample.lr)
        # vp, ll, lr are coordinates, but we need distributions
        vp_gt = utils.dist_from_keypoint(vp, sig=sigma, downscale=4)
        ll_gt = utils.dist_from_keypoint(ll, sig=sigma, downscale=4)
        lr_gt = utils.dist_from_keypoint(lr, sig=sigma, downscale=4)

        y = torch.stack([vp_gt, ll_gt, lr_gt]).unsqueeze(0)

        return y
    except IndexError:
        return None


def test(run_id: str, episode: int, cf_and_bag: str, cam_side: str = 'left_cam',
         train_path: Optional[str] = None, train_cam_side: str = 'both', hparams_kwargs: dict = None):
    do_visualize = True

    bag_path = os.path.join('/Users/tomwoehrle/Documents/research_assistance/cornfield1_labeled_new', cf_and_bag)

    if train_path is None:
        train_path = bag_path

    model_path = os.path.join('models', run_id, 'ep{}'.format(episode))
    if not os.path.exists(model_path):
        download_model(run_id, episode)
    if not os.path.exists(os.path.join('models', run_id, 'hparams.json')):
        download_config(run_id)

    hparams = maml_config.MamlHyperParameters(**load_local_dict(run_id, 'hparams.json'))
    other_config = load_local_dict(run_id, 'other_config.json')

    print(train_path)
    model = load_local_model(run_id, episode)
    task = rowfollow_task.RowfollowTask(train_path, hparams.k, torch.device('cpu'), other_config['sigma'],
                                        support_cam_side=train_cam_side, seed=0)
    train_model(model, task, hparams.alpha, 3)

    input_path = os.path.join(bag_path, cam_side)
    labels_df = pd.read_csv(os.path.join(input_path, 'labels.csv'))

    total_target_loss = 0
    for file_name in os.listdir(input_path):
        if not file_name.endswith('.png'):
            continue
        file_path = os.path.join(input_path, file_name)
        data, img = utils.pre_process_image(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        data = torch.from_numpy(data)

        y = get_y_for_img(file_name, labels_df, other_config['sigma'])
        if y is None:
            continue
        y_hat = model(data.float().unsqueeze(0))
        target_loss = task.calc_loss(y_hat, y, maml_api.Stage.VAL, maml_api.SetToSetType.TARGET).item()
        total_target_loss += target_loss

        if do_visualize:
            img_with_lines = viz.img_with_lines_from_pred(img, y_hat)
            cv2.imshow(str(target_loss), img_with_lines)

            cv2.waitKey()
    print('Total target loss: {}'.format(total_target_loss))


if __name__ == '__main__':
    bag1 = '20220603_cornfield/ts_2022_06_03_02h57m53s'
    bag2 = '20220622_cornfield/ts_2022_06_22_11h45m58s_one_row'
    rep_bag = ('/Users/tomwoehrle/Documents/research_assistance/cornfield_representative_pictures/'
               + '20220603_cornfield/ts_2022_06_03_02h57m53s')

    test('59d7d133156d450895bd0243aeab2937', 18000, bag2) #train_path=rep_bag, train_cam_side='left_cam')
