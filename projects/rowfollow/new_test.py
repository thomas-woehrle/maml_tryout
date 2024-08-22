import os

import cv2
import torch

import maml_api
import maml_config
import maml_eval
from rf_utils import viz

from rowfollow_model import RowfollowModel
from rowfollow_task import RowfollowTask
import rowfollow_utils as utils

run_id = 'a1c1639d073847db983e33785a509868'
episode = 9999
do_visual_test = True

support_bag_path = '/Users/tomwoehrle/Documents/research_assistance/cornfield_representative_pictures/20220603_cornfield/ts_2022_06_03_02h57m53s'
support_cam_side = 'left_cam'
target_bag_path = '/Users/tomwoehrle/Documents/research_assistance/cornfield1_labeled_new/20220603_cornfield/ts_2022_06_03_02h57m53s'
target_cam_side = 'both'
visualize_bag_path = target_bag_path
# visualize_bag_path = '/Users/tomwoehrle/Downloads/front_left_cam'
visualize_cam_side = 'left_cam'

hparams = maml_config.MamlHyperParameters(inner_steps=6)


def sample_task(stage: maml_api.Stage, seed: int = 0):
    return RowfollowTask(support_bag_path, 4, torch.device('cpu'),
                         support_cam_side=support_cam_side,
                         target_bag_path=target_bag_path, target_cam_side=target_cam_side,
                         seed=seed)


evaluator = maml_eval.MamlEvaluator(run_id, episode, sample_task,
                                    lib_directory='/Users/tomwoehrle/Documents/research_assistance/artifact_manager',
                                    hparams=hparams)

# evaluator.finetune()

use_traditional_model = True
ckpt = torch.load('/Users/tomwoehrle/Documents/research_assistance/supervised_train/row-follow-epoch=11-val_loss=0.000404.ckpt')
traditional_model = RowfollowModel()
traditional_model.load_state_dict(ckpt['state_dict'])
traditional_model.eval()
# evaluator.model = traditional_model

if True:
    input_path = os.path.join(visualize_bag_path, visualize_cam_side)
    # input_path = visualize_bag_path
    for file_name in os.listdir(input_path):
        if not file_name.endswith('.png'):
            continue
        file_path = os.path.join(input_path, file_name)
        data, img = utils.pre_process_image(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        data = torch.from_numpy(data).unsqueeze(0)

        if use_traditional_model:
            y_hat = traditional_model(data)
        else:
            y_hat = evaluator.model(data)

        img_with_lines = viz.img_with_lines_from_pred(img, y_hat)
        cv2.imshow('', img_with_lines)

        cv2.waitKey()


total_eval_loss = 0
for i in range(10):
    print('prediction step {}...'.format(i))
    total_eval_loss += evaluator.predict(seed=i)[1].item()
    print(total_eval_loss)


print('Total eval loss:', total_eval_loss)

# TODO visual test
