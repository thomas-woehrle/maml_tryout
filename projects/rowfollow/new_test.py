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

bag_paths = {
    "1006_1": os.path.join("20221006_cornfield", "ts_2022_10_06_10h06m49s_two_random"),
    "1006_2": os.path.join("20221006_cornfield", "ts_2022_10_06_10h15m24s_two_random"),
    "1006_3": os.path.join("20221006_cornfield", "ts_2022_10_06_10h21m44s_four_rows"),
    "1006_4": os.path.join("20221006_cornfield", "ts_2022_10_06_10h29m23s_two_random"),
    "0603_1": "20220603_cornfield/ts_2022_06_03_02h57m53s"
}


def get_bag_path(bag_path_key: str):
    base_path = '/Users/tomwoehrle/Documents/research_assistance/cornfield1_labeled_new/'
    return os.path.join(base_path, bag_paths[bag_path_key])


runs = {
    "only_msl_1": "f0990d091ae446108776ec818bb14f46",
    "only_msl_2": "59d7d133156d450895bd0243aeab2937"
}


run_id = runs['only_msl_1']
episode = 19999
do_visual_test = False

# support_bag_path = '/Users/tomwoehrle/Documents/research_assistance/cornfield_representative_pictures/20220603_cornfield/ts_2022_06_03_02h57m53s'
# support_cam_side = 'left_cam'
# target_bag_path = '/Users/tomwoehrle/Documents/research_assistance/cornfield1_labeled_new/20220603_cornfield/ts_2022_06_03_02h57m53s'
# target_cam_side = 'both'
# visualize_bag_path = target_bag_path
# visualize_bag_path = '/Users/tomwoehrle/Downloads/front_left_cam'
# visualize_cam_side = 'left_cam'

support_bag_path = get_bag_path("0603_1")
support_cam_side = 'left_cam'
target_bag_path = support_bag_path
target_cam_side = support_cam_side
visualize_bag_path = target_bag_path
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

evaluator.finetune()

use_traditional_model = False
ckpt = torch.load('/Users/tomwoehrle/Documents/research_assistance/supervised_train/row-follow-epoch=11-val_loss=0.000404.ckpt')
traditional_model = RowfollowModel()
traditional_model.load_state_dict(ckpt['state_dict'])
traditional_model.eval()
# evaluator.model = traditional_model

if do_visual_test:
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
