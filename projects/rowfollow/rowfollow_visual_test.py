import os
import sys

import cv2
import torch
import torch.utils.data

from rf_utils import viz

import maml_eval
import rowfollow_utils
from projects.rowfollow import rowfollow_task
from projects.rowfollow.rowfollow_test import load_model, load_inner_lrs, load_inner_buffers, TestConfig, get_config_from_file, get_model_from_ckpt_file


def main(config: TestConfig):
    if config.path_to_ckpt_file is None:
        model = load_model(config.run_id, config.episode)
        inner_lrs = load_inner_lrs(config.run_id, config.episode)
        inner_buffers = load_inner_buffers(config.run_id, config.episode)
    else:
        model = get_model_from_ckpt_file(config.path_to_ckpt_file)

    task = rowfollow_task.RowfollowTaskOldDataset(config.support_annotations_file_path,
                                                  config.support_collection_path,
                                                  config.k,
                                                  torch.device(config.device),
                                                  sigma=config.sigma,
                                                  seed=config.seed)

    if config.path_to_ckpt_file is None:
        finetuner = maml_eval.MamlFinetuner(model, inner_lrs, inner_buffers, config.inner_steps, task, config.use_anil)
        finetuner.finetune()

    model.eval()

    target_directory = config.support_collection_path

    for img_name in os.listdir(target_directory):
        img_path = os.path.join(target_directory, img_name)
        if not (img_path.endswith('.png') or img_path.endswith('.jpg')):
            continue
        data, img = rowfollow_utils.pre_process_image_old_data(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        data = torch.from_numpy(data).unsqueeze(0)

        y_hat = model(data)

        img_with_lines = viz.img_with_lines_from_pred(img, y_hat)
        cv2.imshow('', img_with_lines)

        cv2.waitKey()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] == '-h':
        print('USAGE: python rowfollow_visual_test.py path/to/test_config.json')
        sys.exit(1)

    if not os.path.exists(sys.argv[1]):
        print(f'Path {sys.argv[1]} does not exist')
        sys.exit(1)

    path_to_config = sys.argv[1]
    main(get_config_from_file(path_to_config))
