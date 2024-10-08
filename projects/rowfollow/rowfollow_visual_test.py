import os
import sys

import cv2

import torch
import torch.utils.data

import maml_api
from rf_utils import viz

import maml_eval
import rowfollow_utils
from projects.rowfollow import rowfollow_task
from projects.rowfollow.rowfollow_test import load_model, load_inner_lrs, load_inner_buffers, TestConfig, get_config_from_file, get_model_from_ckpt_file, RowfollowValDataset


def main(config: TestConfig):
    if config.use_from_pth:
        model = get_model_from_ckpt_file(config.path_to_pth)
    else:
        model = load_model(config.run_id, config.episode)
        inner_lrs = load_inner_lrs(config.run_id, config.episode)
        inner_buffers = load_inner_buffers(config.run_id, config.episode, torch.device(config.device))

    collection_path = os.path.join(config.base_path, config.visual_test_collection)

    task = rowfollow_task.RowfollowTaskOldDataset(config.annotations_file_path,
                                                  collection_path,
                                                  config.k,
                                                  torch.device(config.device),
                                                  sigma=config.sigma,
                                                  seed=config.seed)

    # TODO add finetuning for nonMAML
    if not config.use_from_pth:
        finetuner = maml_eval.MamlFinetuner(model, inner_lrs, inner_buffers, config.inner_steps, task, config.use_anil)
        finetuner.finetune()

    model.eval()

    dataset = RowfollowValDataset(collection_path, config.annotations_file_path, torch.device(config.device))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

    for idx, (x, y) in enumerate(dataloader):
        # loss calculation
        y_hat = model(x)
        l1_loss = task.calc_loss(y_hat, y, maml_api.Stage.VAL, maml_api.SetToSetType.TARGET)

        # display image
        img = rowfollow_utils.reverse_preprocessing(x.squeeze())

        # display image w/ gt lines
        # img_with_lines = viz.img_with_lines_from_pred(img, y)
        cv2.imshow(f'{idx}-gt {str(l1_loss.item())}', img)

        # display image w/ predicted lines
        img_with_lines = viz.img_with_lines_from_pred(img, y_hat)
        cv2.imshow(f'{idx}-hat {str(l1_loss.item())}', img_with_lines)

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
