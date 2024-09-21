import os
from typing import Optional

import cv2
import torch
import torch.utils.data

from rf_utils import viz

import maml_eval
import rowfollow_utils
from projects.rowfollow import rowfollow_task
from projects.rowfollow.rowfollow_test import load_model, load_inner_lrs, load_inner_buffers, RowfollowValDataset


def main(run_id: str,
         episode: int,
         k: int,
         inner_steps: int,
         support_collection_path: str,
         support_annotations_file_path: str,
         target_directory: str,
         device: torch.device,
         seed: Optional[int],
         sigma: int):
    model = load_model(run_id, episode)
    inner_lrs = load_inner_lrs(run_id, episode)
    inner_buffers = load_inner_buffers(run_id, episode)

    task = rowfollow_task.RowfollowTaskOldDataset(support_annotations_file_path,
                                                  support_collection_path,
                                                  k,
                                                  device,
                                                  sigma=sigma,
                                                  seed=seed)

    finetuner = maml_eval.MamlFinetuner(model, inner_lrs, inner_buffers, inner_steps, task)
    finetuner.finetune()

    model.eval()

    total_loss = 0.0
    batches_processed = 0
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
    run_id = 'b1dfed24fba443c6ac96b2c1b08c44d6'
    episode = 9000
    k = 8
    inner_steps = 3
    base_path = '/Users/tomwoehrle/Documents/research_assistance/evaluate_adaptation/vision_data_latest/'
    support_collection_path = os.path.join(base_path, 'train', 'late_season')
    support_annotations_file_path = os.path.join(base_path, 'train', 'v2_annotations_train.csv')
    target_directory = support_collection_path
    device = torch.device('cpu')
    seed = 0
    sigma = 10

    main(
        run_id,
        episode,
        k,
        inner_steps,
        support_collection_path,
        support_annotations_file_path,
        target_directory,
        device,
        seed,
        sigma
    )
