import os
from dataclasses import dataclass

import mlflow
import pandas as pd
import torch
import torch.utils.data

import maml_api
import maml_eval

import rowfollow_task
import rowfollow_utils

"""
test config:
model + [inner_buffers, inner_lrs] :
    run_id
    episode
k
inner_steps
lr strategy
support collection
validation collections
"""


class RowfollowValDataset(torch.utils.data.Dataset):
    def __init__(self, validation_collections_paths: list[str], validation_annotations_file_path: str):
        self.validation_collections_paths: list[str] = validation_collections_paths
        self.validation_annotations_file_path: str = validation_annotations_file_path

        self.annotations_df: pd.DataFrame = pd.read_csv(self.validation_annotations_file_path)
        self.annotations_df = self._filter_existing_images()
        self.sigma = 10

    def _filter_existing_images(self):
        # Create a list of available image paths from the directories
        available_image_names = []
        for path in self.validation_collections_paths:
            for root, _, files in os.walk(path):
                for file in files:
                    if file.endswith(('jpg', 'jpeg', 'png')):  # Adjust extensions as needed
                        available_image_names.append(file)

        # Convert available image paths to a set for faster lookup
        available_image_names = set(available_image_names)

        # Filter the annotations to keep only those with available image files
        filtered_annotations = self.annotations_df[self.annotations_df['image_name'].isin(available_image_names)]
        return filtered_annotations.reset_index(drop=True)

    def __len__(self):
        return len(self.annotations_df)

    def __getitem__(self, idx):
        annotation_row = self.annotations_df.iloc[idx]
        image_name = annotation_row['image_name']
        collection_name = image_name.split('_cam')[0]

        collection_path = None
        for path in self.validation_collections_paths:
            if collection_name in path:
                collection_path = path
                break

        if collection_path:
            image_path = os.path.join(collection_path, image_name)
        else:
            raise FileNotFoundError(f"Collection {collection_name} not found in the provided paths.")

        vp, ll, lr = rowfollow_task.RowfollowTaskOldDataset.get_kps_for_image(image_name, annotation_row=annotation_row)

        pre_processed_image, _ = rowfollow_utils.pre_process_image_old_data(image_path, new_size=(320, 224))
        pre_processed_image = torch.from_numpy(pre_processed_image)

        # vp, ll, lr are coordinates, but we need distributions
        vp_gt = rowfollow_utils.dist_from_keypoint(vp, sig=self.sigma, downscale=4)
        ll_gt = rowfollow_utils.dist_from_keypoint(ll, sig=self.sigma, downscale=4)
        lr_gt = rowfollow_utils.dist_from_keypoint(lr, sig=self.sigma, downscale=4)

        return pre_processed_image, torch.stack([vp_gt, ll_gt, lr_gt])


@dataclass
class TestConfig:
    run_id: str
    episode: int
    k: int
    inner_steps: int
    support_collection_path: str
    support_annotations_file_path: str
    validation_collections_paths: list[str]
    validation_annotations_file_path: str
    device: torch.device


def load_model(run_id: str, episode: int) -> maml_api.MamlModel:
    model_uri = 'runs:/{}/{}/{}'.format(run_id, 'ep{}'.format(episode), 'model')

    return mlflow.pytorch.load_model(model_uri)


def load_inner_lrs(run_id: str, episode: int) -> maml_api.InnerLrs:
    artifact_uri = 'runs:/{}/{}/{}'.format(run_id, 'ep{}'.format(episode), 'inner_lrs.json')
    return mlflow.artifacts.load_dict(artifact_uri)


def load_inner_buffers(run_id: str, episode: int) -> maml_api.InnerBuffers:
    artifact_uri = 'runs:/{}/{}/{}'.format(run_id, 'ep{}'.format(episode), 'inner_buffers.json')
    inner_buffers = mlflow.artifacts.load_dict(artifact_uri)

    # at this point the inner_buffers do not contain tensors, but only a list -> this needs to be transformed
    for i, named_buffers in inner_buffers.items():
        for n, b in named_buffers.items():
            inner_buffers[i][n] = torch.tensor(b)
    return inner_buffers


def test_main(test_config: TestConfig):
    model = load_model(test_config.run_id, test_config.episode)
    inner_lrs = load_inner_lrs(test_config.run_id, test_config.episode)
    inner_buffers = load_inner_buffers(test_config.run_id, test_config.episode)

    # TODO seed the finetuning
    # TODO save the resulting model
    task = rowfollow_task.RowfollowTaskOldDataset(test_config.support_annotations_file_path,
                                                  test_config.support_collection_path,
                                                  test_config.k,
                                                  test_config.device)

    finetuner = maml_eval.MamlFinetuner(model, inner_lrs, inner_buffers, test_config.inner_steps, task)
    finetuner.finetune()

    val_dataset = RowfollowValDataset(test_config.validation_collections_paths,
                                      test_config.validation_annotations_file_path)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False)
    total_loss = 0.0
    for x, y in val_dataloader:
        y_hat = model(x)
        total_loss += task.calc_loss(y_hat, y, maml_api.Stage.VAL, maml_api.SetToSetType.TARGET).item()
        print('total_loss:', total_loss)

    print('total loss: {}'.format(total_loss))


if __name__ == '__main__':
    base_path = '/Users/tomwoehrle/Documents/research_assistance/evaluate_adaptation/vision_data_latest/'
    base_val_dir_path = os.path.join(base_path, 'val')
    base_train_dir_path = os.path.join(base_path, 'train')

    all_val_collections = [os.path.join(base_val_dir_path, d) for d in os.listdir(base_val_dir_path)
                           if os.path.isdir(os.path.join(base_val_dir_path, d))]

    config = TestConfig(
        run_id='af946f76673142368f2414f00259e51e',
        episode=9999,
        k=4,
        inner_steps=3,
        support_collection_path=os.path.join(base_train_dir_path, 'collection-150620'),
        support_annotations_file_path=os.path.join(base_train_dir_path, 'v2_annotations_train.csv'),
        validation_collections_paths=all_val_collections,
        validation_annotations_file_path=os.path.join(base_val_dir_path, 'v2_annotations_val.csv'),
        device=torch.device('cpu')
    )

    test_main(config)
