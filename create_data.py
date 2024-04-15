import argparse
import ast
import os

import pandas as pd
import torch

from rowfollow_utils import pre_process_image


def unify_labels(dir_path, days):
    unified_df = pd.DataFrame()

    for day in days:
        day_path = os.path.join(dir_path, day)
        if os.path.isdir(day_path):
            for bag in os.listdir(day_path):
                bag_path = os.path.join(day_path, bag)
                if os.path.isdir(bag_path):
                    for cam_dir in ['left_cam', 'right_cam']:
                        labels_path = os.path.join(
                            bag_path, cam_dir, 'labels.csv')
                        if os.path.exists(labels_path):
                            df = pd.read_csv(labels_path)
                            df['day'] = day
                            df['bag'] = bag
                            df['camera'] = cam_dir
                            # Append to the main DataFrame
                            unified_df = pd.concat(
                                [unified_df, df], ignore_index=True)
    return unified_df


def main(data_dir_path, output_path, first_k_days, last_k_days, n, seed):
    device = torch.device('cpu')
    test_days = sorted(os.listdir(data_dir_path))
    test_days = test_days[:first_k_days] + test_days[-last_k_days:]

    labels_df = unify_labels(data_dir_path, test_days)

    samples = labels_df.sample(n, random_state=seed)

    x = []
    y = []

    for idx, sample in samples.iterrows():
        image_path = os.path.join(
            data_dir_path, sample.day, sample.bag, sample.camera, sample.image_name)
        pre_processed_image, _ = pre_process_image(image_path)
        pre_processed_image = torch.from_numpy(pre_processed_image)
        x.append(pre_processed_image)
        vp, ll, lr = ast.literal_eval(sample.vp), ast.literal_eval(
            sample.ll), ast.literal_eval(sample.lr)
        y.append(torch.tensor([vp, ll, lr]))

    x = torch.stack(x)  # .to(device)
    y = torch.stack(y)  # .to(device)

    torch.save(x, os.path.join(output_path, 'x.pt'))
    torch.save(y, os.path.join(output_path, 'y.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir_path')
    parser.add_argument('output_path')
    parser.add_argument('--first_k_days', type=int, default=4)
    parser.add_argument('--last_k_days', type=int, default=5)
    parser.add_argument('--n', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    main(args.data_dir_path, args.output_path, args.first_k_days,
         args.last_k_days, args.n, args.seed)
