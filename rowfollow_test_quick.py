import argparse
import ast
import csv
import math
import os

import cv2
import pandas as pd
import torch
import torch.nn as nn

import maml
import models
import rowfollow_utils as utils
# NOTE fix these imports once rf_utils submodule is introduced
from rowfollow_test_utils.utils import plot_multiple_color_maps, PlotInfo
from rowfollow_test_utils.keypoint_utils import img_with_lines, Line
import shared_utils
import tasks


def write_to_csv(file_path, content: dict):
    with open(file_path, 'a') as csv_file:
        fieldnames = content.keys()

        csv_writer = csv.DictWriter(csv_file, fieldnames)

        if csv_file.tell() == 0:
            csv_writer.writeheader()

        csv_writer.writerow(content)  # this assumes that header fields match


def get_eval_data(base_path):
    data = pd.DataFrame()
    for cam_dir in ['left_cam', 'right_cam']:
        path = os.path.join(base_path, cam_dir, 'labels.csv')
        df = pd.read_csv(path, delimiter=';')
        df['cam_side'] = cam_dir
        data = pd.concat(
            [data, df], ignore_index=True)
    return data


def draw_pred_on_image(img, vp, ll, lr):
    img = cv2.circle(
        img, vp, 5, (255, 0, 0), -1)

    lines = [Line(vp, ll, (0, 255, 0), 2),
             Line(vp, lr, (0, 0, 255), 2)]
    return img_with_lines(img, lines)


def get_data(base_path, df, device):
    x = []
    y = []
    imgs = []

    for idx, row in df.iterrows():
        image_path = os.path.join(
            base_path, row.cam_side, row.image_name)
        pre_processed_image, img = utils.pre_process_image(image_path)
        pre_processed_image = torch.from_numpy(pre_processed_image)
        x.append(pre_processed_image)
        vp, ll, lr = ast.literal_eval(row.vp), ast.literal_eval(
            row.ll), ast.literal_eval(row.lr)
        y.append(torch.tensor([vp, ll, lr]))
        imgs.append(img)

    x = torch.stack(x).to(device)
    y = torch.stack(y).to(device)

    return x, y, imgs


def calc_loss(ckpt_path, stage, device, seed, no_finetuning, k, inner_gradient_steps, alpha, sigma):
    if stage == 'early':
        finetune_data_path = '/Users/tomwoehrle/Documents/research_assistance/cornfield1_labeled_new/20220603_cornfield/ts_2022_06_03_02h54m54s'
        eval_data_path = '/Users/tomwoehrle/Documents/research_assistance/maml_tryout/test_data/20220603_cornfield-10handpicked/ts_2022_06_03_02h54m54s'
    elif stage == 'late':
        finetune_data_path = '/Users/tomwoehrle/Documents/research_assistance/cornfield1_labeled_new/20221006_cornfield/ts_2022_10_06_10h29m23s_two_random'
        eval_data_path = '/Users/tomwoehrle/Documents/research_assistance/maml_tryout/test_data/20221006_cornfield-10handpicked/ts_2022_10_06_10h29m23s_two_random'
    elif stage == 'middle':
        finetune_data_path = '/Users/tomwoehrle/Documents/research_assistance/cornfield1_labeled_new/20220714_cornfield/ts_2022_07_14_12h17m57s_two_random/'
        eval_data_path = '/Users/tomwoehrle/Documents/research_assistance/maml_tryout/test_data/20220714_cornfield-10handpicked/ts_2022_07_14_12h17m57s_two_random/'

    ckpt = torch.load(ckpt_path, map_location=device)
    model = models.RowfollowModel()
    state_dict = ckpt.get('model_state_dict', ckpt.get('state_dict'))
    model.load_state_dict(state_dict)
    model.to(device)

    anil = ckpt.get('anil', False)

    if not no_finetuning:
        params, buffers = model.get_initial_state()
        task = tasks.RowfollowTask(finetune_data_path, k,
                                   device, seed=seed, sigma=sigma)
        params = maml.inner_loop_update_for_testing(anil,
                                                    model, params, buffers, task, alpha, inner_gradient_steps)
        model.load_state_dict(params | buffers)

    # model.eval() # NOTE why not needed/working?

    labels = get_eval_data(eval_data_path)
    loss_fct = nn.L1Loss(reduction='none')

    x, y, imgs = get_data(eval_data_path, labels, device)

    x_hat = shared_utils.get_indices_from_pred(
        model(x)) * 4

    x_hat_on_frame = []
    for idx, pred in enumerate(x_hat):
        vp = pred[0]
        ll = pred[1]
        lr = pred[2]
        vp = vp[0].item(), vp[1].item()
        ll = shared_utils.get_coordinates_on_frame(vp, ll)
        lr = shared_utils.get_coordinates_on_frame(vp, lr)
        x_hat_on_frame.append(torch.tensor([vp, ll, lr]))
        imgs[idx] = draw_pred_on_image(imgs[idx], vp, ll, lr)

    x_hat_on_frame = torch.stack(x_hat_on_frame)

    loss = loss_fct(x_hat, y)
    loss = torch.sum(loss, dim=2).float()
    loss_on_frame = loss_fct(x_hat_on_frame, y)
    loss_on_frame = torch.sum(loss_on_frame, dim=2).float()

    imgs = [PlotInfo(img, isImage=True) for img in imgs]

    return loss, loss_on_frame, torch.mean(loss, dim=0), torch.mean(loss_on_frame, 0), imgs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt_path', type=str,
                        help='Path to the trained model parameters')
    parser.add_argument('stage', type=str)
    parser.add_argument('--n_runs', default=10, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--k', default=5, type=int)
    parser.add_argument('--inner_gradient_steps', default=1, type=int)
    parser.add_argument('--alpha', default=0.4, type=float)
    parser.add_argument('--sigma', default=10, type=int)
    parser.add_argument('--no_finetuning', default=False, type=bool)
    parser.add_argument(
        '--results_file', default='results/quick_results.csv', type=str)
    args = parser.parse_args()

    losses = []  # n_runsxNx3
    losses_on_frame = []  # n_runsxNx3
    mean_losses = []  # n_runsx3, where the Nx3 were reduced to 3 by taking the mean. means per run are stored here
    mean_losses_on_frame = []  # n_runsx3
    all_imgs = []  # n_runsxN
    for i in range(args.n_runs):
        print('Run', i+1, 'out of', args.n_runs, 'starting...')
        loss, loss_on_frame, mean_loss, mean_loss_on_frame, imgs = calc_loss(ckpt_path=args.ckpt_path,
                                                                             stage=args.stage,
                                                                             device=torch.device(
                                                                                 'cpu'),
                                                                             # maybe make device variable
                                                                             seed=args.seed+i,
                                                                             no_finetuning=args.no_finetuning,
                                                                             k=args.k,
                                                                             inner_gradient_steps=args.inner_gradient_steps,
                                                                             alpha=args.alpha,
                                                                             sigma=args.sigma)
        losses.append(loss)
        losses_on_frame.append(loss_on_frame)
        mean_losses.append(mean_loss)
        mean_losses_on_frame.append(mean_loss_on_frame)
        all_imgs.append(imgs)

    losses = torch.cat(losses)
    losses_on_frame = torch.cat(losses_on_frame)
    mean_losses = torch.stack(mean_losses)
    mean_losses_on_frame = torch.stack(mean_losses_on_frame)

    kp_onframe_across_mean = torch.mean(losses_on_frame, dim=0)
    kp_onframe_across_stddev = torch.sqrt(torch.var(losses_on_frame, dim=0))
    kp_onframe_between_stddev = torch.sqrt(
        torch.var(mean_losses_on_frame, dim=0))
    print('------------RESULTS------------')
    print('kp_onframe_across_mean:', kp_onframe_across_mean)
    print('kp_onframe_across_stddev:', kp_onframe_across_stddev)
    print('kp_onframe_between_stddev:', kp_onframe_between_stddev)
    print('------------------------------')
    print('SIMPLIFIED (take with a grain of salt):')
    prediction_performance = math.floor(
        100-kp_onframe_across_mean.sum().item())
    prediction_stability = math.floor(
        100-kp_onframe_across_stddev.sum().item())
    fine_tune_stability = math.floor(
        100-kp_onframe_between_stddev.sum().item())
    print('Prediction performance:', prediction_performance)
    print('Prediction stability:', prediction_stability)
    print('Fine-tune stability:', fine_tune_stability)

    content = {
        'kp_onframe_across_mean': kp_onframe_across_mean,
        'kp_onframe_across_stddev': kp_onframe_across_stddev,
        'kp_onframe_between_stddev': kp_onframe_between_stddev,
        'prediction_performance': prediction_performance,
        'prediction_stability': prediction_stability,
        'fine_tune_stability': fine_tune_stability,
        **vars(args)
    }

    write_to_csv(args.results_file, content)

    plot_multiple_color_maps(*all_imgs)
