import argparse
import os
import random
import torch
import rowfollow_test_utils.keypoint_utils as kpu
from rowfollow_test_utils.utils import process_image, get_filenames, plot_multiple_color_maps, PlotInfo
from models import RowfollowModel
from tasks import RowfollowTask
from maml import inner_loop_update_for_testing


def main():
    parser = argparse.ArgumentParser(prog='python3 visualize_heatmaps.py',
                                     description='Given an input directory runs inference on all pictures inside it and outputs a pdf with the different predictions channels visualized. ',
                                     epilog='NOTES: Directory names must be specified without "/" at the end.'
                                     )
    parser.add_argument('ckpt_path', metavar='', type=str,
                        help='Path to the trained model parameters')
    args = parser.parse_args()
    ###

    device = torch.device("cpu")  # NOTE change to cuda if available
    ckpt = torch.load(args.ckpt_path, map_location=device)
    # train_bags = ckpt['train_bags']
    # image_files = ckpt['test_bags']

    model = RowfollowModel()
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)

    ###

    img_array = []
    lr_hm_array = []
    ll_hm_array = []
    vp_hm_array = []

    # NOTE HYPERPARAMETERS NOTE
    # bag_path = random.choice(train_bags).replace(
    #     '/home/woehrle2/Documents/data', '/Users/tomwoehrle/Documents/research_assistance')
    # bag_path = '/Users/tomwoehrle/Documents/research_assistance/cornfield1_labeled_new/20220714_cornfield/ts_2022_07_14_12h17m57s_two_random/'
    bag_path = '/Users/tomwoehrle/Documents/research_assistance/cornfield1_labeled_new/20220603_cornfield/ts_2022_06_03_02h54m54s/'
    # bag_path = '/Users/tomwoehrle/Documents/research_assistance/cornfield1_labeled_new/20221006_cornfield/ts_2022_10_06_10h06m49s_two_random/'
    k = 1
    inner_gradient_steps = 1
    anil = False
    num_episodes = 60000  # shouldnt matter w/o sigma scheduling if task set up correctly
    current_ep = 45000  # same here

    task = RowfollowTask(bag_path, k, device,
                         num_episodes=num_episodes, sigma=5)
    params, buffers = model.get_initial_state()
    params = inner_loop_update_for_testing(anil,
                                           model, params, buffers, task, 0.4, inner_gradient_steps, current_ep=current_ep)
    model.load_state_dict(params | buffers)
    # model.eval() # NOTE why not needed/working?

    n_tests = 30
    for i in range(n_tests):
        # cam = random.choice(['left_cam', 'right_cam'])
        cam = 'right_cam'
        # image_name = random.choice(get_filenames(os.path.join(bag_path, cam)))
        image_name = sorted(get_filenames(os.path.join(bag_path, cam)))[i]
        full_path = os.path.join(bag_path, cam, image_name)
        print(full_path)
        np_array, image = process_image(full_path)
        pred = model(torch.from_numpy(np_array).to(
            device)).detach().cpu().numpy()[0]

        image_with_lines = kpu.image_with_lines_from_pred(image, pred)

        complete_dist, vp_dist, ll_dist, lr_dist = kpu.get_dists(
            pred, resize_to=image.shape[:2][::-1])

        img_array.append(image_with_lines)
        vp_hm_array.append(vp_dist)
        ll_hm_array.append(ll_dist)
        lr_hm_array.append(lr_dist)

    img_plotinfo_array = [
        PlotInfo(img, isImage=True) for img in img_array]
    vp_hm_plotinfo_array = [
        PlotInfo(hm, isImage=False, gradientColor="red") for hm in vp_hm_array]
    ll_hm_plotinfo_array = [PlotInfo(
        hm, isImage=False, gradientColor="green", vmin=0, vmax=1) for hm in ll_hm_array]
    lr_hm_plotinfo_array = [PlotInfo(
        hm, isImage=False, gradientColor="blue", vmin=0, vmax=1) for hm in lr_hm_array]

    title = f'ckpt: {args.ckpt_path} \n bag: {bag_path} \n k={
        k}, steps={inner_gradient_steps}, no model.eval()'
    plot_multiple_color_maps(
        img_plotinfo_array, vp_hm_plotinfo_array, ll_hm_plotinfo_array, lr_hm_plotinfo_array, full_title=title)


if __name__ == '__main__':
    main()
