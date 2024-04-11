import argparse
import os
import random
import torch
import rowfollow_test_utils.keypoint_utils as kpu
from rowfollow_test_utils.utils import process_image, get_filenames, plot_multiple_color_maps, PlotInfo
from models import RowfollowModel
from tasks import RowfollowTask
from maml import inner_loop_update


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
    train_bags = ckpt['train_bags']
    # image_files = ckpt['test_bags']

    model = RowfollowModel()
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()  # does it already need to be set into eval here ?

    ###

    img_array = []
    lr_hm_array = []
    ll_hm_array = []
    vp_hm_array = []

    # image_dir = os.path.join(bags[0], 'left_cam')
    # image_dir = image_dir.replace(
    #     '/home/woehrle2/Documents/data', '/Users/tomwoehrle/Documents/research_assistance')
    # NOTE change to test_bags for true test:
    # TODO try out and think about k

    bag_path = random.choice(train_bags).replace(
        '/home/woehrle2/Documents/data', '/Users/tomwoehrle/Documents/research_assistance')
    print(bag_path)
    task = RowfollowTask(bag_path, 2, device)
    params, buffers = model.get_initial_state()
    inner_gradient_steps = 1
    for i in range(inner_gradient_steps):
        # NOTE has to be done like this, while inner_loop_update is fixed at 1 gradient step
        # not exactly how maml testing works, cause every time new sample i.e. k = k * inner_gradient_steps
        params = inner_loop_update(
            model, params, buffers, task, 0.4, 'doesnt matter')
    model.load_state_dict(params | buffers)

    n_tests = 20
    for i in range(n_tests):
        cam = random.choice(['left_cam', 'right_cam'])
        image_name = random.choice(get_filenames(os.path.join(bag_path, cam)))
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

    plot_multiple_color_maps(
        img_plotinfo_array, vp_hm_plotinfo_array, ll_hm_plotinfo_array, lr_hm_plotinfo_array)


if __name__ == '__main__':
    main()
