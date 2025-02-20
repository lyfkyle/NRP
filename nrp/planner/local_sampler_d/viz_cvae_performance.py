import os.path as osp
import sys
import os
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))

import torch
import numpy as np
import math
import pickle
from PIL import Image
import random
import argparse
import datetime

from local_sampler_g.model import VAE
import utils
from env.maze import Maze

CUR_DIR = osp.dirname(osp.abspath(__file__))

def visualize_nodes_local(occ_g, pos_samples, neg_samples, neutral_samples, start_pos, goal_pos, show=True, save=False, file_name=None):
    maze = Maze(gui=False, add_robot=False)
    # maze.disable_visual()
    if occ_g is not None:
        maze.load_occupancy_grid(occ_g, add_box=True)

    if len(neg_samples) > 0:
        for i, pos in enumerate(neg_samples):
            if not np.allclose(pos, start_pos):
                pos_tmp = pos.copy()
                pos_tmp[0] += 2
                pos_tmp[1] += 2
                maze.add_robot(pos_tmp, rgba=[0, 0, 0, 1])

    if len(neutral_samples) > 0:
        for i, pos in enumerate(neutral_samples):
            if not np.allclose(pos, start_pos):
                pos_tmp = pos.copy()
                pos_tmp[0] += 2
                pos_tmp[1] += 2
                maze.add_robot(pos_tmp, rgba=[0.5, 0.5, 0.5, 1])

    if len(pos_samples) > 0:
        for i, pos in enumerate(pos_samples):
            if not np.allclose(pos, start_pos):
                pos_tmp = pos.copy()
                pos_tmp[0] += 2
                pos_tmp[1] += 2
                maze.add_robot(pos_tmp, rgba=[1, 1, 1, 1])

    if start_pos is not None:
        start_pos_tmp = start_pos.copy()
        start_pos_tmp[0] += 2
        start_pos_tmp[1] += 2
        maze.add_robot(start_pos_tmp, rgba=[1, 1, 0, 1]) # yellow

    if goal_pos is not None:
        goal_pos_tmp = goal_pos.copy()
        if math.fabs(goal_pos[0]) > 2.0 or math.fabs(goal_pos[1]) > 2.0:
            goal_dir = math.atan2(goal_pos[1], goal_pos[0])

            if math.fabs(goal_pos[0]) < math.fabs(goal_pos[1]):
                goal_pos_tmp[1] = 5 if goal_pos[1] > 0 else -4
                goal_pos_tmp[0] = goal_pos_tmp[1] / math.tan(goal_dir)
            else:
                goal_pos_tmp[0] = 4 if goal_pos[0] > 0 else -4
                goal_pos_tmp[1] = goal_pos_tmp[0] * math.tan(goal_dir)

        print(goal_pos_tmp)
        goal_pos_tmp[0] += 2
        goal_pos_tmp[1] += 2
        maze.add_robot(goal_pos_tmp, rgba=[1, 0, 0, 0.2]) # red

    # maze.enable_visual()
    # input("Press anything to quit")

    img = maze.get_img()
    pil_img = Image.fromarray(img)

    if show:
        pil_img.show()
    if save:
        pil_img.save(file_name)

    return img

def visualize_nodes_local_2(occ_g, samples, scores, start_pos, goal_pos, show=True, save=False, file_name=None):
    maze = Maze(gui=False, add_robot=False)
    # maze.disable_visual()
    if occ_g is not None:
        maze.load_occupancy_grid(occ_g, add_box=True)

    if len(samples) > 0:
        for i, pos in enumerate(samples):
            if not np.allclose(pos, start_pos):
                pos_tmp = pos.copy()
                pos_tmp[0] += 2
                pos_tmp[1] += 2
                maze.add_robot(pos_tmp, rgba=[scores[i], scores[i], scores[i], 1])

    if start_pos is not None:
        start_pos_tmp = start_pos.copy()
        start_pos_tmp[0] += 2
        start_pos_tmp[1] += 2
        maze.add_robot(start_pos_tmp, rgba=[1, 1, 0, 1]) # yellow

    if goal_pos is not None:
        goal_pos_tmp = goal_pos.copy()
        if math.fabs(goal_pos[0]) > 4.0 or math.fabs(goal_pos[1]) > 4.0:
            goal_dir = math.atan2(goal_pos[1], goal_pos[0])

            if math.fabs(goal_pos[0]) < math.fabs(goal_pos[1]):
                goal_pos_tmp[1] = 4 if goal_pos[1] > 0 else -4
                goal_pos_tmp[0] = goal_pos_tmp[1] / math.tan(goal_dir)
            else:
                goal_pos_tmp[0] = 4 if goal_pos[0] > 0 else -4
                goal_pos_tmp[1] = goal_pos_tmp[0] * math.tan(goal_dir)

            goal_color = [0.5, 0, 0, 0.2]
        else:
            goal_color = [1, 0, 0, 0.2]

        goal_pos_tmp[0] += 2
        goal_pos_tmp[1] += 2
        maze.add_robot(goal_pos_tmp, rgba=goal_color) # red

    # maze.enable_visual()
    # input("Press anything to quit")

    img = maze.get_img()
    pil_img = Image.fromarray(img)

    if show:
        pil_img.show()
    if save:
        pil_img.save(file_name)

    return img

def run_model(selector, local_occ_grid, local_start_pos, local_goal_pos):
    with torch.no_grad():
        occ_grid_t = torch.tensor(local_occ_grid, device=device, dtype=torch.float).view(1, occ_grid_dim, occ_grid_dim)
        start_t = torch.tensor(local_start_pos, device=device, dtype=torch.float)
        goal_t = torch.tensor(local_goal_pos, device=device, dtype=torch.float)

        linkinfo = fk.get_link_positions(start_t.view(1, -1)).view(-1)
        start_t = torch.cat((start_t, linkinfo))
        linkinfo = fk.get_link_positions(goal_t.view(1, -1)).view(-1)
        goal_direction = torch.atan2(goal_t[1], goal_t[0]).view(1)
        goal_t = torch.cat((goal_t, linkinfo, goal_direction))
        context_t = torch.cat((start_t, goal_t), dim=-1)

        samples = selector.sample(20, occ_grid_t, context_t).cpu().numpy()

    return samples

def merge_images_2(file1, file2):
    image1 = Image.open(file1)
    image2 = Image.open(file2)

    (width, height) = image1.size
    image2 = image2.resize((width, height))

    result_width = width * 2
    result_height = height

    result = Image.new('RGB', (result_width, result_height))
    result.paste(im=image1, box=(0, 0))
    result.paste(im=image2, box=(width, 0))
    return result

def merge_images_4(file1, file2, file3, file4):
    """Merge two images into one, displayed side by side
    :param file1: path to first image file
    :param file2: path to second image file
    :return: the merged Image object
    """
    image1 = Image.open(file1)
    image2 = Image.open(file2)
    image3 = Image.open(file3)
    image4 = Image.open(file4)

    (width, height) = image1.size
    image2 = image2.resize((width, height))
    image3 = image3.resize((width, height))
    image4 = image4.resize((width, height))

    result_width = width + width + width + width
    result_height = height

    result = Image.new('RGB', (result_width, result_height))
    result.paste(im=image1, box=(0, 0))
    result.paste(im=image2, box=(width, 0))
    result.paste(im=image3, box=(width + width, 0))
    result.paste(im=image4, box=(width + width + width, 0))
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--name', default='')
    parser.add_argument('--test', action="store_true")
    args = parser.parse_args()

    # constants
    z_dim = 5
    robot_dim = 8
    linkpos_dim = 12
    occ_grid_dim = 40
    state_dim = robot_dim + linkpos_dim
    goal_dim = robot_dim + linkpos_dim + 1

    model_path = osp.join(CUR_DIR, "../local_sampler_g/models/cvae_sel.pt")

    if args.test:
        data_dir = osp.join(CUR_DIR, "./dataset/model_t")
        env_num = 5
        dataset_num = 50
    else:
        data_dir = osp.join(CUR_DIR, "./dataset/model")
        env_num = 25
        dataset_num = 2800

    now = datetime.datetime.now()
    if args.name == '':
        date_time = now.strftime("%m-%d-%Y-%H-%M-%S")
        res_dir = osp.join(CUR_DIR, "eval_res/{}".format(date_time))
    else:
        res_dir = osp.join(CUR_DIR, "eval_res/{}".format(args.name))
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    # hyperparameters
    data_cnt = 10000

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    fk = utils.FkTorch(device)
    model = VAE(z_dim, state_dim + goal_dim, state_dim, 512)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    iter_num = 0

    # Run in train env
    for env_idx in range(env_num):
        for _ in range(10):
            dataset_idx = random.randint(0, dataset_num)
            file_path = osp.join(data_dir, "data_{}_{}.pkl".format(env_idx, dataset_idx))

            if not os.path.exists(file_path):
                continue

            with open(file_path, 'rb') as f:
                datas = pickle.load(f)

            free_samples = []
            col_samples = []
            pos_samples = []
            neg_samples = []
            neutral_samples = []
            for i, data in enumerate(datas):
                occ_grid, start_pos, goal_pos, sample_pos, expert_pos, selected_path_len, expert_path_len = data

                if selected_path_len == -1:
                    col_samples.append(sample_pos)
                else:
                    free_samples.append((sample_pos, selected_path_len))

            free_samples.sort(key=lambda a: a[1])

            num_pos = max(1, int(len(free_samples) * 0.25))
            neg_idx = max(1, int(len(free_samples) * 0.5))
            pos_s = free_samples[:num_pos]
            if neg_idx < len(free_samples):
                neg_s = free_samples[neg_idx:]
                neural_s = free_samples[num_pos:neg_idx]
            else:
                neg_s = []
                neural_s = []

            pos_samples.append(expert_pos)
            for sample_pos, _ in pos_s:
                pos_samples.append(sample_pos)
            for sample_pos, _ in neg_s:
                neg_samples.append(sample_pos)
            for sample_pos, _ in neural_s:
                neutral_samples.append(sample_pos)

            # print(len(free_samples), len(pos_samples), len(neg_samples), len(neutral_samples), len(col_samples))
            free_samples = [s[0] for s in free_samples]

            # all_samples = free_samples + col_samples

            samples = run_model(model, occ_grid, start_pos, goal_pos)[:, :robot_dim]
            # print(samples)

            generated_samples_fig = os.path.join(res_dir, "generated_sample_{}_{}.png".format(env_idx, dataset_idx))
            # col_pred_fig = os.path.join(res_dir, "col_pred_{}_{}.png".format(env_idx, dataset_idx))
            # sel_pred_fig = os.path.join(res_dir, "sel_pred_{}_{}.png".format(env_idx, dataset_idx))
            # selected_pred_fig = os.path.join(res_dir, "selected_sample_{}_{}.png".format(env_idx, dataset_idx))
            # col_gt_fig = os.path.join(res_dir, "col_gt_{}_{}.png".format(env_idx, dataset_idx))
            # sel_gt_fig = os.path.join(res_dir, "sel_gt_{}_{}.png".format(env_idx, dataset_idx))
            utils.visualize_nodes_local(occ_grid, samples, start_pos, goal_pos, max_num=float('inf'), show=False, save=True, file_name=generated_samples_fig)
            # utils.visualize_nodes_local(occ_grid, [sorted_sample[-1]], start_pos, goal_pos, max_num=float('inf'), color_coding=True, show=False, save=True, file_name=selected_pred_fig)
            # visualize_nodes_local(occ_grid, free_samples, col_samples, [], start_pos, goal_pos, show=False, save=True, file_name=col_gt_fig)
            # visualize_nodes_local(occ_grid, pos_samples, neg_samples + col_samples, neutral_samples, start_pos, goal_pos, show=False, save=True, file_name=sel_gt_fig)

            # res_img = merge_images_4(generated_samples_fig, col_pred_fig, sel_pred_fig, selected_pred_fig)
            # res_img.save(os.path.join(res_dir, "viz_ext_{}_{}.png".format(env_idx, dataset_idx)))

            # res_img = merge_images_2(col_gt_fig, col_pred_fig)
            # res_img.save(os.path.join(res_dir, "viz_col_{}_{}.png".format(env_idx, dataset_idx)))

            # res_img = merge_images_2(sel_gt_fig, sel_pred_fig)
            # res_img.save(os.path.join(res_dir, "viz_sel_{}_{}.png".format(env_idx, dataset_idx)))