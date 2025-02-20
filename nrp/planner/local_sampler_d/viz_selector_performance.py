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
import matplotlib.pyplot as plt

from model import SelectModelSmall
import utils
from env.maze import Maze

CUR_DIR = osp.dirname(osp.abspath(__file__))

def visualize_nodes_local(occ_g, pos_samples, neg_samples, neutral_samples, start_pos, goal_pos, show=True, save=False, file_name=None):
    fig1 = plt.figure(figsize=(10, 10), dpi=250)

    occ_grid_size = occ_g.shape[0]
    tmp = occ_grid_size // 2
    # s = 225
    s = (10 / occ_g.shape[0] * 60) ** 2

    ax1 = fig1.add_subplot(111, aspect="equal")

    for i in range(occ_g.shape[0]):
        for j in range(occ_g.shape[1]):
            if occ_g[i, j] == 1:
                plt.scatter(
                    (i - tmp + 0.5) * 0.1,
                    (j - tmp + 0.5) * 0.1,
                    color="black",
                    marker="s",
                    s=s,
                    alpha=1,
                )  # init

    if len(neg_samples) > 0:
        for i, pos in enumerate(neg_samples):
            if not np.allclose(pos, start_pos):
                utils.visualize_robot(pos, color=[[0.2, 0.2, 0.2, 1]], ax=ax1)

    if len(neutral_samples) > 0:
        for i, pos in enumerate(neutral_samples):
            if not np.allclose(pos, start_pos):
                utils.visualize_robot(pos, color=[[0.6, 0.6, 0.6, 1]], ax=ax1)

    if len(pos_samples) > 0:
        for i, pos in enumerate(pos_samples):
            if not np.allclose(pos, start_pos):
                utils.visualize_robot(pos, color=[[0, 1, 0, 1]], ax=ax1)

    if start_pos is not None:
        utils.visualize_robot(start_pos, color=[[1, 1, 0, 1]], ax=ax1)

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
        utils.visualize_robot(goal_pos_tmp, color=[[1, 0, 0, 0.2]], ax=ax1)

    # maze.enable_visual()
    # input("Press anything to quit")

    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    if show:
        plt.show()
    if save:
        fig1.savefig(file_name, dpi=fig1.dpi)
        plt.close()

def visualize_nodes_local_2(occ_g, samples, scores, start_pos, goal_pos, show=True, save=False, file_name=None):
    fig1 = plt.figure(figsize=(10, 10), dpi=250)

    occ_grid_size = occ_g.shape[0]
    tmp = occ_grid_size // 2
    # s = 225
    s = (10 / occ_g.shape[0] * 60) ** 2

    ax1 = fig1.add_subplot(111, aspect="equal")

    for i in range(occ_g.shape[0]):
        for j in range(occ_g.shape[1]):
            if occ_g[i, j] == 1:
                plt.scatter(
                    (i - tmp + 0.5) * 0.1,
                    (j - tmp + 0.5) * 0.1,
                    color="black",
                    marker="s",
                    s=s,
                    alpha=1,
                )  # init


    if len(samples) > 0:
        for i, pos in enumerate(samples):
            if not np.allclose(pos, start_pos):
                utils.visualize_robot(pos, color=[[0, scores[i], 0, 1]], ax=ax1)

    if start_pos is not None:
        utils.visualize_robot(start_pos, color=[[1, 1, 0, 1]], ax=ax1)

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

        utils.visualize_robot(goal_pos_tmp, color=[goal_color], ax=ax1)

    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    if show:
        plt.show()
    if save:
        fig1.savefig(file_name, dpi=fig1.dpi)
        plt.close()

def run_selector(selector, local_occ_grid, local_start_pos, local_goal_pos, local_sample_pos):
    with torch.no_grad():
        occ_grid_t = torch.tensor(local_occ_grid, device=device, dtype=torch.float).view(1, occ_grid_dim, occ_grid_dim)
        start_t = torch.tensor(local_start_pos, device=device, dtype=torch.float)
        goal_t = torch.tensor(local_goal_pos, device=device, dtype=torch.float)
        samples_t = torch.tensor(local_sample_pos, device=device, dtype=torch.float)

        # linkpos = fk.get_link_positions(samples_t.view(-1, robot_dim)).view(1, -1, linkpos_dim)
        # samples_t = torch.cat((samples_t, linkpos), dim=-1)
        linkinfo = fk.get_link_positions(start_t.view(1, -1)).view(-1)
        start_t = torch.cat((start_t, linkinfo))
        linkinfo = fk.get_link_positions(goal_t.view(1, -1)).view(-1)
        goal_direction = torch.atan2(goal_t[1], goal_t[0]).view(1)
        goal_t = torch.cat((goal_t, linkinfo, goal_direction))
        linkinfo = fk.get_link_positions(samples_t)
        samples_t = torch.cat((samples_t, linkinfo), dim=-1)

        occ_grid_batch = occ_grid_t.unsqueeze(0) # 1 x 4 x occ_grid_dim x occ_grid_dim x occ_grid_dim_z
        start_batch = start_t.unsqueeze(0) # 1 x dim
        goal_batch = goal_t.unsqueeze(0) # 1 x dim
        samples_batch = samples_t.unsqueeze(0) # 1 x N x dim
        col_scores, sel_scores = selector(occ_grid_batch, start_batch, goal_batch, samples_batch, fixed_env=True)
        col_scores = col_scores.view(-1)
        sel_scores = sel_scores.view(-1)

    return col_scores, sel_scores

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
    robot_dim = 8
    linkpos_dim = 12
    occ_grid_dim = 40
    goal_dim = robot_dim + linkpos_dim + 1

    col_checker_path = osp.join(CUR_DIR, "../local_sampler_d/models/model_col_dagger.pt")
    selector_path = osp.join(CUR_DIR, "../local_sampler_d/models/model_sel_small_v2.pt")

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
        os.makedirs(osp.join(res_dir, "viz_col"))
        os.makedirs(osp.join(res_dir, "viz_sel"))
        os.makedirs(osp.join(res_dir, "viz_ext"))

    # hyperparameters
    data_cnt = 10000

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    fk = utils.FkTorch(device)
    model = SelectModelSmall(robot_dim, occ_grid_dim, col_checker_path, selector_path)
    model.to(device)
    model.eval()

    iter_num = 0
    # while data_cnt < target_data_cnt:

    # Run in train env
    for env_idx in range(env_num):
        for _ in range(10):
            dataset_idx = random.randint(0, dataset_num)
        # for dataset_idx in [35, 51]:
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

            print(len(free_samples), len(pos_samples), len(neg_samples), len(neutral_samples), len(col_samples))
            free_samples = [s[0] for s in free_samples]

            all_samples = free_samples + col_samples

            col_scores, sel_scores = run_selector(model, occ_grid, start_pos, goal_pos, all_samples)
            col_scores = col_scores.cpu().numpy()
            sel_scores = sel_scores.cpu().numpy()
            sorted_indices = np.argsort(sel_scores)
            sorted_sample = np.array(all_samples)[sorted_indices] # small to large

            all_samples_fig = os.path.join(res_dir, "all_sample_{}_{}.png".format(env_idx, dataset_idx))
            col_pred_fig = os.path.join(res_dir, "col_pred_{}_{}.png".format(env_idx, dataset_idx))
            sel_pred_fig = os.path.join(res_dir, "sel_pred_{}_{}.png".format(env_idx, dataset_idx))
            selected_pred_fig = os.path.join(res_dir, "selected_sample_{}_{}.png".format(env_idx, dataset_idx))
            col_gt_fig = os.path.join(res_dir, "col_gt_{}_{}.png".format(env_idx, dataset_idx))
            sel_gt_fig = os.path.join(res_dir, "sel_gt_{}_{}.png".format(env_idx, dataset_idx))
            utils.visualize_nodes_local(occ_grid, all_samples, start_pos, goal_pos, max_num=float('inf'), show=False, save=True, file_name=all_samples_fig)
            visualize_nodes_local_2(occ_grid, all_samples, col_scores, start_pos, goal_pos, show=False, save=True, file_name=col_pred_fig)
            visualize_nodes_local_2(occ_grid, all_samples, sel_scores, start_pos, goal_pos, show=False, save=True, file_name=sel_pred_fig)
            utils.visualize_nodes_local(occ_grid, [sorted_sample[-1]], start_pos, goal_pos, max_num=float('inf'), show=False, save=True, file_name=selected_pred_fig)
            visualize_nodes_local(occ_grid, free_samples, col_samples, [], start_pos, goal_pos, show=False, save=True, file_name=col_gt_fig)
            visualize_nodes_local(occ_grid, pos_samples, neg_samples + col_samples, neutral_samples, start_pos, goal_pos, show=False, save=True, file_name=sel_gt_fig)

            res_img = merge_images_4(all_samples_fig, col_pred_fig, sel_pred_fig, selected_pred_fig)
            res_img.save(os.path.join(res_dir, "viz_ext/viz_ext_{}_{}.png".format(env_idx, dataset_idx)))

            res_img = merge_images_2(col_gt_fig, col_pred_fig)
            res_img.save(os.path.join(res_dir, "viz_col/viz_col_{}_{}.png".format(env_idx, dataset_idx)))

            res_img = merge_images_2(sel_gt_fig, sel_pred_fig)
            res_img.save(os.path.join(res_dir, "viz_sel/viz_sel_{}_{}.png".format(env_idx, dataset_idx)))