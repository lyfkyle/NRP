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

from model import Selector
import utils
from env.maze import Maze

CUR_DIR = osp.dirname(osp.abspath(__file__))

def visualize_nodes_local(occ_g, pos_samples, neg_samples, neutral_samples, start_pos, goal_pos, show=True, save=False, file_name=None):
    fig1 = plt.figure(figsize=(10, 10), dpi=250)

    occ_grid_size = occ_g.shape[0]
    tmp = occ_grid_size // 2
    # s = 225
    s = (10 / occ_g.shape[0] * 60) ** 2

    ax = fig1.add_subplot(111, aspect="equal")

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
                utils.visualize_robot(pos, color=[[0.2, 0.2, 0.2, 1]], ax=ax)

    if len(neutral_samples) > 0:
        for i, pos in enumerate(neutral_samples):
            if not np.allclose(pos, start_pos):
                utils.visualize_robot(pos, color=[[0.6, 0.6, 0.6, 1]], ax=ax)

    if len(pos_samples) > 0:
        for i, pos in enumerate(pos_samples):
            if not np.allclose(pos, start_pos):
                utils.visualize_robot(pos, color=[[0, 1, 0, 1]], ax=ax)

    if start_pos is not None:
        utils.visualize_robot(start_pos, start=True, ax=ax) # yellow

    if goal_pos is not None:
        goal_pos_tmp = goal_pos.copy()
        if math.fabs(goal_pos[0]) > 2.0 or math.fabs(goal_pos[1]) > 2.0:
            goal_dir = math.atan2(goal_pos[1], goal_pos[0])

            if math.fabs(goal_pos[0]) < math.fabs(goal_pos[1]):
                goal_pos_tmp[1] = 2.0 if goal_pos[1] > 0 else -2.0
                goal_pos_tmp[0] = goal_pos_tmp[1] / math.tan(goal_dir)
            else:
                goal_pos_tmp[0] = 2.0 if goal_pos[0] > 0 else -2.0
                goal_pos_tmp[1] = goal_pos_tmp[0] * math.tan(goal_dir)

        utils.visualize_robot(goal_pos_tmp, goal=True, ax=ax) # red

    plt.title("Visualization")
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    if show:
        plt.show()
    if save:
        fig1.savefig(file_name, dpi=fig1.dpi)
        plt.close()

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

def run_selector(selector, local_occ_grid, local_start_pos, local_goal_pos, local_sample_pos):
    with torch.no_grad():
        occ_grid = torch.tensor(local_occ_grid, device=device, dtype=torch.float).view(occ_grid_dim, occ_grid_dim, occ_grid_dim_z)
        occ_grid_t = utils.add_pos_channels(occ_grid)
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
        sel_scores = selector(occ_grid_batch, start_batch, goal_batch, samples_batch, fixed_env=True).view(-1)

    return sel_scores

def merge_images(file1, file2, file3, file4):
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
    robot_dim = 11
    linkpos_dim = 24
    occ_grid_dim = 40
    occ_grid_dim_z = 20
    goal_dim = robot_dim + linkpos_dim + 1
    model_name = args.checkpoint
    model_path = osp.join(CUR_DIR, "models/{}.pt".format(model_name))
    if args.test:
        data_dir = osp.join(CUR_DIR, "./dataset/model_test")
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
    # model = Selector(robot_dim + linkpos_dim, goal_dim)
    # model.load_state_dict(torch.load(model_path))
    # model.to(device)
    # model.eval()

    iter_num = 0
    # while data_cnt < target_data_cnt:

    # Run in train env
    for env_idx in range(env_num):
        for _ in range(2):
            dataset_idx = random.randint(0, dataset_num)
        # for dataset_idx in [35, 51]:
            file_path = osp.join(data_dir, "data_{}_{}.pkl".format(env_idx, dataset_idx))
            print(file_path)
            if not os.path.exists(file_path):
                continue

            # print(f"Opening {file_path}")

            with open(file_path, 'rb') as f:
                datas = pickle.load(f)

            # print(f"Opening {file_path}"
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

            for sample_pos, _ in pos_s:
                pos_samples.append(sample_pos)
            for sample_pos, _ in neg_s:
                neg_samples.append(sample_pos)
            for sample_pos, _ in neural_s:
                neutral_samples.append(sample_pos)

            print(len(free_samples), len(pos_samples), len(neg_samples), len(neutral_samples), len(col_samples))
            free_samples = [s[0] for s in free_samples]

            img1 = os.path.join(res_dir, "free_sample_{}_{}.png".format(env_idx, dataset_idx))
            img2 = os.path.join(res_dir, "gt_sel_label_{}_{}.png".format(env_idx, dataset_idx))
            img3 = os.path.join(res_dir, "gt_col_label_{}_{}.png".format(env_idx, dataset_idx))
            utils.visualize_nodes_local(occ_grid, free_samples, start_pos, goal_pos, max_num=float('inf'), show=False, save=True, file_name=img1)
            visualize_nodes_local(occ_grid, pos_samples, neg_samples, neutral_samples, start_pos, goal_pos, show=False, save=True, file_name=img2)
            visualize_nodes_local(occ_grid, free_samples, col_samples, [], start_pos, goal_pos, show=False, save=True, file_name=img3)
