import os.path as osp
import sys
import os
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))

import torch
import pickle
import math
import random
import copy

import pandas as pd
from utils import visualize_nodes_local, is_robot_within_local_env, interpolate_to_fixed_horizon, get_intersection


CUR_DIR = osp.dirname(osp.abspath(__file__))
LOCAL_ENV_SIZE = 2.0
HORIZON = 5

def get_datas(data_dir, env_num, dataset_num):
    path_lengths = []
    for env_idx in range(env_num):
        for dataset_idx in range(dataset_num):

            file_path = osp.join(data_dir, "data_{}_{}.pkl".format(env_idx, dataset_idx))

            if not os.path.exists(file_path):
                continue

            # print(f"Opening {file_path}")

            with open(file_path, 'rb') as f:
                datas = pickle.load(f)

            for i, data in enumerate(datas):
                occ_grid, start_pos, goal_pos, expert_path, path_len, expert_path_len, collision = data
                print(collision)
                path_lengths.append(path_len)

                visualize_nodes_local(occ_grid, expert_path, start_pos, goal_pos, show=False, save=True, file_name=os.path.join(CUR_DIR, "dataset/viz/straight_line_{}_{}_{}.png".format(env_idx, dataset_idx, i)))

    df = pd.DataFrame(path_lengths, columns =['lengths'])
    print(df.describe())
    # print('max: ', max(expert_path_lengths))
    # print('min: ', max(expert_path_lengths))

    # return all_pos_samples, all_neg_samples, col_samples

def interpolate_datas(data_dir, env_num, dataset_num):
    all_samples = []
    for env_idx in range(env_num):
        for dataset_idx in range(dataset_num):
            samples = []
            file_path = osp.join(data_dir, "data_{}_{}.pkl".format(env_idx, dataset_idx))

            if not os.path.exists(file_path):
                continue

            # print(f"Opening {file_path}")

            with open(file_path, 'rb') as f:
                datas = pickle.load(f)

            for i, data in enumerate(datas):
                occ_grid, start_pos, goal_pos, expert_path, path_len, expert_path_len = data
                # print(expert_path)
                # visualize_nodes_local(occ_grid, expert_path, start_pos, goal_pos, show=False, save=True, file_name=os.path.join(CUR_DIR, "dataset/viz/gt_label_{}_{}.png".format(env_idx, dataset_idx)))
                
                found_first_pos_out = False
                new_path_len = len(expert_path)
                new_expert_path = []
                for i, pos in enumerate(expert_path):
                    if is_robot_within_local_env(pos, LOCAL_ENV_SIZE):
                        continue
                    else:
                        first_pos_out = pos
                        last_pos_in = expert_path[i-1]
                        found_first_pos_out = True
                        new_path_len = i
                        break

                if found_first_pos_out:
                    q_n = get_intersection(last_pos_in, first_pos_out, LOCAL_ENV_SIZE)
                    new_expert_path = copy.deepcopy(expert_path[:new_path_len])
                    new_expert_path.append(q_n)
                else:
                    q_n = expert_path[-1]
                    new_expert_path = copy.deepcopy(expert_path)
                # visualize_nodes_local(occ_grid, new_expert_path, start_pos, goal_pos, show=False, save=True, file_name=os.path.join(CUR_DIR, "dataset/viz/gt_label_{}_{}_cropped.png".format(env_idx, dataset_idx)))
                interpolated_path = interpolate_to_fixed_horizon(new_expert_path, HORIZON)
                visualize_nodes_local(occ_grid, interpolated_path, start_pos, goal_pos, show=False, save=True, file_name=os.path.join(CUR_DIR, "dataset/viz/gt_label_{}_{}_interpolated_{}.png".format(env_idx, dataset_idx, HORIZON)))
                samples.append([occ_grid, start_pos, goal_pos, interpolated_path, path_len, expert_path_len])

                break
            
            all_samples += samples

    return all_samples


if __name__ == '__main__':
    # constants
    model_name = "diffusion"
    data_dir = osp.join(CUR_DIR, "dataset/{}".format(model_name))
    data_dir_t = osp.join(CUR_DIR, "dataset/{}_t".format(model_name))

    # pos_samples, neg_samples, col_samples = get_datas(data_dir, 25, 2000)
    # print(len(pos_samples), len(neg_samples), len(col_samples))
    # print("train data:")
    # get_datas(data_dir, 25, 1)

    # print("test data:")
    get_datas(data_dir, 1, 10)
    # interpolated_samples = interpolate_datas(data_dir, 25, 1)
