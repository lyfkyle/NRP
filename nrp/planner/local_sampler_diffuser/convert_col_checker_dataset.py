import os.path as osp
import sys
import os
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))
import copy

import torch
import pickle
import math
import random
from utils import is_robot_within_local_env, interpolate_to_fixed_horizon, get_intersection

CUR_DIR = osp.dirname(osp.abspath(__file__))
LOCAL_ENV_SIZE = 2.0

def get_datas(data_dir, env_num, dataset_num):
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
                occ_grid, start_pos, goal_pos, label = data
                
                samples.append([occ_grid, start_pos, goal_pos, label])

            all_samples += samples
            print(file_path)

    return all_samples

def get_datas_eval(data_dir, env_num, dataset_num, num_to_sample):
    all_samples = []
    for env_idx in range(env_num):
        for _ in range(num_to_sample):
            dataset_idx = random.randint(0, dataset_num)
            samples = []

            file_path = osp.join(data_dir, "data_{}_{}.pkl".format(env_idx, dataset_idx))

            if not os.path.exists(file_path):
                continue

            # print(f"Opening {file_path}")

            with open(file_path, 'rb') as f:
                datas = pickle.load(f)

            for i, data in enumerate(datas):
                occ_grid, start_pos, goal_pos, label = data
                
                samples.append([occ_grid, start_pos, goal_pos, label])

            # visualize_nodes_local(occ_grid, pos_samples, neg_samples, neutral_samples, start_pos, goal_pos, show=False, save=True, file_name=os.path.join(CUR_DIR, "res/topk_viz/gt_label_{}_{}.png".format(env_idx, dataset_idx)))

            all_samples += samples
            print(file_path)

    return all_samples

def get_datas_test(data_dir, env_num, dataset_num):
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
                occ_grid, start_pos, goal_pos, label = data

                samples.append([occ_grid, start_pos, goal_pos, label])

            # visualize_nodes_local(occ_grid, pos_samples, neg_samples, neutral_samples, start_pos, goal_pos, show=False, save=True, file_name=os.path.join(CUR_DIR, "res/topk_viz/gt_label_{}_{}.png".format(env_idx, dataset_idx)))

            all_samples += samples

    return all_samples

if __name__ == '__main__':
    # constants
    model_name = "col_checker"
    data_dir = osp.join(CUR_DIR, "./dataset/{}".format(model_name))
    data_dir_t = osp.join(CUR_DIR, "./dataset/{}_t".format(model_name))

    sel_train_data_dir = osp.join(CUR_DIR, "./dataset/{}_train".format(model_name))
    if not os.path.exists(sel_train_data_dir):
        os.makedirs(sel_train_data_dir)
    eval_data_dir = osp.join(CUR_DIR, "./dataset/{}_eval".format(model_name))
    if not os.path.exists(eval_data_dir):
        os.makedirs(eval_data_dir)
    test_data_dir = osp.join(CUR_DIR, "./dataset/{}_test".format(model_name))
    if not os.path.exists(test_data_dir):
        os.makedirs(test_data_dir)

    # hyperparameters
    # data_cnt = 1258258
    train_data_cnt = 0
    train_data_pos_cnt = 0
    eval_data_cnt = 0
    test_data_cnt = 0


    # selection
    all_samples = get_datas(data_dir, 25, 2000)
    print(len(all_samples))
    for sample in all_samples:
        new_file_path = osp.join(sel_train_data_dir, "data_{}.pkl".format(train_data_cnt))
        with open(new_file_path, 'wb') as f:
            print("Dumping to {}".format(new_file_path))
            pickle.dump(sample, f)
        train_data_cnt += 1
        train_data_pos_cnt += sample[-1]

    # Evaluation dataset. No need to do balance
    all_samples = get_datas_eval(data_dir, 25, 2000, 50)
    print(len(all_samples))
    for sample in all_samples:
        new_file_path = osp.join(eval_data_dir, "data_{}.pkl".format(eval_data_cnt))
        with open(new_file_path, 'wb') as f:
            print("Dumping to {}".format(new_file_path))
            pickle.dump(sample, f)
        eval_data_cnt += 1

    # Test dataset. No need to do balance
    all_samples = get_datas_test(data_dir_t, 5, 50)
    print(len(all_samples))
    for sample in all_samples:
        new_file_path = osp.join(test_data_dir, "data_{}.pkl".format(test_data_cnt))
        with open(new_file_path, 'wb') as f:
            print("Dumping to {}".format(new_file_path))
            pickle.dump(sample, f)
        test_data_cnt += 1

    print("train:", train_data_cnt, "eval:", eval_data_cnt, "test:", test_data_cnt)
    print("positive_in_train:", train_data_pos_cnt)
