import os.path as osp
import sys
import os
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))

import torch
import pickle
import math
import random

CUR_DIR = osp.dirname(osp.abspath(__file__))

def get_datas(data_dir, env_num, dataset_num):
    all_pos_samples = []
    all_neg_samples = []
    col_samples = []
    for env_idx in range(env_num):
        for dataset_idx in range(dataset_num):
            pos_samples = []
            neg_samples = []
            neutral_samples = []

            file_path = osp.join(data_dir, "data_{}_{}.pkl".format(env_idx, dataset_idx))

            if not os.path.exists(file_path):
                continue

            # print(f"Opening {file_path}")

            with open(file_path, 'rb') as f:
                datas = pickle.load(f)

            free_samples = []
            for i, data in enumerate(datas):
                occ_grid, start_pos, goal_pos, sample_pos, expert_pos, selected_path_len, expert_path_len = data

                if selected_path_len == -1:
                    col_samples.append([occ_grid, start_pos, goal_pos, sample_pos, 0, 0])
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
                pos_samples.append([occ_grid, start_pos, goal_pos, sample_pos, 1, 1])
            for sample_pos, _ in neg_s:
                neg_samples.append([occ_grid, start_pos, goal_pos, sample_pos, 1, 0])
            for sample_pos, _ in neural_s:
                neutral_samples.append([occ_grid, start_pos, goal_pos, sample_pos, 1, -1])

            # visualize_nodes_local(occ_grid, pos_samples, neg_samples, neutral_samples, start_pos, goal_pos, show=False, save=True, file_name=os.path.join(CUR_DIR, "res/topk_viz/gt_label_{}_{}.png".format(env_idx, dataset_idx)))

            all_pos_samples += pos_samples
            all_neg_samples += neg_samples

    return all_pos_samples, all_neg_samples, col_samples

def get_datas_eval(data_dir, env_num, dataset_num, num_to_sample):
    all_pos_samples = []
    all_neg_samples = []
    col_samples = []
    for env_idx in range(env_num):
        for _ in range(num_to_sample):
            dataset_idx = random.randint(0, dataset_num)

            pos_samples = []
            neg_samples = []
            neutral_samples = []

            file_path = osp.join(data_dir, "data_{}_{}.pkl".format(env_idx, dataset_idx))

            if not os.path.exists(file_path):
                continue

            # print(f"Opening {file_path}")

            with open(file_path, 'rb') as f:
                datas = pickle.load(f)

            free_samples = []
            for i, data in enumerate(datas):
                occ_grid, start_pos, goal_pos, sample_pos, expert_pos, selected_path_len, expert_path_len = data

                if selected_path_len == -1:
                    col_samples.append([occ_grid, start_pos, goal_pos, sample_pos, 0, 0])
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
                pos_samples.append([occ_grid, start_pos, goal_pos, sample_pos, 1, 1])
            for sample_pos, _ in neg_s:
                neg_samples.append([occ_grid, start_pos, goal_pos, sample_pos, 1, 0])
            for sample_pos, _ in neural_s:
                neutral_samples.append([occ_grid, start_pos, goal_pos, sample_pos, 1, -1])

            # visualize_nodes_local(occ_grid, pos_samples, neg_samples, neutral_samples, start_pos, goal_pos, show=False, save=True, file_name=os.path.join(CUR_DIR, "res/topk_viz/gt_label_{}_{}.png".format(env_idx, dataset_idx)))

            all_pos_samples += pos_samples
            all_neg_samples += neg_samples

    return all_pos_samples, all_neg_samples, col_samples

def get_datas_test(data_dir, env_num, dataset_num):
    all_pos_samples = []
    all_neg_samples = []
    col_samples = []
    for env_idx in range(env_num):
        for dataset_idx in range(dataset_num):
            pos_samples = []
            neg_samples = []
            neutral_samples = []

            file_path = osp.join(data_dir, "data_{}_{}.pkl".format(env_idx, dataset_idx))

            if not os.path.exists(file_path):
                continue

            # print(f"Opening {file_path}")

            with open(file_path, 'rb') as f:
                datas = pickle.load(f)

            free_samples = []
            for i, data in enumerate(datas):
                occ_grid, start_pos, goal_pos, sample_pos, expert_pos, selected_path_len, expert_path_len  = data

                if selected_path_len == -1:
                    col_samples.append([occ_grid, start_pos, goal_pos, sample_pos, 0, 0])
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
                pos_samples.append([occ_grid, start_pos, goal_pos, sample_pos, 1, 1])
            for sample_pos, _ in neg_s:
                neg_samples.append([occ_grid, start_pos, goal_pos, sample_pos, 1, 0])
            for sample_pos, _ in neural_s:
                neutral_samples.append([occ_grid, start_pos, goal_pos, sample_pos, 1, -1])

            # visualize_nodes_local(occ_grid, pos_samples, neg_samples, neutral_samples, start_pos, goal_pos, show=False, save=True, file_name=os.path.join(CUR_DIR, "res/topk_viz/gt_label_{}_{}.png".format(env_idx, dataset_idx)))

            all_pos_samples += pos_samples
            all_neg_samples += neg_samples

    return all_pos_samples, all_neg_samples, col_samples

if __name__ == '__main__':
    # constants
    model_name = "model"
    data_dir = osp.join(CUR_DIR, "../local_sampler_d/dataset/{}_global".format(model_name))
    data_dir_t = osp.join(CUR_DIR, "../local_sampler_d/dataset/{}_t".format(model_name))

    sel_train_data_dir = osp.join(CUR_DIR, "./dataset/{}_train_global".format(model_name))
    if not os.path.exists(sel_train_data_dir):
        os.makedirs(sel_train_data_dir)
    eval_data_dir = osp.join(CUR_DIR, "./dataset/{}_eval_global".format(model_name))
    if not os.path.exists(eval_data_dir):
        os.makedirs(eval_data_dir)
    test_data_dir = osp.join(CUR_DIR, "./dataset/{}_test_global".format(model_name))
    if not os.path.exists(test_data_dir):
        os.makedirs(test_data_dir)

    # hyperparameters
    # data_cnt = 1258258
    col_train_data_cnt = 0
    sel_train_data_cnt = 0
    integrated_train_data_cnt = 0
    eval_data_cnt = 0
    test_data_cnt = 0

    pos_samples, neg_samples, col_samples = get_datas(data_dir, 25, 500)
    print(len(pos_samples), len(neg_samples), len(col_samples))

    random.shuffle(pos_samples)
    train_samples = pos_samples[:100000]

    # selection
    sel_pos_samples = train_samples
    for sample in train_samples:
        new_file_path = osp.join(sel_train_data_dir, "data_{}.pkl".format(sel_train_data_cnt))
        with open(new_file_path, 'wb') as f:
            # print("Dumping to {}".format(file_path))
            pickle.dump(sample, f)
        sel_train_data_cnt += 1

    # Evaluation dataset. No need to do balance
    # pos_samples, neg_samples, col_samples = get_datas_eval(data_dir, 25, 2000, 50)
    # print(len(pos_samples), len(neg_samples), len(col_samples))
    eval_samples = pos_samples[100000:110000]
    for sample in eval_samples:
        new_file_path = osp.join(eval_data_dir, "data_{}.pkl".format(eval_data_cnt))
        with open(new_file_path, 'wb') as f:
            # print("Dumping to {}".format(file_path))
            pickle.dump(sample, f)
        eval_data_cnt += 1

    # Test dataset. No need to do balance
    # pos_samples, neg_samples, col_samples = get_datas_test(data_dir_t, 5, 50)
    # for sample in pos_samples:
    #     new_file_path = osp.join(test_data_dir, "data_{}.pkl".format(test_data_cnt))
    #     with open(new_file_path, 'wb') as f:
    #         # print("Dumping to {}".format(file_path))
    #         pickle.dump(sample, f)
    #     test_data_cnt += 1

    # print(sel_train_data_cnt, eval_data_cnt, test_data_cnt)
