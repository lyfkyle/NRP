
import os.path as osp
import sys
import os
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../../'))

import torch.multiprocessing as mp
from pathlib import Path
import pickle
import math
import numpy as np
import random
import networkx as nx
import torch

from env.fetch_11d.maze import Fetch11DEnv

CUR_DIR = osp.dirname(osp.abspath(__file__))

NUM_OF_COL_SAMPLE = 100
LOCAL_ENV_SIZE = 2.0
PATH_LEN_DIFF_THRESHOLD = 2.0

PROCESS_NUM = 40

def sample_problems(env, G):
    free_nodes = [n for n in G.nodes() if not G.nodes[n]['col']]

    i = 0
    s_name = None
    g_name = None
    p = None
    while True:
        s_name = random.choice(free_nodes)
        start_pos = env.utils.node_to_numpy(G, s_name).tolist()

        g_name = random.choice(free_nodes)
        goal_pos = env.utils.node_to_numpy(G, g_name).tolist()

        try:
            path = nx.shortest_path(G, source=s_name, target=g_name)
        except:
            continue

        p = [env.utils.node_to_numpy(G, n).tolist() for n in path]
        # for x in p:
        #     x[0] += 2
        #     x[1] += 2

        # if len(p) > 2 and math.fabs(goal_pos[0] - start_pos[0]) > 2 and math.fabs(goal_pos[1] - start_pos[1]) > 2:
        #     break

        if len(p) > 2 and math.sqrt((goal_pos[0] - start_pos[0])**2 + (goal_pos[1] - start_pos[1])**2) > 5:
            break

        i += 1

    return p, start_pos, goal_pos


def collect_gt(train_env_dirs, env_idx, num_samples_per_env=100, data_dir=""):
    maze_dir = train_env_dirs[env_idx]
    env = Fetch11DEnv(gui=False)

    # occ_grid = env.utils.get_occ_grid(maze_dir)
    orig_G = env.utils.get_prm(maze_dir)
    # mesh_path = env.utils.get_mesh_path(maze_dir)

    # env.clear_obstacles()
    # env.load_mesh(mesh_path)
    # env.load_occupancy_grid(occ_grid, add_enclosing=True)

    idx = 0
    while idx < num_samples_per_env:
        dataset_list = []
        file_path = osp.join(data_dir, "data_{}_{}.pkl".format(env_idx, idx))
        if os.path.exists(file_path):
            idx += 1
            print("Skipping data_{}_{}".format(env_idx, idx))
            continue

        # Get expert path
        expert_path, _, _ = sample_problems(env, orig_G)
        if expert_path is None:
            print("not path exists between sampled start and goal position")
            continue

        # utils.visualize_nodes_global(global_occ_grid, new_expert_path, None, None, show=False, save=True, file_name=osp.join(CUR_DIR, "res/expert_path_without_global_information_{}_{}.png".format(env_idx, idx)))
        dataset_list.append([maze_dir, expert_path])
        print("{}: Adding gt {}/{}".format(env_idx, idx, num_samples_per_env))

        with open(file_path, 'wb') as f:
            # print("Dumping to {}".format(file_path))
            pickle.dump(dataset_list, f)

        idx += 1


if __name__ == '__main__':
    # constants
    model_name = "model_fire"
    data_dir = osp.join(CUR_DIR, "./dataset/{}".format(model_name))
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    # test_data_dir = osp.join(CUR_DIR, "./dataset/{}_t".format(model_name))
    # if not os.path.exists(test_data_dir):
    #     os.mkdir(test_data_dir)

    # env dirs
    env_num = 25
    train_env_dir = osp.join(CUR_DIR, "../../dataset/fetch_11d/gibson/train")
    train_env_dirs = []
    for p in Path(train_env_dir).rglob('env_final.obj'):
        train_env_dirs.append(p.parent)
    assert len(train_env_dirs) == env_num
    # test_env_dir = osp.join(CUR_DIR, "../dataset/gibson/mytest")
    # test_env_dirs = []
    # for p in Path(test_env_dir).rglob('env_small.obj'):
    #     test_env_dirs.append(p.parent)
    # assert len(test_env_dirs) == 5

    # hyperparameters
    target_data_cnt = 100000

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    iter_num = 0
    data_cnt = 0
    # while data_cnt < target_data_cnt:

    # Run in train env
    print("----------- Collecting from train env -------------")
    print("Collecting gt")
    process_num = 25
    manager = mp.Manager()
    # dataset_dict = manager.dict()

    env_obj_dict = manager.dict()

    # collect_gt(train_env_dirs, 0, env_obj_dict, 1)

    # test
    j = 0
    while j < len(train_env_dirs):
        processes = []
        print("Running on env {} to {}".format(j, min(len(train_env_dirs), j + process_num)))
        for i in range(j, min(len(train_env_dirs), j + process_num)):
            p = mp.Process(target=collect_gt, args=(train_env_dirs, i, 100, data_dir), daemon=True)
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        j += process_num

    print("----------- Collecting from test env -------------")
    # print("Collecting gt")
    # process_num = 5
    # manager = mp.Manager()
    # # dataset_dict = manager.dict()

    # env_obj_dict = manager.dict()
    # for env_idx in range(len(test_env_dirs)):
    #     env_obj_dict[env_idx] = 0

    # # test
    # j = 0
    # while j < len(test_env_dirs):
    #     processes = []
    #     print("Running on env {} to {}".format(j, min(len(test_env_dirs), j + process_num)))
    #     for i in range(j, min(len(test_env_dirs), j + process_num)):
    #         p = mp.Process(target=collect_gt, args=(test_env_dirs, i, env_obj_dict, 50, test_data_dir), daemon=True)
    #         p.start()
    #         processes.append(p)

    #     for p in processes:
    #         p.join()

    #     j += process_num
