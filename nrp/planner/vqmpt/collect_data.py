import os.path as osp
import sys
import os
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../../'))

import torch
import torch.multiprocessing as mp
import pickle
import math
import random
import numpy as np
import networkx as nx

# from env.snake_8d.maze import Snake8DEnv
# from env.snake_8d import utils
from env.fetch_11d import utils
from pathlib import Path

CUR_DIR = osp.dirname(osp.abspath(__file__))


def generate_trajectories(train_env_dir, num_samples, output_dir):

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # maze = Snake8DEnv(gui=False)

    # occ_grid_file = osp.join(train_env_dir, "occ_grid_small.txt")
    # occ_grid_target = osp.join(output_dir, "occ_grid_small.txt")
    occ_grid_file = osp.join(train_env_dir, "occ_grid_final.npy")
    occ_grid_target = osp.join(output_dir, "occ_grid_final.npy")
    os.system(f"cp {occ_grid_file} {occ_grid_target}")

    # G = nx.read_graphml(osp.join(train_env_dir, "dense_g_small.graphml.xml"))
    G = utils.get_prm(train_env_dir)
    # occ_grid = np.loadtxt(osp.join(train_env_dir, "occ_grid_small.txt")).astype(np.uint8)
    # mesh_path = osp.join(train_env_dir, "env_small.obj")

    idx = 0
    while idx < num_samples:
        # get random start and goal state from PRM
        start_node, goal_node = random.sample(list(G.nodes), 2)
        try:
            path_nodes = nx.shortest_path(G, start_node, goal_node, "weight")
        except nx.exception.NetworkXNoPath:
            print("no path, trying again")
            continue

        if len(path_nodes) > 0:
            path = [utils.node_to_numpy(G, n) for n in path_nodes]  # list of np.array
            path_interpolated = utils.interpolate(path)  # list of np.array

            print(f"Collected Path {idx}")
            traj_data = {'path': path, 'path_interpolated': path_interpolated, 'success': True}
            pickle.dump(traj_data, open(osp.join(output_dir, f'path_{idx}.p'), 'wb'))
            idx += 1


if __name__ == '__main__':
    # constants
    robot_dim = 11
    model_name = "11d"
    train_data_dir = osp.join(CUR_DIR, "./dataset", model_name, "train")
    if not os.path.exists(train_data_dir):
        os.mkdir(train_data_dir)
    test_data_dir = osp.join(CUR_DIR, "./dataset", model_name, "val")
    if not os.path.exists(test_data_dir):
        os.mkdir(test_data_dir)

    # stage 1 empty env
    train_empty_env_dir = osp.join(train_data_dir, "env_-1")
    test_empty_env_dir = osp.join(test_data_dir, "env_-1")

    # env dirs
    env_num = 25
    # train_env_dir = osp.join(CUR_DIR, "../../env/snake_8d/dataset/gibson/train")
    train_env_dir = osp.join(CUR_DIR, "../../env/fetch_11d/dataset/gibson/train")
    train_env_dirs = []
    # for p in Path(train_env_dir).rglob("env_small.obj"):
    for p in Path(train_env_dir).rglob("env_final.obj"):
        train_env_dirs.append(p.parent)
    assert len(train_env_dirs) == env_num

    test_env_num = 5
    # test_env_dir = osp.join(CUR_DIR, "../../env/snake_8d/dataset/gibson/mytest")
    test_env_dir = osp.join(CUR_DIR, "../../env/fetch_11d/dataset/gibson/mytest")
    test_env_dirs = []
    # for p in Path(test_env_dir).rglob('env_small.obj'):
    for p in Path(test_env_dir).rglob('env.obj'):
        test_env_dirs.append(p.parent)
    assert len(test_env_dirs) == test_env_num

    # hyperparameters
    train_empty_env_data_cnt = 2000
    train_data_cnt_per_env = 20000 // env_num

    test_empty_env_data_cnt = 20
    test_data_cnt_per_env = 20

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    print("----------- Stage 1: Collecting data from empty env -------------")

    generate_trajectories(train_empty_env_dir, train_empty_env_data_cnt, train_empty_env_dir)
    generate_trajectories(test_empty_env_dir, test_empty_env_data_cnt, test_empty_env_dir)

    print("----------- Stage 2: Collecting data from train env -------------")
    process_num = 25

    j = 0
    while j < len(train_env_dirs):
        processes = []
        print("Running on env {} to {}".format(j, min(len(train_env_dirs), j + process_num)))
        for i in range(j, min(len(train_env_dirs), j + process_num)):
            output_dir = osp.join(train_data_dir, f"env_{i}")
            p = mp.Process(target=generate_trajectories, args=(train_env_dirs[i], train_data_cnt_per_env, output_dir), daemon=True)
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        j += process_num

    print("----------- Stage 2: Collecting data from test env -------------")
    process_num = 5

    j = 0
    while j < len(test_env_dirs):
        processes = []
        print("Running on env {} to {}".format(j, min(len(test_env_dirs), j + process_num)))
        for i in range(j, min(len(test_env_dirs), j + process_num)):
            output_dir = osp.join(test_data_dir, f"env_{i}")
            p = mp.Process(target=generate_trajectories, args=(test_env_dirs[i], test_data_cnt_per_env, output_dir), daemon=True)
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        j += process_num