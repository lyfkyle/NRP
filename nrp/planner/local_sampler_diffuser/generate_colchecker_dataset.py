import os.path as osp
import sys
import os
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))

import torch
import networkx as nx
import random
import numpy as np
import math
import pickle
from pathlib import Path
import torch.multiprocessing as mp
import itertools

import utils
from env.maze import Maze

CUR_DIR = osp.dirname(osp.abspath(__file__))

NUM_OF_RANDOM_LOCAL_SAMPLE = 100
LOCAL_ENV_SIZE = 2.0
FORCE_VALID_STATE_RATE = 0.7

# FORCE_CLOSE_RANGE_RATE = 0.8
# CLOSE_BASE_RANGE = 0.5
# CLOSE_JOINT_RANGE = 1.0

PROCESS_NUM = 40


def collect_gt(train_env_dirs, save_data_dir, env_idx, env_obj_dict, num_samples_per_env=100):
    maze_dir = train_env_dirs[env_idx]
    maze = Maze(gui=False)

    # print("Evaluation on {}".format(maze_dir))

    occ_grid = np.loadtxt(osp.join(maze_dir, "occ_grid_small.txt")).astype(np.uint8)
    # orig_G = nx.read_graphml(osp.join(maze_dir, "dense_g_small.graphml"))
    maze.clear_obstacles()
    # maze.load_occupancy_grid(occ_grid)

    idx = 0
    while idx < num_samples_per_env:
        dataset_list = []
        file_path = osp.join(save_data_dir, "data_{}_{}.pkl".format(env_idx, idx))
        if os.path.exists(file_path):
            idx += 1
            print("Skipping data_{}_{}".format(env_idx, idx))
            continue

        maze.clear_obstacles()
        maze.load_mesh(osp.join(maze_dir, "env_small.obj"))
        maze.load_occupancy_grid(occ_grid)

        high = maze.robot.get_joint_higher_bounds()
        low = maze.robot.get_joint_lower_bounds()

        # sample v_pos
        random_state = [0] * maze.robot.num_dim
        for i in range(maze.robot.num_dim):
            random_state[i] = random.uniform(low[i], high[i])
        start_pos = np.array(random_state)
        # start_pos = utils.node_to_numpy(orig_G, start)
        # goal_pos = utils.node_to_numpy(orig_G, goal)
        local_start_pos = utils.global_to_local(start_pos, start_pos)

        # assert math.fabs(local_goal_pos[0]) > LOCAL_ENV_SIZE or math.fabs(local_goal_pos[1]) > LOCAL_ENV_SIZE

        local_occ_grid = maze.get_local_occ_grid(start_pos)

        obj_idx = env_obj_dict[env_idx]
        tmp_mesh_file_name = "tmp_{}_{}.obj".format(env_idx, obj_idx)
        env_obj_dict[env_idx] += 1
        global_occ_grid, new_mesh_path = maze.clear_obstacles_outside_local_occ_grid(start_pos, tmp_mesh_file_name)

        # utils.visualize_nodes_global(osp.join(maze_dir, "env_large.obj"), occ_grid, [], None, None, show=False, save=True, file_name=osp.join(CUR_DIR, "res/global_{}_{}.png".format(env_idx, idx)))
        # utils.visualize_nodes_global(new_mesh_path, global_occ_grid, [], None, None, show=False, save=True, file_name=osp.join(CUR_DIR, "res/local_{}_{}.png".format(env_idx, idx)))

        # G = copy.deepcopy(orig_G)

        # connect edges outside local environment:
        # # free_nodes = [node for node in G.nodes() if not G.nodes[node]['col']]
        # for node in G.nodes():
        #     node_pos = utils.node_to_numpy(G, node)
        #     # for node outsiode local environment, directly connect to goal
        #     if math.fabs(node_pos[0] - start_pos[0]) > LOCAL_ENV_SIZE or math.fabs(node_pos[1] - start_pos[1]) > LOCAL_ENV_SIZE:
        #         nx.set_node_attributes(G, {node: False}, "col")
        #         if not G.has_edge(node, goal):
        #             G.add_edge(node, goal, weight=utils.calc_edge_len(node_pos, goal_pos))

        # Get random nodes within local environment except expert node + nodes availabel in PRM
        high = maze.robot.get_joint_higher_bounds()
        high[0] = LOCAL_ENV_SIZE
        high[1] = LOCAL_ENV_SIZE
        low = maze.robot.get_joint_lower_bounds()
        low[0] = -LOCAL_ENV_SIZE
        low[1] = -LOCAL_ENV_SIZE

        local_sample_pos_v1 = []
        local_sample_pos_v2 = []
        while len(local_sample_pos_v1) < NUM_OF_RANDOM_LOCAL_SAMPLE:
            random_state = [0] * maze.robot.num_dim
            for i in range(maze.robot.num_dim):
                random_state[i] = random.uniform(low[i], high[i])
            if random.random() > FORCE_VALID_STATE_RATE or maze.pb_ompl_interface.is_state_valid(utils.local_to_global(random_state, start_pos)):
                local_sample_pos_v1.append(random_state)
            else:
                continue

        while len(local_sample_pos_v2) < NUM_OF_RANDOM_LOCAL_SAMPLE:
            random_state = [0] * maze.robot.num_dim
            for i in range(maze.robot.num_dim):
                random_state[i] = random.uniform(low[i], high[i])
            if random.random() > FORCE_VALID_STATE_RATE or maze.pb_ompl_interface.is_state_valid(utils.local_to_global(random_state, start_pos)):
                local_sample_pos_v2.append(random_state)
            else:
                continue


        # dataset_list.append([env_idx, local_occ_grid, start, goal, sample_nodes, expert_next_node])
        # dataset_list.append([local_occ_grid, local_start_pos, local_goal_pos, local_sample_pos, local_expert_pos, col_status, expert_node_path, start_pos])
        print("{}: Adding gt {}/{}".format(env_idx, idx, num_samples_per_env))

        # print(local_start_pos, local_goal_pos, local_expert_pos)
        # utils.visualize_nodes_local(local_occ_grid, [local_expert_pos], local_start_pos, local_goal_pos, show=False, save=True, file_name=osp.join(CUR_DIR, "res/gt_{}.png".format(i)))

        # 0 = collision, 1 = no collision
        num_pos = 0  # no collision
        for i in range(NUM_OF_RANDOM_LOCAL_SAMPLE):
            v1_pos = utils.local_to_global(local_sample_pos_v1[i], start_pos)
            v2_pos = utils.local_to_global(local_sample_pos_v2[i], start_pos)
            edge_col_free = utils.is_edge_free(maze, v1_pos, v2_pos)

            dataset_list.append([local_occ_grid, local_sample_pos_v1[i], local_sample_pos_v2[i], int(edge_col_free)])
            num_pos += int(edge_col_free)

        print("num_pos {}, num_neg: {}".format(num_pos, NUM_OF_RANDOM_LOCAL_SAMPLE - num_pos))
        print("{}: Adding to output {}/{}".format(env_idx, idx, num_samples_per_env))

        # utils.visualize_nodes_local(local_occ_grid, pos_samples, local_start_pos, local_goal_pos, show=False, save=True, file_name=osp.join(CUR_DIR, "res/gt_pos_{}_{}.png".format(env_idx, idx)))
        # utils.visualize_nodes_local(local_occ_grid, neg_samples, local_start_pos, local_goal_pos, show=False, save=True, file_name=osp.join(CUR_DIR, "res/gt_neg_{}_{}.png".format(env_idx, idx)))
        # utils.visualize_nodes_local(local_occ_grid, col_samples, local_start_pos, local_goal_pos, show=False, save=True, file_name=osp.join(CUR_DIR, "res/gt_neg_col_{}_{}.png".format(env_idx, idx)))

        # global_pos_samples = [utils.local_to_global(s, start_pos) for s in pos_samples]
        # global_neg_samples = [utils.local_to_global(s, start_pos) for s in neg_samples]
        # global_col_samples = [utils.local_to_global(s, start_pos) for s in col_samples]
        # global_unlabeled_samples = [utils.local_to_global(s, start_pos) for s in unlabeled_samples]
        # utils.visualize_nodes_global(new_mesh_path, global_occ_grid, global_pos_samples, start_pos, goal_pos, show=False, save=True, file_name=osp.join(CUR_DIR, "res/gt_pos_{}_{}.png".format(env_idx, idx)))
        # utils.visualize_nodes_global(new_mesh_path, global_occ_grid, global_neg_samples, start_pos, goal_pos, show=False, save=True, file_name=osp.join(CUR_DIR, "res/gt_neg_{}_{}.png".format(env_idx, idx)))
        # utils.visualize_nodes_global(new_mesh_path, global_occ_grid, global_col_samples, start_pos, goal_pos, show=False, save=True, file_name=osp.join(CUR_DIR, "res/gt_neg_col_{}_{}.png".format(env_idx, idx)))
        # utils.visualize_nodes_global(new_mesh_path, global_occ_grid, global_unlabeled_samples, start_pos, goal_pos, show=False, save=True, file_name=osp.join(CUR_DIR, "res/gt_unlabeled_{}_{}.png".format(env_idx, idx)))

        if os.path.exists(new_mesh_path):
            os.remove(new_mesh_path)
        assert not os.path.exists(new_mesh_path)

        file_path = osp.join(save_data_dir, "data_{}_{}.pkl".format(env_idx, idx))
        with open(file_path, 'wb') as f:
            # print("Dumping to {}".format(file_path))
            pickle.dump(dataset_list, f)

        idx += 1

# def save_dataset(dataset, data_cnt, start_idx, end_idx):
#     print("Saving dataset {} to {}".format(start_idx, end_idx))
#     for idx in range(start_idx, end_idx):
#         file_path = osp.join(data_dir, "data_{}.pkl".format(data_cnt + idx))
#         with open(file_path, 'wb') as f:
#             # print("Dumping to {}".format(file_path))
#             pickle.dump(dataset[idx], f)

if __name__ == '__main__':
    # constants
    robot_dim = 8
    linkpos_dim = 12
    occ_grid_dim = 40
    model_name = "col_checker"
    data_dir = osp.join(CUR_DIR, "./dataset/{}".format(model_name))
    data_dir_t = osp.join(CUR_DIR, "./dataset/{}_t".format(model_name))
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    if not os.path.exists(data_dir_t):
        os.mkdir(data_dir_t)

    # env dirs
    env_num = 25
    train_env_dir = osp.join(CUR_DIR, "../dataset/gibson/train")
    train_env_dirs = []
    for p in Path(train_env_dir).rglob('env_small.obj'):
        train_env_dirs.append(p.parent)
    assert len(train_env_dirs) == env_num
    test_env_dir = osp.join(CUR_DIR, "../dataset/gibson/mytest")
    test_env_dirs = []
    for p in Path(test_env_dir).rglob('env_small.obj'):
        test_env_dirs.append(p.parent)
    assert len(test_env_dirs) == 5

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
    for env_idx in range(len(train_env_dirs)):
        env_obj_dict[env_idx] = 0

    # collect_gt(train_env_dirs, 0, env_obj_dict, 1)

    # test
    j = 0
    while j < len(train_env_dirs):
        processes = []
        print("Running on env {} to {}".format(j, min(len(train_env_dirs), j + process_num)))
        for i in range(j, min(len(train_env_dirs), j + process_num)):
            p = mp.Process(target=collect_gt, args=(train_env_dirs, data_dir, i, env_obj_dict, 2000), daemon=True)
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        j += process_num

    print("----------- Collecting from test env -------------")
    print("Collecting gt")
    process_num = 5
    manager = mp.Manager()
    # dataset_dict = manager.dict()

    env_obj_dict = manager.dict()
    for env_idx in range(len(test_env_dirs)):
        env_obj_dict[env_idx] = 0

    # test
    j = 0
    while j < len(test_env_dirs):
        processes = []
        print("Running on env {} to {}".format(j, min(len(test_env_dirs), j + process_num)))
        for i in range(j, min(len(test_env_dirs), j + process_num)):
            p = mp.Process(target=collect_gt, args=(test_env_dirs, data_dir_t, i, env_obj_dict, 50), daemon=True)
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        j += process_num