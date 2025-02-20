import os
import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), "../"))

import os
import os.path as osp
import numpy as np
from pathlib import Path
import networkx as nx
import random
import pickle
import torch.multiprocessing as mp

import utils
from env.maze import Maze

CUR_DIR = osp.dirname(osp.abspath(__file__))

def sample_problems(G):
    # path = dict(nx.all_pairs_shortest_path(G))
    free_nodes = [n for n in G.nodes() if not G.nodes[n]['col']]

    max_trial = 10000
    i = 0
    while i < max_trial:
        s_name = random.choice(free_nodes)
        start_pos = utils.node_to_numpy(G, s_name).tolist()

        g_name = random.choice(free_nodes)
        goal_pos = utils.node_to_numpy(G, g_name).tolist()

        try:
            node_path = nx.shortest_path(G, source=s_name, target = g_name)
        except:
            continue

        path = [utils.node_to_numpy(G, n).tolist() for n in node_path]
        # for x in p:
        #     x[0] += 2
        #     x[1] += 2

        if len(path) > 4 or utils.cal_path_len_base(path) > 5:
            return s_name, g_name, path

        i += 1

    return None, None, None


maze = Maze(gui=False)

env_num = 25
plan_num = 500

data_dir = os.path.join(CUR_DIR, "../dataset/gibson/train")
output_data_dir = os.path.join(CUR_DIR, "dataset/train_raw")
output_data_dir2 = os.path.join(CUR_DIR, "dataset/train")

maze_dirs = []
for path in Path(data_dir).rglob('env_small.obj'):
    maze_dirs.append(path.parent)

# def collect_path(env_idx):
#     data_cnt = 0
#     maze_dir = maze_dirs[env_idx]
#     print("generating test problem from {}".format(maze_dir))

#     occ_grid = np.loadtxt(osp.join(maze_dir, "occ_grid_small.txt")).astype(np.uint8)
#     G = nx.read_graphml(osp.join(maze_dir, "dense_g_small.graphml.xml"))

#     maze.clear_obstacles()
#     maze.load_mesh(osp.join(maze_dir, "env_small.obj"))
#     maze.load_occupancy_grid(occ_grid)

#     for j in range(plan_num):
#         print("{}: Generating test env {}".format(env_idx, j))

#         file_path = osp.join(output_data_dir, "data_{}_{}.pkl".format(env_idx, j))
#         if os.path.exists(file_path):
#             idx += 1
#             print("Skipping data_{}_{}".format(env_idx, data_cnt))
#             continue

#         s_node, g_node, expert_path = sample_problems(G)
#         # start_pos = utils.node_to_numpy(G, s_node).tolist()
#         # goal_pos = utils.node_to_numpy(G, g_node).tolist()

#         # tmp_dataset = []
#         # for idx in range(1, len(expert_path)):
#         #     pos = expert_path[idx - 1]
#         #     next_pos = expert_path[idx]
#         #     dist_to_g = utils.cal_path_len(expert_path[idx-1:])
#         #     tmp_dataset.append([occ_grid, start_pos, goal_pos, pos, next_pos, dist_to_g])

#         with open(file_path, 'wb') as f:
#             pickle.dump(expert_path, f)
#         data_cnt += 1

# j = 0
# processes = []
# for i in range(env_num):
#     p = mp.Process(target=collect_path, args=(i, ), daemon=True)
#     p.start()
#     processes.append(p)

# for p in processes:
#     p.join()


data_cnt = 0
for env_idx in range(env_num):
    maze_dir = maze_dirs[env_idx]
    print("generating test problem from {}".format(maze_dir))

    occ_grid = np.loadtxt(osp.join(maze_dir, "occ_grid_small.txt")).astype(np.uint8)

    for j in range(plan_num):
        file_path = osp.join(output_data_dir, "data_{}_{}.pkl".format(env_idx, j))
        with open(file_path, 'rb') as f:
            expert_path = pickle.load(f)

        start_pos = expert_path[0]
        goal_pos = expert_path[-1]

        tmp_dataset = []
        for idx in range(1, len(expert_path)):
            pos = expert_path[idx - 1]
            next_pos = expert_path[idx]
            dist_to_g = utils.cal_path_len(expert_path[idx-1:])
            tmp_dataset.append([occ_grid, start_pos, goal_pos, pos, next_pos, dist_to_g])

        for data in tmp_dataset:
            new_data_path = osp.join(output_data_dir2, "data_{}.pkl".format(data_cnt))
            with open(new_data_path, 'wb') as f:
                pickle.dump(data, f)
            data_cnt += 1

print(data_cnt)