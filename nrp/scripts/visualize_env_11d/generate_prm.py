import os
import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))
# print(sys.path)

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import shutil
import json
from PIL import Image
import itertools
import random
from pathlib import Path
import time
import math
from multiprocessing import Process

from env.maze import Maze
import utils

CUR_DIR = osp.dirname(osp.abspath(__file__))

def state_to_numpy(state):
    strlist = state.split(',')
    val_list = [float(s) for s in strlist]
    return np.array(val_list)

# maze = Maze2D(gui=False)

data_dir = os.path.join(CUR_DIR, "../dataset/Connellsville")

env_dirs = []
for path in Path(data_dir).rglob('env_final.obj'):
    env_dirs.append(path.parent)

dense_num = 10000
def process_env(env_idx):
    print("Process for env {}".format(env_idx))
    maze = Maze(gui=False)

    start_time = time.time()
    env_dir = env_dirs[env_idx]
    print("Process {}: generating env:{}".format(id, env_dir))

    # env
    maze.clear_obstacles()
    with open(os.path.join(env_dir, "occ_grid_final.npy"), 'rb') as f:
        occ_grid = np.load(f)
    print(occ_grid.shape)

    maze.load_mesh(osp.join(env_dir, "env_final.obj"))
    maze.load_occupancy_grid(occ_grid, add_enclosing=True)

    # utils.visualize_nodes_global(osp.join(env_dir, "env_final.obj"), occ_grid, [], None, None, show=False, save=True, file_name=osp.join(env_dir, "env.png"))

    # states
    low = maze.robot.get_joint_lower_bounds()
    high = maze.robot.get_joint_higher_bounds()

    print(low, high)

    # random sampling
    col_status = []
    states = []
    for _ in range(dense_num):
        random_state = [0] * maze.robot.num_dim
        for i in range(maze.robot.num_dim):
            random_state[i] = random.uniform(low[i], high[i])
        col_status.append(maze.pb_ompl_interface.is_state_valid(random_state))
        states.append(random_state)
    # collision_states = np.array(collision_states)
    dense_G = nx.Graph()
    dense_G.add_nodes_from([("n{}".format(i), {"coords": ','.join(map(str, state)), "col": not col_status[i]}) for i, state in enumerate(states)])

    # stratified sampling, sample base first
    # free_base_states = []
    # for _ in range(dense_num):
    #     random_state = [0] * maze.robot_base.num_dim
    #     for i in range(maze.robot_base.num_dim):
    #         random_state[i] = random.uniform(low[i], high[i])
    #     col_status = maze.pb_ompl_interface_base.is_state_valid(random_state)
    #     if col_status:
    #         random_state[0] -= 2
    #         random_state[1] -= 2
    #         free_base_states.append(random_state)
    # free_states_viz = [x + maze.robot_base.fixed_joint_pos for x in free_base_states]
    # utils.visualize_nodes_local(occ_grid, free_states_viz, None, None, show=False, save=True, file_name=osp.join(env_dir, "dense_free_base.png"))

    # col_status = []
    # states = []
    # for i in range(len(free_base_states)):
    #     random_state = [0] * maze.robot.num_dim
    #     random_state[0] = free_base_states[i][0]
    #     random_state[1] = free_base_states[i][1]
    #     while True:
    #         for i in range(2, maze.robot.num_dim):
    #             random_state[i] = random.uniform(low[i], high[i])
    #         is_valid = maze.pb_ompl_interface.is_state_valid(random_state)
    #         if is_valid:
    #             col_status.append(is_valid)
    #             state_tmp = random_state.copy()
    #             state_tmp[0] -= 2
    #             state_tmp[1] -= 2
    #             states.append(state_tmp)
    #             break
    # collision_states = np.array(collision_states)
    # dense_G = nx.Graph()
    # dense_G.add_nodes_from([("n{}".format(i), {"coords": ','.join(map(str, state)), "col": not col_status[i]}) for i, state in enumerate(states)])

    # node_pos = np.array(states)
    # node_pos = np.array([state_to_numpy(dense_G.nodes[node]['coords']) for node in dense_G.nodes()])
    # utils.visualize_nodes(occ_grid, node_pos, None, None, show=False, save=True, file_name=osp.join(directory, "dense.png"))
    node_pos = np.array([state_to_numpy(dense_G.nodes[node]['coords']) for node in dense_G.nodes() if not dense_G.nodes[node]['col']])
    num_free_state = len(node_pos)
    # utils.visualize_nodes_global(occ_grid, node_pos, None, None, show=False, save=True, file_name=osp.join(env_dir, "dense_free.png"))

    print("Process {}: connecting dense graph, num_free_state = {}".format(env_idx, num_free_state))
    # nodes = dense_G.nodes()
    nodes = [node for node in dense_G.nodes() if not dense_G.nodes[node]['col']]
    node_pairs = itertools.combinations(nodes, 2)
    # print(len(list(node_pairs)))
    # print(list(node_pairs))
    pairs_to_check = []
    for node_pair in node_pairs:
        if dense_G.nodes[node_pair[0]]['col'] or dense_G.nodes[node_pair[1]]['col']:
            continue

        if not dense_G.has_edge(node_pair[0], node_pair[1]):
            s1 = state_to_numpy(dense_G.nodes[node_pair[0]]['coords'])
            s2 = state_to_numpy(dense_G.nodes[node_pair[1]]['coords'])

            # ignore edges far apart
            if math.fabs(s2[0] - s1[0]) > 2.0 or math.fabs(s2[1] - s1[1]) > 2.0:
                continue

            pairs_to_check.append((s1, s2, node_pair))

    print("Process {}: connecting dense graph, num edges to check = {}".format(env_idx, len(pairs_to_check)))
    for s1, s2, node_pair in pairs_to_check:
        if utils.is_edge_free(maze, s1, s2):
            dense_G.add_edge(node_pair[0], node_pair[1])

    print("Process {}: edge_num: {}".format(env_idx, dense_G.number_of_edges()))

    for u, v in dense_G.edges:
        s1 = state_to_numpy(dense_G.nodes[u]['coords'])
        s2 = state_to_numpy(dense_G.nodes[v]['coords'])
        dense_G[u][v]['weight'] = utils.calc_edge_len(s1, s2).item()
        assert not dense_G.nodes[u]['col']
        assert not dense_G.nodes[v]['col']

    nx.write_graphml(dense_G, osp.join(env_dir, "dense_g_rs_final.graphml"))

    end_time = time.time()
    time_taken = end_time - start_time

    print("Process {}, Estimated time left = {}".format(env_idx, -1))

# split into 8 processes
# ENV_NUM = len(env_dirs)
ENV_NUM = len(env_dirs)
print(ENV_NUM)
process_num = ENV_NUM
j = 0
while j < ENV_NUM:
    processes = []
    for i in range(j, min(ENV_NUM, j + process_num)):
        p = Process(target=process_env, args=(i,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    j += process_num



