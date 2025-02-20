import os
import os.path as osp
import sys
# sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))
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

data_dir = os.path.join(CUR_DIR, "dataset/gibson")

env_dirs = []
for path in Path(data_dir).rglob('env_small.obj'):
    env_dirs.append(path.parent)

dense_num = 10000
def process_env(env_idx):
    print("Process for env {}".format(env_idx))
    maze = Maze(gui=False)

    start_time = time.time()
    env_dir = env_dirs[env_idx]
    print("generating env:{}".format(env_dir))

    # env
    maze.clear_obstacles()
    occ_grid = np.loadtxt(osp.join(env_dir, "occ_grid_small.txt")).astype(np.uint8)
    print(occ_grid.shape)

    maze.load_mesh(osp.join(env_dir, "env_small.obj"))
    maze.load_occupancy_grid(occ_grid)

    # states
    low = maze.robot.get_joint_lower_bounds()
    high = maze.robot.get_joint_higher_bounds()

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

    # node_pos = np.array(states)
    # node_pos = np.array([state_to_numpy(dense_G.nodes[node]['coords']) for node in dense_G.nodes()])
    # utils.visualize_nodes(occ_grid, node_pos, None, None, show=False, save=True, file_name=osp.join(directory, "dense.png"))
    node_pos = np.array([state_to_numpy(dense_G.nodes[node]['coords']) for node in dense_G.nodes() if not dense_G.nodes[node]['col']])
    num_free_state = len(node_pos)
    utils.visualize_nodes_global(occ_grid, node_pos, None, None, show=False, save=True, file_name=osp.join(env_dir, "dense_free_small.png"))

    print("Process {}: connecting dense graph, num_free_state = {}".format(env_idx, num_free_state))
    # nodes = dense_G.nodes()
    nodes = [node for node in dense_G.nodes() if not dense_G.nodes[node]['col']]
    node_pairs = itertools.combinations(nodes, 2)
    # print(len(list(node_pairs)))
    # print(list(node_pairs))
    for node_pair in node_pairs:
        if dense_G.nodes[node_pair[0]]['col'] or dense_G.nodes[node_pair[1]]['col']:
            continue

        if not dense_G.has_edge(node_pair[0], node_pair[1]):
            s1 = state_to_numpy(dense_G.nodes[node_pair[0]]['coords'])
            s2 = state_to_numpy(dense_G.nodes[node_pair[1]]['coords'])

            # ignore edges far apart
            if math.fabs(s2[0] - s1[0]) > 2.0 or math.fabs(s2[1] - s1[1]) > 2.0:
                continue

            if utils.is_edge_free(maze, s1, s2):
                dense_G.add_edge(node_pair[0], node_pair[1])

    print("Process {}: edge_num: {}".format(id, dense_G.number_of_edges()))

    for u, v in dense_G.edges:
        s1 = state_to_numpy(dense_G.nodes[u]['coords'])
        s2 = state_to_numpy(dense_G.nodes[v]['coords'])
        dense_G[u][v]['weight'] = utils.calc_edge_len(s1, s2).item()
        assert not dense_G.nodes[u]['col']
        assert not dense_G.nodes[v]['col']

    nx.write_graphml(dense_G, osp.join(env_dir, "dense_g_small.graphml"))

    end_time = time.time()
    time_taken = end_time - start_time

    print("Process {} finished".format(env_idx))

# split into processes
env_num = len(env_dirs)
print(env_num)
process_num = env_num
j = 0
while j < env_num:
    processes = []
    for i in range(j, min(env_num, j + process_num)):
        p = Process(target=process_env, args=(i,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    j += process_num



