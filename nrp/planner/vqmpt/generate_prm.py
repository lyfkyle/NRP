import os
import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../../'))
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

# from env.snake_8d.maze import Snake8DEnv
# from env.snake_8d import utils
from env.fetch_11d.maze import Fetch11DEnv
from env.fetch_11d import utils

CUR_DIR = osp.dirname(osp.abspath(__file__))

def state_to_numpy(state):
    strlist = state.split(',')
    val_list = [float(s) for s in strlist]
    return np.array(val_list)


def process_env(env_dir, env_idx, dense_num, empty=False):
    print("Process for env {}".format(env_idx))
    # maze = Snake8DEnv(gui=False)
    maze = Fetch11DEnv(gui=False)

    start_time = time.time()
    if empty:
        print("Process {}: generating env: empty".format(id))
        maze.clear_obstacles()
        # occ_grid = np.zeros((100, 100))
        occ_grid = np.zeros((150, 150, 20))
        base_x_bounds = [0, 15]
        base_y_bounds = [0, 15]
        maze.robot.set_base_bounds(base_x_bounds, base_y_bounds)
    else:
        print("Process {}: generating env:{}".format(id, env_dir))

        # env
        maze.clear_obstacles()
        occ_grid = utils.get_occ_grid(env_dir)
        print(occ_grid.shape)

        maze.load_mesh(utils.get_mesh_path(env_dir))
        maze.load_occupancy_grid(occ_grid, add_enclosing=True)

        # utils.visualize_nodes_global(osp.join(env_dir, "env_final.obj"), occ_grid, [], None, None, show=False, save=True, file_name=osp.join(env_dir, "env.png"))

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
    # utils.visualize_nodes_global(occ_grid, node_pos, None, None, show=False, save=True, file_name=osp.join(env_dir, "dense_free_small.png"))
    # utils.visualize_nodes_global(None, None, node_pos, None, None, show=False, save=True, file_name=osp.join(env_dir, "dense_free_small.png"))

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

    # nx.write_graphml(dense_G, osp.join(env_dir, "dense_g_small.graphml"))
    nx.write_graphml(dense_G, osp.join(env_dir, "dense_g_rs_final.graphml"))

    end_time = time.time()
    time_taken = end_time - start_time

    print("Process {} finished".format(env_idx))


if __name__ == '__main__':

    # data_dir = os.path.join(CUR_DIR, "../../env/snake_8d/dataset/gibson/mytest")
    data_dir = os.path.join(CUR_DIR, "../../env/fetch_11d/dataset/gibson/mytest")

    # env_dirs = []
    # # for path in Path(data_dir).rglob('env_small.obj'):
    # for path in Path(data_dir).rglob('env_final.obj'):
    #     env_dirs.append(path.parent)

    dense_num = 10000

    # split into 8 processes
    # ENV_NUM = len(env_dirs)
    # ENV_NUM = len(env_dirs)
    # print(ENV_NUM)
    # process_num = ENV_NUM
    # j = 0
    # while j < ENV_NUM:
    #     processes = []
    #     for i in range(j, min(ENV_NUM, j + process_num)):
    #         p = Process(target=process_env, args=(env_dirs, i, dense_num))
    #         p.start()
    #         processes.append(p)

    #     for p in processes:
    #         p.join()

    #     j += process_num

    # empty_data_dir = os.path.join(CUR_DIR, "dataset/8d/stage1")
    empty_data_dir = os.path.join(CUR_DIR, "dataset/11d/train/env_-1")
    if not os.path.exists(empty_data_dir):
        os.mkdir(empty_data_dir)

    process_env(empty_data_dir, 0, dense_num, empty=True)



