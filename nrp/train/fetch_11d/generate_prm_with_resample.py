import os
import os.path as osp
import networkx as nx
import numpy as np
import itertools
import random
from pathlib import Path
import time
import math
from multiprocessing import Process
import argparse

from nrp.env.fetch_11d.env import Fetch11DEnv
from nrp.env.fetch_11d import utils

CUR_DIR = osp.dirname(osp.abspath(__file__))
TURNING_RADIUS = 0.1
PRM_CONNECT_RADIUS = 2.0


def state_to_numpy(state):
    strlist = state.split(",")
    val_list = [float(s) for s in strlist]
    return np.array(val_list)


def process_env(env_dirs, env_idx):
    print("Process for env {}".format(env_idx))
    env = Fetch11DEnv(gui=False)

    start_time = time.time()
    env_dir = env_dirs[env_idx]
    print("Process {}: generating env:{}".format(id, env_dir))

    # env
    occ_grid = utils.get_occ_grid(env_dir)
    mesh_path = utils.get_mesh_path(env_dir)

    env.clear_obstacles()
    env.load_mesh(mesh_path)
    env.load_occupancy_grid(occ_grid, add_enclosing=True)

    # utils.visualize_nodes_global(osp.join(env_dir, "env_final.obj"), occ_grid, [], None, None, show=False, save=True, file_name=osp.join(env_dir, "env.png"))

    # states
    low = env.robot.get_joint_lower_bounds()
    high = env.robot.get_joint_higher_bounds()

    print(low, high)

    dense_G = nx.Graph()

    # random sampling
    for idx in range(dense_num):
        sample_pos = [0] * env.robot.num_dim
        for i in range(env.robot.num_dim):
            sample_pos[i] = random.uniform(low[i], high[i])

        col_status = not env.pb_ompl_interface.is_state_valid(sample_pos)
        sample_node = "n{}".format(idx)
        dense_G.add_node(sample_node, coords=",".join(map(str, sample_pos)), col=col_status)

        if col_status:
            continue

        # Connect to neighbours
        for node in dense_G.nodes():
            node_pos = state_to_numpy(dense_G.nodes[node]["coords"]).tolist()

            # ignore edges far apart
            if (
                math.fabs(sample_pos[0] - node_pos[0]) > PRM_CONNECT_RADIUS
                or math.fabs(sample_pos[1] - node_pos[1]) > PRM_CONNECT_RADIUS
            ):
                continue

            if utils.is_edge_free(env, node_pos, sample_pos):
                dense_G.add_edge(node, sample_node, weight=utils.calc_edge_len(node_pos, sample_pos))

    utils.visualize_tree_simple(
        occ_grid, dense_G, None, None, show=False, save=True, file_name=osp.join(env_dir, f"tree_v2_1.png")
    )

    print(f"num of edges: {dense_G.number_of_edges()}")
    for idx in range(dense_num, dense_num_resample):
        # weighted sampling
        # - select node_pos according to weight
        free_nodes = [node for node in dense_G.nodes() if not dense_G.nodes[node]["col"]]
        nodes_weights = np.array([(1.0 / (dense_G.degree[n] + 1)) for n in free_nodes], dtype=np.float32)
        nodes_weights = nodes_weights / np.sum(nodes_weights)
        indices = np.arange(len(nodes_weights))
        node_indice = np.random.choice(indices, p=nodes_weights)
        node = free_nodes[node_indice]
        node_pos = state_to_numpy(dense_G.nodes[node]["coords"])

        # - sample around node_pos
        cur_low = low.copy()
        cur_high = high.copy()
        cur_low[0] = max(node_pos[0] - 1.0, low[0])
        cur_low[1] = max(node_pos[1] - 1.0, low[1])
        cur_high[0] = min(node_pos[0] + 1.0, high[0])
        cur_high[1] = min(node_pos[1] + 1.0, high[1])
        sample_pos = [0] * env.robot.num_dim
        for i in range(env.robot.num_dim):
            sample_pos[i] = random.uniform(cur_low[i], cur_high[i])

        # - calculate collision
        col_status = not env.pb_ompl_interface.is_state_valid(sample_pos)
        sample_node = "n{}".format(idx)
        dense_G.add_node(sample_node, coords=",".join(map(str, sample_pos)), col=col_status)
        if col_status:
            continue

        # - Connect to neighbours
        for node in dense_G.nodes():
            node_pos = state_to_numpy(dense_G.nodes[node]["coords"]).tolist()

            # ignore edges far apart
            if (
                math.fabs(sample_pos[0] - node_pos[0]) > PRM_CONNECT_RADIUS
                or math.fabs(sample_pos[1] - node_pos[1]) > PRM_CONNECT_RADIUS
            ):
                continue

            if utils.is_edge_free(env, node_pos, sample_pos):
                dense_G.add_edge(node, sample_node, weight=utils.calc_edge_len(node_pos, sample_pos))

    nx.write_graphml(dense_G, osp.join(env_dir, f"dense_g_v2.graphml"))
    utils.visualize_tree_simple(
        occ_grid, dense_G, None, None, show=False, save=True, file_name=osp.join(env_dir, f"tree_v2_2.png")
    )


if __name__ == "__main__":
    # env_dirs = utils.TEST_ENV_DIRS
    env_dirs = utils.TRAIN_ENV_DIRS

    dense_num = 5000
    dense_num_resample = 10000
    max_process_num = 20

    # split into processes
    env_num = len(env_dirs)
    print(env_num)
    process_num = min(env_num, max_process_num)
    j = 0
    while j < env_num:
        processes = []
        for i in range(j, min(env_num, j + process_num)):
            p = Process(target=process_env, args=(env_dirs, i))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        j += process_num
