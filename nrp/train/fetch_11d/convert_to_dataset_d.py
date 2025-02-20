import os.path as osp
import os
import shutil
import networkx as nx
import random
import numpy as np
import math
import pickle
from pathlib import Path
import argparse
from natsort import natsorted
import multiprocessing as mp
from multiprocessing import Lock
from multiprocessing.sharedctypes import Value

from nrp.env.fetch_11d import utils
from nrp.env.fetch_11d.maze import Fetch11DEnv


CUR_DIR = osp.dirname(osp.abspath(__file__))


def convert_data(env_idx, train_env_dirs, input_data_parent_dir, output_data_parent_dir, data_cnt, lock):
    env = Fetch11DEnv(gui=False)

    print(env_idx)
    env_dir = train_env_dirs[env_idx]

    occ_grid = utils.get_occ_grid(env_dir)
    global_prm = utils.get_prm(env_dir)
    mesh_path = utils.get_mesh_path(env_dir)

    env.clear_obstacles()
    env.load_mesh(mesh_path)
    env.load_occupancy_grid(occ_grid)

    local_data_dir = os.path.join(input_data_parent_dir)
    new_data_dir = os.path.join(output_data_parent_dir)
    if not os.path.exists(new_data_dir):
        os.makedirs(new_data_dir)

    dir_list = natsorted(os.listdir(local_data_dir))
    for file in dir_list:
        if not file.endswith("pkl"):
            continue

        file_str = file[:-4].split("_")  # remove .pkl
        if int(file_str[1]) != env_idx:
            continue

        print(f"converting env_idx:{env_idx}, file_name:{file}")
        with open(os.path.join(local_data_dir, file), "rb") as f:
            expert_path = pickle.load(f)

        start_pos = expert_path[0]
        goal_pos = expert_path[-1]
        local_occ_grid = env.get_local_occ_grid(start_pos, local_env_size=2)

        local_start_pos = utils.global_to_local(start_pos, start_pos)
        local_goal_pos = utils.global_to_local(goal_pos, start_pos)
        local_path = [utils.global_to_local(p, start_pos) for p in expert_path]
        local_expert_wp = local_path[1]

        local_env_idx = int(file_str[2])
        goal_idx = int(file_str[3])
        local_prm_path = osp.join(local_data_dir, f"dense_g_{env_idx}_{local_env_idx}_{goal_idx}.graphml")
        local_prm = nx.read_graphml(local_prm_path)

        # Find start and goal
        start_node = None
        goal_node = None
        for node in local_prm.nodes():
            node_pos = utils.node_to_numpy(local_prm, node)
            if np.allclose(np.array(node_pos), np.array(start_pos)):
                start_node = node

            if np.allclose(np.array(node_pos), np.array(goal_pos)):
                goal_node = node

        # Calculate path difference
        expert_path_len = utils.calc_path_len(expert_path)
        expert_wp_to_goal_path_len = utils.calc_path_len(expert_path[1:])
        optimal_data = []
        optimal_data.append([local_occ_grid, local_start_pos, local_goal_pos, local_expert_wp, 1, env_idx, start_pos])
        non_optimal_data = []
        for node in local_prm.nodes():
            node_pos = utils.node_to_numpy(local_prm, node)
            local_node_pos = utils.global_to_local(node_pos, start_pos)
            if not utils.is_robot_within_local_env(start_pos, node_pos, env_size=2.0):
                continue

            if node == start_node:
                continue

            # must be connectable
            if not local_prm.has_edge(start_node, node):
                continue

            node_2_goal_node_path = nx.shortest_path(local_prm, node, goal_node)
            node_2_goal_path = [utils.node_to_numpy(local_prm, n) for n in node_2_goal_node_path]
            node_2_goal_path_len = utils.calc_path_len(node_2_goal_path)
            start_2_goal_passing_node_path_len = local_prm[start_node][node]["weight"] + node_2_goal_path_len
            # print(start_2_goal_passing_node_path_len, expert_path_len)

            optimal = False
            if (
                start_node not in node_2_goal_node_path
                and ((node_2_goal_path_len - expert_wp_to_goal_path_len) / (expert_wp_to_goal_path_len + 1e-6)) < 0.1
            ):
                optimal = True

            if optimal:
                optimal_data.append(
                    [local_occ_grid, local_start_pos, local_goal_pos, local_node_pos, 1, env_idx, start_pos]
                )
            else:
                non_optimal_data.append(
                    [local_occ_grid, local_start_pos, local_goal_pos, local_node_pos, 0, env_idx, start_pos]
                )

        # Use the global PRM to add collision samples
        col_data = []
        for node in global_prm.nodes():
            nodepos = utils.node_to_numpy(global_prm, node)
            if (
                math.fabs(nodepos[0] - start_pos[0]) > local_env_size
                or math.fabs(nodepos[1] - start_pos[1]) > local_env_size
            ):
                continue

            if global_prm.nodes[node]["col"]:
                local_node_pos = utils.global_to_local(node_pos, start_pos)
                col_data.append(
                    [local_occ_grid, local_start_pos, local_goal_pos, local_node_pos, -1, env_idx, start_pos]
                )

        print(
            f"num of near-optimal data: {len(optimal_data)}, non-optimal data: {len(non_optimal_data)}, col data: {len(col_data)}"
        )

        # Add the same number of optimal and non-optimal samples into dataset
        num_optimal_data = min(len(non_optimal_data) + len(col_data), min(5, len(optimal_data)))  # at most 5
        num_non_optimal_data = min(int(num_optimal_data / 2), len(non_optimal_data))
        num_col_data = num_optimal_data - num_non_optimal_data

        assert num_optimal_data >= 0
        assert num_non_optimal_data >= 0
        assert num_col_data >= 0

        optimal_data = optimal_data[:num_optimal_data]
        non_optimal_data = non_optimal_data[:num_non_optimal_data]
        col_data = col_data[:num_col_data]

        lock.acquire()
        for data in optimal_data:
            with open(os.path.join(new_data_dir, f"data_{data_cnt.value}.pkl"), "wb") as f:
                pickle.dump(data, f)
            data_cnt.value += 1

        for data in non_optimal_data:
            with open(os.path.join(new_data_dir, f"data_{data_cnt.value}.pkl"), "wb") as f:
                pickle.dump(data, f)
            data_cnt.value += 1

        for data in col_data:
            with open(os.path.join(new_data_dir, f"data_{data_cnt.value}.pkl"), "wb") as f:
                pickle.dump(data, f)
            data_cnt.value += 1
        lock.release()

    print(data_cnt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--env", default="fetch_11d")
    args = parser.parse_args()

    # constants
    robot_dim = 11
    linkpos_dim = 24
    occ_grid_dim = 40
    occ_grid_dim_z = 20
    goal_dim = robot_dim + linkpos_dim + 1
    model_name = "train_01_critical_v2"
    output_parent_data_dir = osp.join(CUR_DIR, f"dataset/train_01_critical_v2_out_d")
    viz_data_dir = osp.join(CUR_DIR, f"dataset/train_01_critical_v2_out_d_viz")
    local_env_size = 2.0

    input_data_parent_dir = osp.join(CUR_DIR, "./dataset/{}".format(model_name))

    if not os.path.exists(viz_data_dir):
        os.makedirs(viz_data_dir)

    process_num = 25
    train_env_dirs = utils.TRAIN_ENV_DIRS
    datacnt = Value("i", 0)
    lock = Lock()
    j = 0
    while j < len(train_env_dirs):
        processes = []
        print("Running on env {} to {}".format(j, min(len(train_env_dirs), j + process_num)))
        for i in range(j, min(len(train_env_dirs), j + process_num)):
            p = mp.Process(
                target=convert_data,
                args=(i, train_env_dirs, input_data_parent_dir, output_parent_data_dir, datacnt, lock),
                daemon=True,
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        j += process_num
