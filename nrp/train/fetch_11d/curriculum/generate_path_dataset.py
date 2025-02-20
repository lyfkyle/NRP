import os.path as osp
import sys
import os

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), "../../"))

import shutil
import networkx as nx
import random
import numpy as np
import math
import pickle
from pathlib import Path
import argparse
from natsort import natsorted
from collections import defaultdict

# from multiprocessing import Process, Manager
import torch.multiprocessing as mp

import itertools

from env.fetch_11d import utils
from env.fetch_11d.maze import Fetch11DEnv

CUR_DIR = osp.dirname(osp.abspath(__file__))


PROCESS_NUM = 40

# def calc_edge_len_2(start_pos, q1, q2, step_size=0.2):
#     path = utils.interpolate([q1, q2], step_size=step_size)

#     boundary_idx = 0
#     for i, p in enumerate(path):
#         if not utils.is_robot_within_local_env_2(start_pos, p):
#             boundary_idx = i - 1
#             break

#     path_len = step_size * boundary_idx + (len(path) - boundary_idx) * step_size

#     return path_len


def generate_new_prm(orig_G, env, start_pos, local_env_size=2.0, mesh=None, occ_g=None):
    dense_G = nx.create_empty_copy(orig_G)  # remove all edges
    # print(dense_G.number_of_edges(), dense_G.number_of_nodes())

    # goal_pos = utils.node_to_numpy(dense_G, goal_node)
    # print(start_pos, goal_pos)

    # print("Connecting outside nodes to goal")
    # Outside nodes are those that are within 1 meter away from local env
    size = local_env_size + 1
    outside_nodes = []
    for node in dense_G.nodes():
        node_pos = utils.node_to_numpy(dense_G, node)
        if (
            math.fabs(node_pos[0] - start_pos[0]) > local_env_size
            or math.fabs(node_pos[1] - start_pos[1]) > local_env_size
        ) and (math.fabs(node_pos[0] - start_pos[0]) <= size and math.fabs(node_pos[1] - start_pos[1]) <= size):
            nx.set_node_attributes(dense_G, {node: {"col": False, "coords": ",".join(map(str, node_pos))}})
            outside_nodes.append(node)

    for node in outside_nodes:
        node_pos = utils.node_to_numpy(dense_G, node)
        assert not utils.is_robot_within_local_env_2(start_pos, node_pos)

    # print(len(outside_nodes))

    # check valid outside nodes. path from valid outside nodes to goal should not pass through the local environment
    # valid_outside_nodes = []
    # for node in outside_nodes:
    #     node_pos = utils.node_to_numpy(dense_G, node)
    #     path_to_goal = utils.interpolate([goal_pos, node_pos])

    #     valid = True
    #     for p in path_to_goal:
    #         if math.fabs(p[0] - start_pos[0]) <= local_env_size and math.fabs(p[1] - start_pos[1]) <= local_env_size:
    #             valid = False
    #             break

    #     if valid:
    #         valid_outside_nodes.append(node)
    #         dense_G.add_edge(node, goal_node, weight=utils.calc_edge_len(node_pos, goal_pos))

    # print("Connecting inside nodes using the original graph")
    # Inside nodes are nodes within local environment
    inside_nodes = []
    for node in dense_G.nodes():
        node_pos = utils.node_to_numpy(dense_G, node)
        if (
            math.fabs(node_pos[0] - start_pos[0]) <= local_env_size
            and math.fabs(node_pos[1] - start_pos[1]) <= local_env_size
        ):
            if not dense_G.nodes[node]["col"]:
                inside_nodes.append(node)

    # use the original graph to connect inside nodes
    node_pairs = itertools.combinations(inside_nodes, 2)
    for node_pair in node_pairs:
        if orig_G.has_edge(node_pair[0], node_pair[1]):
            s1 = utils.node_to_numpy(dense_G, node_pair[0])
            s2 = utils.node_to_numpy(dense_G, node_pair[1])
            dense_G.add_edge(node_pair[0], node_pair[1], weight=utils.calc_edge_len(s1, s2))

    # print("Connecting start_pos to inside nodes")
    s_node = "n{}".format(dense_G.number_of_nodes() + 1)
    dense_G.add_node(s_node, coords=",".join(map(str, start_pos)), col=False)
    s_cnt = 0
    for node in inside_nodes:
        node_pos = utils.node_to_numpy(dense_G, node)

        # ignore edges far apart
        if s_cnt < 50 and math.fabs(node_pos[0] - start_pos[0]) < 1.5 and math.fabs(node_pos[1] - start_pos[1]) < 1.5:
            if utils.is_edge_free(env, start_pos, node_pos):
                dense_G.add_edge(s_node, node, weight=utils.calc_edge_len(start_pos, node_pos))

            s_cnt += 1

    # print("Connecting outside nodes using the original graph")
    pairs_to_check = []
    node_pairs = itertools.combinations(outside_nodes, 2)
    for node_pair in node_pairs:
        s1 = utils.node_to_numpy(dense_G, node_pair[0])
        s2 = utils.node_to_numpy(dense_G, node_pair[1])
        if orig_G.has_edge(node_pair[0], node_pair[1]):
            dense_G.add_edge(node_pair[0], node_pair[1], weight=utils.calc_edge_len(s1, s2))
        else:
            if math.fabs(s2[0] - s1[0]) < 1.0 and math.fabs(s2[1] - s1[1]) < 1.0:
                pairs_to_check.append((s1, s2, node_pair))

    # print("num edges to check = {}".format(len(pairs_to_check)))
    random.shuffle(pairs_to_check)
    if len(pairs_to_check) > 2500:
        pairs_to_check = pairs_to_check[:2500]

    for s1, s2, node_pair in pairs_to_check:
        if dense_G.has_edge(node_pair[0], node_pair[1]):
            continue
        else:
            path = utils.interpolate([s1, s2])

            # If the edge does not pass through local environment, then it is definitely collision-free.
            valid = True
            for p in path:
                if (
                    math.fabs(p[0] - start_pos[0]) <= local_env_size
                    and math.fabs(p[1] - start_pos[1]) <= local_env_size
                ):
                    valid = False
                    break

            if valid:
                dense_G.add_edge(node_pair[0], node_pair[1], weight=utils.calc_edge_len(s1, s2))

    # print("Connecting inside nodes to outside nodes")
    node_pairs_to_check = []
    for node in inside_nodes:
        for node2 in outside_nodes:
            node_pairs_to_check.append([node, node2])
    random.shuffle(node_pairs_to_check)

    pairs_to_check = []
    for node_pair in node_pairs_to_check:
        if dense_G.nodes[node_pair[0]]["col"] or dense_G.nodes[node_pair[1]]["col"]:
            continue

        s1 = utils.node_to_numpy(dense_G, node_pair[0])
        s2 = utils.node_to_numpy(dense_G, node_pair[1])

        if orig_G.has_edge(node_pair[0], node_pair[1]):
            dense_G.add_edge(node_pair[0], node_pair[1], weight=utils.calc_edge_len(s1, s2))
            continue

        # ignore edges far apart
        if np.allclose(s1, s2) or math.fabs(s2[0] - s1[0]) > 1.0 or math.fabs(s2[1] - s1[1]) > 1.0:
            continue

        pairs_to_check.append((s1, s2, node_pair))

    # print("num edges to check = {}".format(len(pairs_to_check)))
    random.shuffle(pairs_to_check)
    if len(pairs_to_check) > 2500:
        pairs_to_check = pairs_to_check[:2500]

    for s1, s2, node_pair in pairs_to_check:
        if dense_G.has_edge(node_pair[0], node_pair[1]):
            continue
        else:
            if utils.is_edge_free(env, s1, s2):
                dense_G.add_edge(node_pair[0], node_pair[1], weight=utils.calc_edge_len(s1, s2))

    # print("Connecting expert node path")
    # for i in range(1, len(expert_node_path)):
    #     node1 = expert_node_path[i - 1]
    #     node2 = expert_node_path[i]
    #     s1 = utils.node_to_numpy(dense_G, node1)
    #     s2 = utils.node_to_numpy(dense_G, node2)
    #     dense_G.add_edge(node1, node2, weight=utils.calc_edge_len(s1, s2))

    dense_G.remove_nodes_from(list(nx.isolates(dense_G)))
    outside_nodes = [node for node in outside_nodes if dense_G.has_node(node)]
    # print(dense_G.number_of_edges(), dense_G.number_of_nodes())

    return s_node, dense_G, outside_nodes


def collect_gt(
    data_dir, train_env_dirs, env_idx, env_obj_dict, local_env_size, num_local_envs=10, num_goal_per_local_env=10
):
    env_dir = train_env_dirs[env_idx]
    env = Fetch11DEnv(gui=False)

    # print("Evaluation on {}".format(env_dir))

    occ_grid = utils.get_occ_grid(env_dir)
    orig_G = utils.get_prm(env_dir)
    mesh_path = utils.get_mesh_path(env_dir)

    env.clear_obstacles()
    # env.load_occupancy_grid(occ_grid)

    data_idx = 0
    local_env_idx = 0
    while local_env_idx < num_local_envs:
        file_path = osp.join(data_dir, "data_{}_{}_{}.pkl".format(env_idx, local_env_idx, num_goal_per_local_env - 1))
        if os.path.exists(file_path):
            print("Skipping data_{}_{}".format(env_idx, local_env_idx))
            local_env_idx += 1
            continue

        env.clear_obstacles()
        env.load_mesh(mesh_path)
        env.load_occupancy_grid(occ_grid)

        high = env.robot.get_joint_higher_bounds()
        low = env.robot.get_joint_lower_bounds()

        # sample start_pos
        while True:
            random_state = [0] * env.robot.num_dim
            for i in range(env.robot.num_dim):
                random_state[i] = random.uniform(low[i], high[i])
            if env.pb_ompl_interface.is_state_valid(random_state):
                start_pos = np.array(random_state)
                break

        # start_pos = utils.node_to_numpy(orig_G, start)
        # goal_pos = utils.node_to_numpy(orig_G, goal)
        # local_start_pos = utils.global_to_local(start_pos, start_pos)
        # assert math.fabs(local_goal_pos[0]) > local_env_size or math.fabs(local_goal_pos[1]) > local_env_size

        obj_idx = env_obj_dict[env_idx]
        tmp_mesh_file_name = "tmp_{}_{}.obj".format(env_idx, obj_idx)
        env_obj_dict[env_idx] += 1
        global_occ_grid, new_mesh_path = env.clear_obstacles_outside_local_occ_grid(
            start_pos, local_env_size, tmp_mesh_file_name
        )
        # utils.visualize_nodes_global(osp.join(env_dir, "env_large.obj"), occ_grid, [], None, None, show=False, save=True, file_name=osp.join(CUR_DIR, "res/global_{}_{}.png".format(env_idx, idx)))
        # utils.visualize_nodes_global(new_mesh_path, global_occ_grid, [], None, None, show=False, save=True, file_name=osp.join(CUR_DIR, "res/local_{}_{}.png".format(env_idx, idx)))

        start_node, G, outside_nodes = generate_new_prm(
            orig_G,
            env,
            start_pos,
            local_env_size,
            new_mesh_path,
            global_occ_grid,
        )
        # utils.visualize_tree(global_occ_grid, G, start_pos, goal_pos, show=False, save=True, file_name=os.path.join(CUR_DIR, "res/tree_viz.png"))

        for goal_idx in range(num_goal_per_local_env):
            cur_G = G.copy()

            # sample goal_pos
            while True:
                random_state = [0] * env.robot.num_dim
                for i in range(env.robot.num_dim):
                    random_state[i] = random.uniform(low[i], high[i])
                if env.pb_ompl_interface.is_state_valid(random_state):
                    goal_pos = np.array(random_state)
                    local_goal_pos = utils.global_to_local(goal_pos, start_pos)
                    if math.fabs(local_goal_pos[0]) > local_env_size or math.fabs(local_goal_pos[1]) > local_env_size:
                        break

            # add goal node to G
            goal_node = "n{}".format(cur_G.number_of_nodes() + 1)
            cur_G.add_node(goal_node, coords=",".join(map(str, goal_pos)), col=False)

            # connect outside nodes to goal
            for node in outside_nodes:
                node_pos = utils.node_to_numpy(cur_G, node)
                path_to_goal = utils.interpolate([goal_pos, node_pos])

                # If it does not pass through local environment, it is connectable to goal.
                valid = True
                for p in path_to_goal:
                    if (
                        math.fabs(p[0] - start_pos[0]) <= local_env_size
                        and math.fabs(p[1] - start_pos[1]) <= local_env_size
                    ):
                        valid = False
                        break

                if valid:
                    cur_G.add_edge(node, goal_node, weight=utils.calc_edge_len(node_pos, goal_pos))

            # Get expert path
            try:
                expert_node_path = nx.shortest_path(cur_G, start_node, goal_node)
                expert_next_node = expert_node_path[1]
                expert_next_node_pos = utils.node_to_numpy(cur_G, expert_next_node)
                local_expert_pos = utils.global_to_local(expert_next_node_pos, start_pos)
            except:
                print("No path to sampled goal")
                # utils.visualize_nodes_global(
                #     new_mesh_path,
                #     global_occ_grid,
                #     [],
                #     start_pos,
                #     goal_pos,
                #     show=False,
                #     save=True,
                #     file_name=osp.join(
                #         CUR_DIR,
                #         "tmp/no_path_{}_{}_{}.png".format(
                #             env_idx, local_env_idx, goal_idx
                #         ),
                #     ),
                # )
                continue

            if math.fabs(local_expert_pos[0]) > local_env_size or math.fabs(local_expert_pos[1]) > local_env_size:
                print("Local expert pos outside local environment!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                continue

            expert_path = [utils.node_to_numpy(cur_G, n) for n in expert_node_path]
            if local_env_idx < 2 and goal_idx < 2:
                utils.visualize_nodes_global(
                    new_mesh_path,
                    global_occ_grid,
                    expert_path,
                    None,
                    None,
                    show=False,
                    save=True,
                    file_name=osp.join(
                        CUR_DIR,
                        "../tmp/expert_path_without_global_information_{}_{}_{}_{}.png".format(
                            env_idx, local_env_idx, goal_idx, local_env_size
                        ),
                    ),
                )

            file_path = osp.join(data_dir, "data_{}_{}_{}.pkl".format(env_idx, local_env_idx, goal_idx))
            with open(file_path, "wb") as f:
                # print("Dumping to {}".format(file_path))
                pickle.dump(expert_path, f)

        # Get random nodes within local environment except expert node + nodes availabel in PRM
        # high = env.robot.get_joint_higher_bounds()
        # high[0] = LOCAL_ENV_SIZE
        # high[1] = LOCAL_ENV_SIZE
        # low = env.robot.get_joint_lower_bounds()
        # low[0] = -LOCAL_ENV_SIZE
        # low[1] = -LOCAL_ENV_SIZE

        # local_sample_pos = []
        # for _ in range(NUM_OF_RANDOM_LOCAL_SAMPLE):
        #     random_state = [0] * env.robot.num_dim
        #     for i in range(env.robot.num_dim):
        #         random_state[i] = random.uniform(low[i], high[i])
        #     local_sample_pos.append(random_state)

        # for node in G.nodes():
        #     node_pos = utils.node_to_numpy(G, node)
        #     if G.has_edge(node, s_node) and node != expert_next_node and len(local_sample_pos) < NUM_OF_RANDOM_LOCAL_SAMPLE*2:
        #         local_sample_pos.append(utils.global_to_local(node_pos, start_pos))
        # # print(len(local_sample_pos))

        # # dataset_list.append([env_idx, local_occ_grid, start, goal, sample_nodes, expert_next_node])
        # # dataset_list.append([local_occ_grid, local_start_pos, local_goal_pos, local_sample_pos, local_expert_pos, col_status, expert_node_path, start_pos])
        # print("{}: Adding gt {}/{}".format(env_idx, idx, num_samples_per_env))

        # # print(local_start_pos, local_goal_pos, local_expert_pos)
        # # utils.visualize_nodes_local(local_occ_grid, [local_expert_pos], local_start_pos, local_goal_pos, show=False, save=True, file_name=osp.join(CUR_DIR, "res/gt_{}.png".format(i)))

        # # path_differ = True
        # neg_samples = []
        # pos_samples = [local_expert_pos]
        # col_samples = []
        # for idx2, local_v_pos in enumerate(local_sample_pos):
        #     v_pos = utils.local_to_global(local_v_pos, start_pos)
        #     if idx2 < NUM_OF_RANDOM_LOCAL_SAMPLE:
        #         edge_col_free = utils.is_edge_free(env, start_pos, v_pos)
        #     else:
        #         edge_col_free = True

        #     if edge_col_free:
        #         path_differ, selected_path_len, expert_path_len = is_path_differ(env, G, start_pos, expert_node_path, v_pos, local_occ_grid, idx2, global_occ_grid, new_mesh_path)
        #         dataset_list.append([local_occ_grid, local_start_pos, local_goal_pos, local_v_pos, local_expert_pos, selected_path_len, expert_path_len])
        #         if path_differ == 1:
        #             neg_samples.append(local_v_pos)
        #         elif path_differ == 0:
        #             pos_samples.append(local_v_pos)
        #     else:
        #         dataset_list.append([local_occ_grid, local_start_pos, local_goal_pos, local_v_pos, local_expert_pos, -1, -1])
        #         col_samples.append(local_v_pos)

        # print("num_pos {}, num_neg: {}, num_col: {}".format(len(pos_samples), len(neg_samples), len(col_samples)))
        # print("{}: Adding to output {}/{}".format(env_idx, idx, num_samples_per_env))

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

        # file_path = osp.join(data_dir, "data_{}_{}.pkl".format(env_idx, idx))
        # with open(file_path, 'wb') as f:
        #     # print("Dumping to {}".format(file_path))
        #     pickle.dump(dataset_list, f)

        if os.path.exists(new_mesh_path):
            os.remove(new_mesh_path)
        assert not os.path.exists(new_mesh_path)

        local_env_idx += 1


def convert_data(train_env_dirs, input_data_parent_dir, output_data_parent_dir, local_env_size=None):
    env = Fetch11DEnv(gui=False)
    data_cnts = defaultdict(int)

    for env_size in range(2, 6):
        # data_cnt = data_cnts[local_env_size]
        new_data_dir = os.path.join(output_data_parent_dir, f"{env_size}")
        if not os.path.exists(new_data_dir):
            os.makedirs(new_data_dir)

        data_cnts[env_size] = len(os.listdir(new_data_dir))

    for env_idx in range(25):
        print(env_idx)
        env_dir = train_env_dirs[env_idx]

        occ_grid = utils.get_occ_grid(env_dir)
        # orig_G = utils.get_prm(env_dir)
        mesh_path = utils.get_mesh_path(env_dir)

        env.clear_obstacles()
        env.load_mesh(mesh_path)
        env.load_occupancy_grid(occ_grid)

        if local_env_size is None:
            min_local_env = 2
            max_local_env = 5
        else:
            min_local_env = local_env_size
            max_local_env = local_env_size

        for env_size in range(min_local_env, max_local_env + 1):
            local_data_dir = os.path.join(input_data_parent_dir, f"{env_size}")
            new_data_dir = os.path.join(output_data_parent_dir, f"{env_size}")
            if not os.path.exists(new_data_dir):
                os.makedirs(new_data_dir)

            dir_list = natsorted(os.listdir(local_data_dir))
            data_cnt = data_cnts[env_size]
            for file in dir_list:
                if int(file.split("_")[1]) != env_idx:
                    continue

                print(f"converting env_idx:{env_idx}, local_env_size:{env_size}, file_name:{file}")
                with open(os.path.join(local_data_dir, file), "rb") as f:
                    expert_path = pickle.load(f)

                start_pos = expert_path[0]
                goal_pos = expert_path[-1]
                local_occ_grid = env.get_local_occ_grid(start_pos, local_env_size=2)

                local_start_pos = utils.global_to_local(start_pos, start_pos)
                local_goal_pos = utils.global_to_local(goal_pos, start_pos)
                local_path = [utils.global_to_local(p, start_pos) for p in expert_path]

                with open(os.path.join(new_data_dir, f"data_{data_cnt}.pkl"), "wb") as f:
                    pickle.dump([local_occ_grid, local_start_pos, local_goal_pos, local_path], f)

                # shutil.copy(os.path.join(local_data_dir, file), os.path.join(new_data_dir, f"data_{data_cnt}.pkl"))
                # print(file)
                data_cnt += 1

            data_cnts[env_size] = data_cnt

    print(data_cnts)


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
    model_name = "train"
    output_parent_data_dir = osp.join(CUR_DIR, f"dataset/train2")

    # env dirs
    env_num = 25
    train_env_dir = osp.join(CUR_DIR, f"../../env/{args.env}/dataset/gibson/train")
    train_env_dirs = []
    for p in Path(train_env_dir).rglob("env.obj"):
        train_env_dirs.append(p.parent)
    assert len(train_env_dirs) == env_num
    test_env_dir = osp.join(CUR_DIR, f"../../env/{args.env}/dataset/gibson/mytest")
    test_env_dirs = []
    for p in Path(test_env_dir).rglob("env.obj"):
        test_env_dirs.append(p.parent)
    assert len(test_env_dirs) == 5

    # hyperparameters
    target_data_cnt = 500000

    iter_num = 0
    data_cnt = 0
    # while data_cnt < target_data_cnt:

    # Run in train env
    print("----------- Collecting from train env -------------")
    print("Collecting gt")
    process_num = 25
    manager = mp.Manager()
    # dataset_dict = manager.dict()

    local_env_size = 2
    while local_env_size <= 5:
        data_dir = osp.join(CUR_DIR, "./dataset/{}/{}".format(model_name, local_env_size))
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        env_obj_dict = manager.dict()
        for env_idx in range(len(train_env_dirs)):
            env_obj_dict[env_idx] = 0

        j = 0
        while j < len(train_env_dirs):
            processes = []
            print("Running on env {} to {}".format(j, min(len(train_env_dirs), j + process_num)))
            for i in range(j, min(len(train_env_dirs), j + process_num)):
                p = mp.Process(
                    target=collect_gt,
                    args=(data_dir, train_env_dirs, i, env_obj_dict, local_env_size, 200),
                    daemon=True,
                )
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

            j += process_num

        local_env_size += 1

    # convert data
    input_data_parent_dir = osp.join(CUR_DIR, "./dataset/{}".format(model_name))
    convert_data(train_env_dirs, input_data_parent_dir, output_parent_data_dir)
