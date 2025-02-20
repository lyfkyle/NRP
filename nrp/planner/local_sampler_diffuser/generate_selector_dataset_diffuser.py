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
MAX_NUM_OF_PATHS = 100
LOCAL_ENV_SIZE = 2.0
PATH_LEN_DIFF_THRESHOLD = 2.0

PROCESS_NUM = 40

def generate_new_prm(orig_G, maze, start_pos, goal_node, mesh=None, occ_g=None, size=LOCAL_ENV_SIZE + 1):
    dense_G = nx.create_empty_copy(orig_G) # remove all edges
    # print(dense_G.number_of_edges(), dense_G.number_of_nodes())

    goal_pos = utils.node_to_numpy(dense_G, goal_node)
    # print(start_pos, goal_pos)

    # print("Connecting outside nodes to goal")
    outside_nodes = []
    for node in dense_G.nodes():
        node_pos = utils.node_to_numpy(dense_G, node)
        if (math.fabs(node_pos[0] - start_pos[0]) > LOCAL_ENV_SIZE or math.fabs(node_pos[1] - start_pos[1]) > LOCAL_ENV_SIZE) and \
            (math.fabs(node_pos[0] - start_pos[0]) <= size and math.fabs(node_pos[1] - start_pos[1]) <= size):
            nx.set_node_attributes(dense_G, {node: {"col": False, "coords": ','.join(map(str, node_pos))}})
            outside_nodes.append(node)
    # print(len(outside_nodes))

    # check valid outside nodes. path from valid outside nodes to goal should not pass through the local environment
    valid_outside_nodes = []
    for node in outside_nodes:
        node_pos = utils.node_to_numpy(dense_G, node)
        path_to_goal = utils.interpolate([goal_pos, node_pos])

        valid = True
        for p in path_to_goal:
            if math.fabs(p[0] - start_pos[0]) <= LOCAL_ENV_SIZE and math.fabs(p[1] - start_pos[1]) <= LOCAL_ENV_SIZE:
                valid = False
                break

        if valid:
            valid_outside_nodes.append(node)
            dense_G.add_edge(node, goal_node, weight=utils.calc_edge_len(node_pos, goal_pos))

    # print("Connecting inside nodes using the original graph")
    inside_nodes = []
    for node in dense_G.nodes():
        node_pos = utils.node_to_numpy(dense_G, node)
        if math.fabs(node_pos[0] - start_pos[0]) <= LOCAL_ENV_SIZE and math.fabs(node_pos[1] - start_pos[1]) <= LOCAL_ENV_SIZE:
            if not dense_G.nodes[node]['col']:
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
    dense_G.add_node(s_node, coords=','.join(map(str, start_pos)), col=False)
    s_cnt = 0
    for node in inside_nodes:
        node_pos = utils.node_to_numpy(dense_G, node)

        # ignore edges far apart
        if s_cnt < 50 and math.fabs(node_pos[0] - start_pos[0]) < 1.5 and math.fabs(node_pos[1] - start_pos[1]) < 1.5:
            if utils.is_edge_free(maze, start_pos, node_pos):
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

            valid = True
            for p in path:
                if math.fabs(p[0] - start_pos[0]) <= LOCAL_ENV_SIZE and math.fabs(p[1] - start_pos[1]) <= LOCAL_ENV_SIZE:
                    valid = False
                    break

            if valid:
                dense_G.add_edge(node_pair[0], node_pair[1], weight=utils.calc_edge_len(s1, s2))

    # print("Connecting inside nodes to outside nodes")
    node_pairs_to_check = []
    for node in inside_nodes:
        for node2 in valid_outside_nodes:
            node_pairs_to_check.append([node, node2])
    random.shuffle(node_pairs_to_check)

    pairs_to_check = []
    for node_pair in node_pairs_to_check:
        if dense_G.nodes[node_pair[0]]['col'] or dense_G.nodes[node_pair[1]]['col']:
            continue

        if orig_G.has_edge(node_pair[0], node_pair[1]):
            dense_G.add_edge(node_pair[0], node_pair[1], weight=utils.calc_edge_len(s1, s2))
            continue

        s1 = utils.node_to_numpy(dense_G, node_pair[0])
        s2 = utils.node_to_numpy(dense_G, node_pair[1])

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
            if utils.is_edge_free(maze, s1, s2):
                dense_G.add_edge(node_pair[0], node_pair[1], weight=utils.calc_edge_len(s1, s2))

    # print("Connecting expert node path")
    # for i in range(1, len(expert_node_path)):
    #     node1 = expert_node_path[i - 1]
    #     node2 = expert_node_path[i]
    #     s1 = utils.node_to_numpy(dense_G, node1)
    #     s2 = utils.node_to_numpy(dense_G, node2)
    #     dense_G.add_edge(node1, node2, weight=utils.calc_edge_len(s1, s2))

    dense_G.remove_nodes_from(list(nx.isolates(dense_G)))
    # print(dense_G.number_of_edges(), dense_G.number_of_nodes())

    return s_node, dense_G

def plan_using_PRM(maze, G, v_pos, s_node, g_node):
    free_nodes = [node for node in G.nodes() if not G.nodes[node]['col']]
    random.shuffle(free_nodes)

    # special case where v_pos is already in G.
    for node in free_nodes:
        node_pos = utils.node_to_numpy(G, node)
        if np.allclose(node_pos, v_pos):
            try:
                node_path = nx.shortest_path(G, node, g_node)
                path = [utils.node_to_numpy(G, n) for n in node_path]
            except:
                print("No path found!!")
                node_path = None
                path = None

            return node_path, path

    # Add v_node to G
    number_of_nodes = G.number_of_nodes()
    s_pos = utils.node_to_numpy(G, s_node)
    g_pos = utils.node_to_numpy(G, g_node)
    v_node = "n{}".format(number_of_nodes + 1)
    # g_node = "n{}".format(number_of_nodes + 2)
    G.add_node(v_node, coords=','.join(map(str, v_pos)), col=False)
    # G.add_node(g_node, coords=','.join(map(str, g)), col=False)

    # Connect v_node to nearby nodes
    # add connection to start_node
    # if utils.is_edge_free(maze, s_pos, v_pos):
    G.add_edge(s_node, v_node, weight=utils.calc_edge_len(v_pos, s_pos))
    # else:
    #     print("Edge not free!!")
    #     print(s_pos, v_pos)
    #     return None

    s_cnt = 0
    # g_cnt = 0
    for node in free_nodes:
        if node == s_node:
            continue

        node_pos = utils.node_to_numpy(G, node)

        # ignore edges far apart
        if math.fabs(node_pos[0] - v_pos[0]) < LOCAL_ENV_SIZE and math.fabs(node_pos[1] - v_pos[1]) < LOCAL_ENV_SIZE:
            if utils.is_edge_free(maze, v_pos, node_pos):
                G.add_edge(v_node, node, weight=utils.calc_edge_len(v_pos, node_pos))

    try:
        node_path = nx.shortest_path(G, v_node, g_node)
        path = [utils.node_to_numpy(G, node) for node in node_path]
    except:
        print("No path found!!")
        path = None

    G.remove_node(v_node)

    return node_path, path

def is_path_differ(maze, G, start_pos, expert_node_path, selected_pos, local_occ_grid, i, global_occ_grid, new_mesh_path, res_folder=None):
    g_node = expert_node_path[-1]
    s_node = expert_node_path[0]
    expert_path = [utils.node_to_numpy(G, node) for node in expert_node_path]
    g_pos = expert_path[-1]

    # path_to_boundary.append(g_pos)
    # path_to_boundary.pop(0)
    expert_waypoint_to_goal = expert_path[1:] # exclude start
    expert_path_len = utils.calc_path_len(expert_waypoint_to_goal)
    expert_path_base_len = utils.calc_path_len_base(expert_waypoint_to_goal)

    # make sure the selected position is far enough way from start
    dist_expert_to_start = utils.calc_edge_len(start_pos, expert_path[1])
    dist_selected_pos_to_start = utils.calc_edge_len(start_pos, selected_pos)
    # if dist_selected_pos_to_start - dist_expert_to_start < -0.5:
    #     return True, None, None

    # print("expert_path:", expert_waypont_to_goal)
    # print("expert_path_len:", expert_path_len)
    # utils.visualize_nodes_global(global_occ_grid, expert_path, start_pos, g_pos, show=False, save=True, file_name=osp.join(CUR_DIR, "res/gt_path_to_boundary_{}.png".format(i)))

    selected_node_path, selected_path = plan_using_PRM(maze, G, selected_pos, s_node, g_node)
    if selected_path  is None:  # there is no path from selected_pos to goal
        print("No path from selected_pos to g_pos")
        # print(selected_pos, expert_path[0])
        assert utils.is_edge_free(maze, expert_path[0], selected_pos)
        node_path = nx.shortest_path(G, s_node, g_node)
        assert node_path is not None
        return 1, -1, expert_path_len

    # path_to_boundary.append(g_pos)
    selected_waypont_to_goal = selected_path
    selected_path_len = utils.calc_path_len(selected_waypont_to_goal)
    selected_path_base_len = utils.calc_path_len_base(selected_waypont_to_goal)

    # for path to not differ. s_node must not be in the path from waypoint to goal, and the total length should not differ by 25%
    path_differ = -1
    if s_node not in selected_node_path and ((selected_path_len - expert_path_len) / (expert_path_len + 1e-6)) < 0.1 and dist_selected_pos_to_start - dist_expert_to_start >= -0.5:
        path_differ = 0

    if s_node in selected_node_path or ((selected_path_len - expert_path_len) / (expert_path_len + 1e-6)) >= 0.3:
        path_differ = 1

    # print("selected_path:", selected_waypont_to_goal)
    # print("selected_path_len:", selected_path_len)
    # print("selected_path_len_base:", selected_path_base_len)
    # utils.visualize_nodes_global(global_occ_grid, selected_waypont_to_goal[:-1], start_pos, g_pos, show=False, save=True, file_name=osp.join(CUR_DIR, "res/selected_path_to_boundary_{}_{}.png".format(i, path_differ)))

    return path_differ, selected_path_len, expert_path_len

def sample_problems(G):
    # path = dict(nx.all_pairs_shortest_path(G))
    free_nodes = [n for n in G.nodes() if not G.nodes[n]['col']]
    random.shuffle(free_nodes)

    for i, s_name in enumerate(free_nodes):
        s_name = free_nodes[i]
        start_pos = utils.node_to_numpy(G, s_name).tolist()

        for g_name in free_nodes[i:]:
            goal_pos = utils.node_to_numpy(G, g_name).tolist()

            try:
                path = nx.shortest_path(G, source=s_name, target=g_name)
            except:
                continue

            # p = [utils.node_to_numpy(G, n).tolist() for n in path]
            # for x in p:
            #     x[0] += 2
            #     x[1] += 2

            if len(path) > 2 and \
                math.fabs(goal_pos[0] - start_pos[0]) > LOCAL_ENV_SIZE and \
                math.fabs(goal_pos[1] - start_pos[1]) > LOCAL_ENV_SIZE:
                return s_name, g_name, path

    return None, None, []

def collect_gt(train_env_dirs, env_idx, env_obj_dict, data_output_dir, num_samples_per_env=100):
    maze_dir = train_env_dirs[env_idx]
    maze = Maze(gui=False)

    # print("Evaluation on {}".format(maze_dir))

    occ_grid = np.loadtxt(osp.join(maze_dir, "occ_grid_small.txt")).astype(np.uint8)
    orig_G = nx.read_graphml(osp.join(maze_dir, "dense_g_small.graphml.xml"))
    maze.clear_obstacles()
    # maze.load_occupancy_grid(occ_grid)

    idx = 0
    while idx < num_samples_per_env:
        dataset_list = []
        file_path = osp.join(data_output_dir, "data_{}_{}.pkl".format(env_idx, idx))
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
        while True:
            random_state = [0] * maze.robot.num_dim
            for i in range(maze.robot.num_dim):
                random_state[i] = random.uniform(low[i], high[i])
            if maze.pb_ompl_interface.is_state_valid(random_state):
                start_pos = np.array(random_state)
                break

        # sample goal position
        free_nodes = [node for node in orig_G.nodes() if not orig_G.nodes[node]['col']]
        while True:
            g_node = random.choice(free_nodes)
            goal_pos = utils.node_to_numpy(orig_G, g_node)
            local_goal_pos = utils.global_to_local(goal_pos, start_pos)

            if math.fabs(local_goal_pos[0]) > LOCAL_ENV_SIZE or math.fabs(local_goal_pos[1]) > LOCAL_ENV_SIZE:
                break

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

        s_node, G = generate_new_prm(orig_G, maze, start_pos, g_node, new_mesh_path, global_occ_grid)
        # utils.visualize_tree(global_occ_grid, G, start_pos, goal_pos, show=False, save=True, file_name=os.path.join(CUR_DIR, "res/tree_viz.png"))

        # Get expert path
        try:
            expert_node_path = nx.shortest_path(G, s_node, g_node)
            # expert_next_node = expert_node_path[1]
            # expert_next_node_pos = utils.node_to_numpy(G, expert_next_node)
            # local_expert_pos = utils.global_to_local(expert_next_node_pos, start_pos)
            all_node_paths = nx.all_simple_paths(G, s_node, g_node, cutoff=len(expert_node_path))  # only return simple paths no longer than shortest path

        except:
            print("This should not happen!!!")

            if os.path.exists(new_mesh_path):
                os.remove(new_mesh_path)
            assert not os.path.exists(new_mesh_path)

            continue

        # nx.all_simple_paths returns a generator, so need to exhaust it first
        # all_node_paths = list(all_node_paths)
        # print("Num of all simple paths: {}".format(len(all_node_paths)))

        # No need to make sure local export pos inside local env any more, because we are doing the clipping
        # if math.fabs(local_expert_pos[0]) > LOCAL_ENV_SIZE or math.fabs(local_expert_pos[1]) > LOCAL_ENV_SIZE:
        #     print("Local expert pos outside local environment!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        #     if os.path.exists(new_mesh_path):
        #         os.remove(new_mesh_path)
        #     assert not os.path.exists(new_mesh_path)

        #     continue

        new_expert_path = [utils.node_to_numpy(G, n) for n in expert_node_path]
        local_expert_path = [utils.global_to_local(n, start_pos) for n in new_expert_path]
        expert_path_len = utils.calc_path_len(new_expert_path)
        dataset_list.append([local_occ_grid, local_start_pos, local_goal_pos, local_expert_path, expert_path_len, expert_path_len])

        # all_node_paths.sort(key=lambda a: len(a))  # sort base on number of nodes
        # all_node_paths.sort(key=lambda a: utils.calc_path_len([utils.node_to_numpy(G, n) for n in a]))  # sort base on path length

        for node_path in all_node_paths:
            new_path = [utils.node_to_numpy(G, n) for n in node_path]
            local_path = [utils.global_to_local(n, start_pos) for n in new_path]
            path_len = utils.calc_path_len(new_path)
            dataset_list.append([local_occ_grid, local_start_pos, local_goal_pos, local_path, path_len, expert_path_len])
            # print(path_len)
            if len(dataset_list) >= MAX_NUM_OF_PATHS:
                break

        print("total num of samples: {}".format(len(dataset_list)))
        print("{}: Adding gt {}/{}".format(env_idx, idx, num_samples_per_env))
        print("expert_path_node_num: {}".format(len(local_expert_path)))
        print("expert_path_length: {}".format(expert_path_len))

        # utils.visualize_nodes_global(global_occ_grid, new_expert_path, None, None, show=False, save=True, file_name=osp.join(CUR_DIR, "res/expert_path_without_global_information_{}_{}.png".format(env_idx, idx)))

        # Get random nodes within local environment except expert node + nodes availabel in PRM
        # high = maze.robot.get_joint_higher_bounds()
        # high[0] = LOCAL_ENV_SIZE
        # high[1] = LOCAL_ENV_SIZE
        # low = maze.robot.get_joint_lower_bounds()
        # low[0] = -LOCAL_ENV_SIZE
        # low[1] = -LOCAL_ENV_SIZE

        # local_sample_pos = []
        # for _ in range(NUM_OF_RANDOM_LOCAL_SAMPLE):
        #     random_state = [0] * maze.robot.num_dim
        #     for i in range(maze.robot.num_dim):
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
        #         edge_col_free = utils.is_edge_free(maze, start_pos, v_pos)
        #     else:
        #         edge_col_free = True

        #     if edge_col_free:
        #         path_differ, selected_path_len, expert_path_len = is_path_differ(maze, G, start_pos, expert_node_path, v_pos, local_occ_grid, idx2, global_occ_grid, new_mesh_path)
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

        if os.path.exists(new_mesh_path):
            os.remove(new_mesh_path)
        assert not os.path.exists(new_mesh_path)

        file_path = osp.join(data_output_dir, "data_{}_{}.pkl".format(env_idx, idx))
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
    robot_dim = 11
    linkpos_dim = 24
    occ_grid_dim = 40
    goal_dim = robot_dim + linkpos_dim + 1
    model_name = "model"
    data_dir = osp.join(CUR_DIR, "./dataset/{}".format(model_name))
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    data_dir_t = osp.join(CUR_DIR, "./dataset/{}_t".format(model_name))
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
            p = mp.Process(target=collect_gt, args=(train_env_dirs, i, env_obj_dict, data_dir, 400), daemon=True)
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
            print(data_dir_t)
            p = mp.Process(target=collect_gt, args=(test_env_dirs, i, env_obj_dict, data_dir_t, 10), daemon=True)
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        j += process_num