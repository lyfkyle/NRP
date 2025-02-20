import os
import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))

import json
import numpy as np
import math
import matplotlib.pyplot as plt
import argparse
import networkx as nx
import random
import torch.multiprocessing as mp

from env.snake_8d import utils
from env.snake_8d.maze import Snake8DEnv

CUR_DIR = osp.dirname(osp.abspath(__file__))

EDGE_LEN_THRESHOLD = 1.0
LOAD = False

def get_dist_to_g(maze, G, v_pos, goal_pos):
    free_nodes = [node for node in G.nodes() if not G.nodes[node]['col']]
    random.shuffle(free_nodes)

    # find g_node
    g_node = None
    for node in free_nodes:
        node_pos = utils.node_to_numpy(G, node)
        if np.allclose(node_pos, np.array(goal_pos)):
            g_node = node
            break
    assert g_node is not None

    # Add v_node to G
    number_of_nodes = G.number_of_nodes()
    v_node = "n{}".format(number_of_nodes + 1)
    G.add_node(v_node, coords=','.join(map(str, v_pos)), col=False)

    # Connect v_node to nearby nodes
    for node in free_nodes:
        node_pos = utils.node_to_numpy(G, node)

        cnt = 0
        # ignore edges far apart
        if math.fabs(node_pos[0] - v_pos[0]) < EDGE_LEN_THRESHOLD and math.fabs(node_pos[1] - v_pos[1]) < EDGE_LEN_THRESHOLD:
            if cnt < 50 and utils.is_edge_free(maze, v_pos, node_pos):
                G.add_edge(v_node, node, weight=utils.calc_edge_len(v_pos, node_pos))
                cnt += 1

    try:
        node_path = nx.shortest_path(G, v_node, g_node)
        path = [utils.node_to_numpy(G, node) for node in node_path]
        path_len = utils.calc_path_len(path)
    except:
        print("No path found!!")
        path_len = float('inf')

    G.remove_node(v_node)

    return path_len

def analyze_env(i, res_list):
    maze = Snake8DEnv(gui=False)
    # if i % 25 == 0:
    #     draw_tree = True
    # else:
    #     draw_tree = False
    draw_tree = False

    print("Analyzing env {}".format(i))
    data_log_dir = osp.join(data_dir, "{}".format(i))
    res_log_dir = osp.join(res_dir, "{}".format(i))
    env_dir = osp.join(CUR_DIR, "../env/snake_8d/dataset/test_env/{}".format(i))
    if not osp.exists(res_log_dir):
        os.makedirs(res_log_dir)

    if not os.path.exists(osp.join(res_log_dir, "visibility.json")) or not os.path.exists(osp.join(res_log_dir, "dist_to_global_g.json")):
        with open(osp.join(env_dir, "start_goal.json")) as f:
            start_goal = json.load(f)
        G = nx.read_graphml(osp.join(env_dir, "dense_g_small.graphml.xml"))
        occ_grid = np.loadtxt(osp.join(env_dir, "occ_grid_small.txt")).astype(np.uint8)
        free_nodes = [node for node in G.nodes() if not G.nodes[node]['col']]

        maze.clear_obstacles()
        maze.load_mesh(osp.join(env_dir, "env_small.obj"))
        maze.load_occupancy_grid(occ_grid)

        visible_nodes = set()
        analyzed_nodes = set()
        visibility = []
        dist_to_global_g = []
        ext_num = 0
        while ext_num <= 300:
            print("loading tree nodes at step {}".format(ext_num))
            try:
                tree = nx.read_graphml(osp.join(data_log_dir, "tree_nodes_{}.graphml".format(ext_num)))
                tree_nodes = tree.nodes()
                tree_nodes = [tuple(utils.string_to_numpy(x[1:-1])) for x in tree_nodes]
            except:
                ext_num += 1
                continue

            # visualize
            if draw_tree:
                if ext_num % 10 == 0:
                    print("saving tree at ext_num {}".format(ext_num))
                    if os.path.exists(osp.join(res_log_dir, "tree_nodes_{}.png".format(ext_num))):
                        print("skip ", osp.join(res_log_dir, "tree_nodes_{}.png".format(ext_num)))
                        ext_num += 1
                    else:
                        tree_nodes_base = [list(node)[:2] for node in tree_nodes]
                        # print(tree_nodes, tree_nodes_base)
                        utils.visualize_nodes_global(occ_grid, tree_nodes_base, start_goal[0], start_goal[1], show=False, save=True, file_name=osp.join(res_log_dir, "tree_nodes_{}.png".format(ext_num)))

            # exploration and exploitation
            best_dist_to_goal = float('inf') if len(dist_to_global_g) == 0 else dist_to_global_g[-1]
            print("calculating visibility and distance to goal", len(visible_nodes), best_dist_to_goal, len(analyzed_nodes))
            for node_pos in tree_nodes:
                if node_pos not in analyzed_nodes:
                    # visibility
                    for G_node in free_nodes:
                        if G_node not in visible_nodes:
                            G_node_pos = utils.node_to_numpy(G, G_node)
                            if utils.is_edge_free(maze, node_pos, G_node_pos):
                                visible_nodes.add(G_node)

                    # distance
                    if best_dist_to_goal > 1e-4:
                        dist_to_goal = get_dist_to_g(maze, G, node_pos, start_goal[1])
                        if dist_to_goal < best_dist_to_goal:
                            best_dist_to_goal = dist_to_goal

                    analyzed_nodes.add(node_pos)

            print(best_dist_to_goal)
            print(len(visible_nodes))
            visibility.append(len(visible_nodes))
            dist_to_global_g.append(best_dist_to_goal)

            ext_num += 1

        with open(osp.join(res_log_dir, "visibility.json"), "w") as f:
            json.dump(visibility, f)
        with open(osp.join(res_log_dir, "dist_to_global_g.json"), "w") as f:
            json.dump(dist_to_global_g, f)

        plt.figure()
        plt.plot(np.arange(len(visibility)), np.array(visibility), 'o-')
        plt.savefig(osp.join(res_log_dir, "visibility.png"))
        plt.close()

        plt.figure()
        plt.plot(np.arange(len(dist_to_global_g)), np.array(dist_to_global_g), 'o-')
        plt.savefig(osp.join(res_log_dir, "dist_to_global_goal.png"))
        plt.close()

    else:
        print("Directly loading data for env {}".format(i))
        with open(osp.join(res_log_dir, "visibility.json"), "r") as f:
            visibility = json.load(f)
        with open(osp.join(res_log_dir, "dist_to_global_g.json"), "r") as f:
            dist_to_global_g = json.load(f)

    tree_visibility_growth = [(visibility[x] - visibility[x - 1]) for x in range(1, len(visibility))]
    mean_tree_visibility_growth = np.mean(np.array(tree_visibility_growth))
    print(mean_tree_visibility_growth)

    dist_to_global_g_shortened = [(dist_to_global_g[x - 1] - dist_to_global_g[x]) for x in range(1, len(dist_to_global_g))]
    mean_dist_to_global_g_shortened = np.mean(np.array(dist_to_global_g_shortened))
    print(mean_dist_to_global_g_shortened)

    res_list.append([mean_tree_visibility_growth, mean_dist_to_global_g_shortened])

manager = mp.Manager()
res_list = manager.list()
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--planner', default='neural_d_no_col_1')
args = parser.parse_args()

data_dir = osp.join(CUR_DIR, "eval_res/snake_8d/test_ext_500/{}".format(args.planner))
print(data_dir)
res_dir = osp.join(CUR_DIR, "eval_res/snake_8d/tree_viz/{}".format(args.planner))
print(res_dir)

env_num = 250
if not LOAD:
    process_num = 125
    j = 0
    while j < env_num:
        processes = []
        print("Running on env {} to {}".format(j, min(env_num, j + process_num)))
        for i in range(j, min(env_num, j + process_num)):
            p = mp.Process(target=analyze_env, args=(i, res_list), daemon=True)
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        j += process_num
else:
    for i in range(env_num):
        res_env_dir = osp.join(res_dir, "{}".format(i))
        with open(osp.join(res_env_dir, "visibility.json"), "r") as f:
            visibility = json.load(f)
        with open(osp.join(res_env_dir, "dist_to_global_g.json"), "r") as f:
            dist_to_global_g = json.load(f)

        for end_idx, dist in enumerate(dist_to_global_g):
            if dist < 1e-4:
                break

        dist_to_global_g = dist_to_global_g[:end_idx]
        visibility = visibility[:end_idx]

        tree_visibility_growth = np.array([(visibility[x] - visibility[x - 1]) for x in range(1, len(visibility))])
        mean_tree_visibility_growth = np.mean(tree_visibility_growth)
        print(mean_tree_visibility_growth)
        dist_to_global_g_shortened = np.array([(dist_to_global_g[x - 1] - dist_to_global_g[x]) for x in range(1, len(dist_to_global_g))])
        mean_dist_to_global_g_shortened = np.mean(dist_to_global_g_shortened)
        print(mean_dist_to_global_g_shortened)
        res_list.append([mean_tree_visibility_growth, mean_dist_to_global_g_shortened])

avg_tree_visibility_growth = np.array([res[0] for res in res_list])
avg_dist_to_global_g_shortened = np.array([res[1] for res in res_list])
avg_tree_visibility_growth = avg_tree_visibility_growth[~np.isnan(avg_tree_visibility_growth)]
avg_dist_to_global_g_shortened = avg_dist_to_global_g_shortened[~np.isnan(avg_dist_to_global_g_shortened)]
print("visibility: ", np.mean(avg_tree_visibility_growth))
print("dist_to_global_g_shortened: ", np.mean(avg_dist_to_global_g_shortened))