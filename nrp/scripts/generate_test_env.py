import os
import os.path as osp
import numpy as np
import json
from pathlib import Path
import shutil
import networkx as nx
import random
import argparse

from nrp.env.fetch_11d.maze import Fetch11DEnv
from nrp.env.snake_8d.maze import Snake8DEnv
from nrp import ROOT_DIR


def sample_problems(G):
    # path = dict(nx.all_pairs_shortest_path(G))
    free_nodes = [n for n in G.nodes() if not G.nodes[n]["col"]]

    max_trial = 100
    i = 0
    while i < max_trial:
        s_name = random.choice(free_nodes)
        start_pos = utils.node_to_numpy(G, s_name).tolist()

        g_name = random.choice(free_nodes)
        goal_pos = utils.node_to_numpy(G, g_name).tolist()

        try:
            node_path = nx.shortest_path(G, source=s_name, target=g_name)
        except:
            continue

        path = [utils.node_to_numpy(G, n).tolist() for n in node_path]
        # for x in p:
        #     x[0] += 2
        #     x[1] += 2

        if len(path) > 4 or utils.calc_path_len_base(path) > 10:
            break

        i += 1

    return s_name, g_name, path


CUR_DIR = osp.dirname(osp.abspath(__file__))

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("--name", default="")
parser.add_argument("--env", default="snake_8d")
args = parser.parse_args()

if args.env == "snake_8d":
    env = Snake8DEnv(gui=False)
    from nrp.env.snake_8d import utils

elif args.env == "fetch_11d":
    env = Fetch11DEnv(gui=False)
    from nrp.env.fetch_11d import utils

env_num = 5
plan_num = 50

data_dir = os.path.join(CUR_DIR, "dataset/gibson/test")
env_dirs = utils.TEST_ENV_DIRS
output_dir = osp.join(ROOT_DIR, f"env/{args.env}/dataset/test_env_01")

for i in range(env_num):
    env_dir = env_dirs[i]
    print("generating test problem from {}".format(env_dir))

    # occ_grid = np.loadtxt(osp.join(maze_dir, "occ_grid_small.txt")).astype(np.uint8)
    # G = nx.read_graphml(osp.join(maze_dir, "dense_g_small.graphml"))
    occ_grid = utils.get_occ_grid(env_dir)
    G = utils.get_prm(env_dir)
    mesh_path = utils.get_mesh_path(env_dir)

    env.clear_obstacles()
    env.load_mesh(mesh_path)
    env.load_occupancy_grid(occ_grid)

    for j in range(plan_num):
        print("Generating test env {}".format(i * plan_num + j))

        output_env_dir = osp.join(output_dir, "{}".format(i * plan_num + j))
        if not osp.exists(output_env_dir):
            os.makedirs(output_env_dir)

        s_node, g_node, expert_path = sample_problems(G)
        start_pos = utils.node_to_numpy(G, s_node).tolist()
        goal_pos = utils.node_to_numpy(G, g_node).tolist()

        if args.env == "snake_8d":
            shutil.copy(osp.join(env_dir, "occ_grid.txt"), os.path.join(output_env_dir, "occ_grid.txt"))
        elif args.env == "fetch_11d":
            shutil.copy(osp.join(env_dir, "occ_grid.npy"), os.path.join(output_env_dir, "occ_grid.npy"))
        shutil.copy(osp.join(env_dir, "env.obj"), os.path.join(output_env_dir, "env.obj"))
        shutil.copy(osp.join(env_dir, "dense_g.graphml"), os.path.join(output_env_dir, "dense_g.graphml"))

        start_goal = []
        start_goal.append(start_pos)
        start_goal.append(goal_pos)
        with open(osp.join(output_env_dir, "start_goal.json"), "w") as f:
            json.dump(start_goal, f)

        path_viz = utils.interpolate(expert_path, 1)
        utils.visualize_nodes_global(
            mesh_path,
            occ_grid,
            path_viz,
            start_pos,
            goal_pos,
            show=False,
            save=True,
            file_name=osp.join(output_env_dir, "problem.png"),
        )
