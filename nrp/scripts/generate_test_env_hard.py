import os
import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), "../"))

import numpy as np
import json
from pathlib import Path
import shutil
import networkx as nx
import random
import math
import argparse

from env.fetch_11d.maze import Fetch11DEnv
from env.snake_8d.maze import Snake8DEnv
from planner.rrt_planner import RRTPlanner

CUR_DIR = osp.dirname(osp.abspath(__file__))


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


parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("--env", default="snake_8d")
args = parser.parse_args()

if args.env == "snake_8d":
    from env.snake_8d import utils
    env = Snake8DEnv(gui=False)
    data_dir = os.path.join(CUR_DIR, "../env/snake_8d/dataset/gibson/mytest")
elif args.env == "fetch_11d":
    from env.fetch_11d import utils
    env = Fetch11DEnv(gui=False)
    data_dir = os.path.join(CUR_DIR, "../env/fetch_11d/dataset/gibson/mytest")

# maze = Maze(gui=False)

env_num = 5
plan_num = 20


maze_dirs = []
for path in Path(data_dir).rglob("env.obj"):
    maze_dirs.append(path.parent)

rrt_planner = RRTPlanner(optimal=False)
rrt_planner.algo.goal_bias = 0.3


for i in range(env_num):
    maze_dir = maze_dirs[i]
    print("generating test problem from {}".format(maze_dir))

    with open(os.path.join(maze_dir, "occ_grid.npy"), "rb") as f:
        occ_grid = np.load(f)
    print(occ_grid.shape)

    G = nx.read_graphml(osp.join(maze_dir, "dense_g.graphml"))

    env.clear_obstacles()
    mesh_path = osp.join(maze_dir, "env.obj")
    env.load_mesh(mesh_path)
    env.load_occupancy_grid(occ_grid)

    # for j in range(plan_num):
    j = 0
    while j < plan_num:
        print("Generating test env {}".format(i * plan_num + j))

        env_dir = osp.join(maze_dir, "../../../test_env_hard/{}".format(i * plan_num + j))
        # if osp.exists(env_dir) and osp.exists(osp.join(env_dir, "start_goal.json")) and osp.exists(osp.join(env_dir, "problem.png")):
        #     continue
        if not osp.exists(env_dir):
            os.makedirs(env_dir)

        # s_node, g_node, expert_path = sample_problems(G)
        # start_pos = utils.node_to_numpy(G, s_node).tolist()
        # goal_pos = utils.node_to_numpy(G, g_node).tolist()
        env.sample_start_goal()
        res = rrt_planner.solve_step_time(env, env.start, env.goal, max_time=20, step_size=10, mesh=mesh_path)
        # If solution can't be found in 10 sec but can be found in 20 sec.
        if res[0][0] or not res[1][0]:
            continue

        with open(os.path.join(env_dir, "occ_grid.npy"), "wb") as f:
            np.save(f, occ_grid)
        shutil.copy(osp.join(maze_dir, "env.obj"), os.path.join(env_dir, "env.obj"))
        shutil.copy(osp.join(maze_dir, "dense_g.graphml"), os.path.join(env_dir, "dense_g.graphml"))

        start_goal = []
        start_goal.append(env.start)
        start_goal.append(env.goal)
        with open(osp.join(env_dir, "start_goal.json"), "w") as f:
            json.dump(start_goal, f)

        expert_path = res[1][1]
        path_viz = utils.interpolate(expert_path, 1)
        utils.visualize_nodes_global(
            mesh_path,
            occ_grid,
            path_viz,
            np.array(env.start),
            np.array(env.goal),
            show=False,
            save=True,
            file_name=osp.join(env_dir, "problem.png"),
        )

        j += 1
