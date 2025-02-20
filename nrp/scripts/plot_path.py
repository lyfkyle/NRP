import os
import os.path as osp
import json
import numpy as np
import argparse
import networkx as nx
from env.maze import Maze
from pathlib import Path

import utils

CUR_DIR = osp.dirname(osp.abspath(__file__))

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--name',  default='')
parser.add_argument('--planner', default='rrt_star')
args = parser.parse_args()


res_dir = osp.join(CUR_DIR, "planner/eval_res/{}/{}".format(args.name, args.planner))
viz_res_dir = osp.join(CUR_DIR, "planner/eval_res/qualitative/viz_path/{}".format(args.planner))
if not os.path.exists(viz_res_dir):
    os.makedirs(viz_res_dir)

env_num = 250
print(res_dir)
maze = Maze(gui=False)
# for env_idx in range(0, env_num, 25):
for _ in range(1):
    env_idx = 0
    print("Analyzing env {}".format(env_idx))
    log_dir = osp.join(res_dir, "{}".format(env_idx))
    env_dir = osp.join(CUR_DIR, "dataset/test_env/{}".format(env_idx))
    with open(osp.join(env_dir, "start_goal.json")) as f:
        start_goal = json.load(f)

    occ_grid = np.loadtxt(osp.join(env_dir, "occ_grid_small.txt")).astype(np.uint8)
    mesh_path = osp.join(env_dir, "env_small.obj")

    maze.clear_obstacles()
    maze.load_mesh(mesh_path)
    maze.load_occupancy_grid(occ_grid)

    plan_success = osp.isfile(osp.join(log_dir, "planned_path.json"))
    if not plan_success:
        idx = 0
        file_path = os.path.join(log_dir, "planned_path_{}.json".format(idx))
        while os.path.exists(file_path):
            with open(os.path.join(log_dir, "planned_path_{}.json".format(idx)), "r") as f:
                planned_path = json.load(f)

            if len(planned_path) > 1:
                # print(planned_path)
                interpolated_path = utils.interpolate(planned_path)
                utils.visualize_nodes_global(occ_grid, interpolated_path, start_goal[0], start_goal[1], show=False, save=True, file_name=osp.join(viz_res_dir, "planned_path_{}_{}.png".format(env_idx, idx)))

            idx += 1
            file_path = os.path.join(log_dir, "planned_path_{}.json".format(idx))

    else:
        with open(os.path.join(log_dir, "planned_path.json"), "r") as f:
            planned_path = json.load(f)

        interpolated_path = utils.interpolate(planned_path)
        utils.visualize_nodes_global(occ_grid, planned_path, start_goal[0], start_goal[1], show=False, save=True, file_name=osp.join(viz_res_dir, "planned_path_{}.png".format(env_idx)))