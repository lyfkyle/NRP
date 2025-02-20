import os
import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))
import numpy as np
import json

from env.maze import Maze
import utils

CUR_DIR = osp.dirname(osp.abspath(__file__))

maze = Maze(gui=False)

env_num = 10
plan_num = 1

maze.clear_obstacles()
with open(os.path.join(CUR_DIR, "map/occ_grid.npy"), 'rb') as f:
    occ_grid = np.load(f)
mesh_path = osp.join(CUR_DIR, "map/rls_fixed.obj")

maze.load_mesh(mesh_path)
maze.load_occupancy_grid(occ_grid, add_enclosing=True)

for i in range(env_num):
    for j in range(plan_num):
        env_idx = i * plan_num + j
        print("Loading env {}".format(env_idx))
        # env_dir = osp.join(CUR_DIR, "../eval_env/{}".format(7)) # `18`
        # with open(osp.join(env_dir, "obstacle_dict.json")) as f:
        #     obstacle_dict = json.load(f)

        occ_grid = maze.get_occupancy_grid()
        with open(osp.join(CUR_DIR, "test_path/test_path_{}.json".format(env_idx)), "r") as f:
            start_goal = json.load(f)

        utils.visualize_nodes_global(mesh_path, occ_grid, [], start_goal[0], start_goal[1], show=False, save=True, file_name=osp.join(CUR_DIR, "test_path/test_path_{}.png".format(env_idx)))
