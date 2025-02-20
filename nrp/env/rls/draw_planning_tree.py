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
import shutil

import utils
from env.maze import Maze

CUR_DIR = osp.dirname(osp.abspath(__file__))


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--planner', default='base')
args = parser.parse_args()

res_dir = osp.join(CUR_DIR, "eval_res/debug/{}".format(args.planner))
print(res_dir)
env_num = 10
maze = Maze(gui=False)
maze.clear_obstacles()
with open(os.path.join(CUR_DIR, "map/occ_grid.npy"), 'rb') as f:
    occ_grid = np.load(f)
mesh_path = osp.join(CUR_DIR, "map/rls_fixed.obj")

maze.load_mesh(mesh_path)
maze.load_occupancy_grid(occ_grid, add_enclosing=True)

for i in range(env_num):
    print("Analyzing env {}".format(i))
    log_dir = osp.join(res_dir, "{}".format(i))
    env_dir = osp.join(CUR_DIR, "../dataset/test_env/{}".format(i))

    ext_num = 0
    while ext_num <= 500:
        print("loading tree nodes at step {}".format(ext_num))

        occ_grid = maze.get_occupancy_grid()
        with open(osp.join(CUR_DIR, "test_path/test_path_{}.json".format(i)), "r") as f:
            start_goal = json.load(f)

        try:
            tree = nx.read_graphml(osp.join(log_dir, "s_tree_nodes_{}.graphml".format(ext_num)))
            tree_nodes = tree.nodes()
            tree_nodes = [tuple(utils.string_to_numpy(x[1:-1])) for x in tree_nodes]
        except:
            break

        # visualize
        if ext_num % 100 == 0:
            utils.visualize_tree(occ_grid, tree, start_goal[0], start_goal[1], show=False, save=True, file_name=osp.join(log_dir, "s_tree_nodes_{}.png".format(ext_num)), string=True)



        try:
            tree = nx.read_graphml(osp.join(log_dir, "g_tree_nodes_{}.graphml".format(ext_num)))
            tree_nodes = tree.nodes()
            tree_nodes = [tuple(utils.string_to_numpy(x[1:-1])) for x in tree_nodes]
        except:
            break

        # visualize
        if ext_num % 100 == 0:
            utils.visualize_tree(occ_grid, tree, start_goal[0], start_goal[1], show=False, save=True, file_name=osp.join(log_dir, "g_tree_nodes_{}.png".format(ext_num)), string=True)
        # input("here")

        ext_num += 1