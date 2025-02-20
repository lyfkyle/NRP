import os
import os.path as osp
import sys
# sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))
# print(sys.path)

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import shutil
import json
from PIL import Image
import itertools
import random
from pathlib import Path
import time
import math
from multiprocessing import Process

from env.maze import Maze
import utils

CUR_DIR = osp.dirname(osp.abspath(__file__))

def state_to_numpy(state):
    strlist = state.split(',')
    val_list = [float(s) for s in strlist]
    return np.array(val_list)

# maze = Maze2D(gui=False)

data_dir = os.path.join(CUR_DIR, "dataset/gibson")

env_dirs = []
for path in Path(data_dir).rglob('env_small.obj'):
    env_dirs.append(path.parent)


maze = Maze(gui=False)
for env_idx in range(5):
    start_time = time.time()
    env_dir = env_dirs[env_idx]
    print("generating env:{}".format(env_dir))

    # env
    maze.clear_obstacles()
    occ_grid = np.loadtxt(osp.join(env_dir, "occ_grid_small.txt")).astype(np.uint8)
    print(occ_grid.shape)

    maze.load_mesh(osp.join(env_dir, "env_small.obj"))
    maze.load_occupancy_grid(occ_grid)

    low = maze.robot.get_joint_lower_bounds()
    high = maze.robot.get_joint_higher_bounds()

    random_state = [0] * maze.robot.num_dim
    random_state[0] = 5
    random_state[1] = 5
    while True:
        if maze.pb_ompl_interface.is_state_valid(random_state):
            break
        for i in range(2):
            random_state[i] = random.uniform(4, 6)

    # utils.visualize_nodes_global(occ_grid, [], None, None, show=False, save=True, file_name=osp.join(env_dir, "env.png"))
    utils.visualize_nodes_global(occ_grid, [random_state], None, None, show=False, save=True, file_name=osp.join(env_dir, "env_with_robot.png"))
