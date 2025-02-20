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

data_dir = os.path.join(CUR_DIR, "dataset/gibson/mytest")

env_dirs = []
for path in Path(data_dir).rglob('env_small.obj'):
    env_dirs.append(path.parent)

output_dir = os.path.join(CUR_DIR, "dataset/gibson_viz/test")

dense_num = 10000
def process_env(env_idx):
    print("Process for env {}".format(env_idx))
    maze = Maze(gui=False)

    start_time = time.time()
    env_dir = env_dirs[env_idx]
    print("generating env:{}".format(env_dir))

    # env
    maze.clear_obstacles()
    occ_grid = np.loadtxt(osp.join(env_dir, "occ_grid_small.txt")).astype(np.uint8)
    print(occ_grid.shape)

    maze.load_mesh(osp.join(env_dir, "env_small.obj"))
    maze.load_occupancy_grid(occ_grid)

    env_name = str(env_dir).split("/")[-1]
    utils.visualize_nodes_global(occ_grid, [], None, None, show=False, save=True, file_name=osp.join(output_dir, f"{env_name}_env_small.png"))


# split into processes
env_num = len(env_dirs)
print(env_num)
process_num = env_num
j = 0
while j < env_num:
    processes = []
    for i in range(j, min(env_num, j + process_num)):
        p = Process(target=process_env, args=(i,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    j += process_num



