import os
import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))

import numpy as np
import argparse
import matplotlib.pyplot as plt
import math
import random
import torch
import datetime
import time
import json

from env.maze import Maze
import utils
# from planners.cvae_planner_curve_v1_orig import CVAEPlannerCurveV1
# from cvae_planner_curve_v4 import CVAEPlannerCurveV4
from rrt_planner import RRTPlanner
from neural_planner import NeuralPlanner
# from lazy_neural_planner import LazyNeuralPlanner

CUR_DIR = osp.dirname(osp.abspath(__file__))

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--name',  default='qualitative')
parser.add_argument('--planner', default='base')
args = parser.parse_args()

now = datetime.datetime.now()
if args.name == '':
    date_time = now.strftime("%m-%d-%Y-%H-%M-%S")
    res_dir = osp.join(CUR_DIR, "eval_res/{}".format(date_time))
else:
    res_dir = osp.join(CUR_DIR, "eval_res/{}".format(args.name))
planner_res_dir = osp.join(res_dir, args.planner)
if not osp.exists(res_dir):
    os.mkdir(res_dir)
if not osp.exists(planner_res_dir):
    os.mkdir(planner_res_dir)

env_num = 219
dim = 11
occ_grid_dim = 40
occ_grid_dim_z = 20

# Hyperparameters:
neural_num_sampels = 1250
neural_goal_bias = 0.5
col_pred_threshold = 0.6
neural_num_ext = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

maze = Maze(gui=False)

# col_checker_path = osp.join(CUR_DIR, "../local_sampler_d/models/model_col_point_selfcol_linkinfo.pt")
col_checker_path = osp.join(CUR_DIR, "../local_sampler_d/models/model_col_edge_large_v7.pt")
selector_path = osp.join(CUR_DIR, "../local_sampler_d/models/model_sel_bce_bal.pt")

base_planner = RRTPlanner()
planner = NeuralPlanner(col_checker_path, selector_path)
# lazy_planner = LazyNeuralPlanner(col_checker_path, selector_path)
planner.algo.goal_bias = neural_goal_bias
planner.max_num_recur = neural_num_ext
planner.sampler.col_pred_threshold = col_pred_threshold
planner.sampler.num_of_samples = neural_num_sampels
planner.sampler.visualize = True
planner.sampler.use_col_checker = True
planner.sampler.use_selector = True

max_extension = 100
ext_step_size = 100

maze.clear_obstacles()
maze_dir = osp.join(CUR_DIR, "../dataset/test_env/{}".format(env_num))
with open(os.path.join(maze_dir, "occ_grid_large.npy"), 'rb') as f:
    occ_grid = np.load(f)
print(occ_grid.shape)

mesh_path = osp.join(maze_dir, "env_large.obj")
maze.load_mesh(mesh_path)
maze.load_occupancy_grid(occ_grid, add_enclosing=True)

print("Loading env {}".format(env_num))
# env_dir = osp.join(CUR_DIR, "../dataset/test_env/{}".format(i * plan_num + j))
# env_dir = osp.join(CUR_DIR, "../dataset/test_env/{}".format(env_num)) # `18`
# with open(osp.join(env_dir, "obstacle_dict.json")) as f:
#     obstacle_dict = json.load(f)

# maze.default_obstacles(mode=Maze2D.BOX_ONLY)
# occ_grid_2 = maze.get_occupancy_grid()
# assert np.allclose(occ_grid_2, occ_grid)

with open(osp.join(maze_dir, "start_goal.json")) as f:
    start_goal = json.load(f)

# start_goal = json.load(osp.join(env_dir, "start_goal.json"))
maze.start = start_goal[0]
maze.goal = start_goal[1]
maze.robot.set_state(maze.start)

# utils.visualize_nodes(occ_grid_small, [], None, None, maze.start, maze.goal, show=False, save=True, file_name=osp.join(log_dir, "problem.png"))

# Get robot bounds
low_bounds = maze.robot.get_joint_lower_bounds()
high_bounds = maze.robot.get_joint_higher_bounds()
low_bounds[0] = -2
low_bounds[1] = -2
high_bounds[0] = 2
high_bounds[1] = 2
# print(low_bounds, high_bounds)
planner.sampler.set_robot_bounds(low_bounds, high_bounds)

if args.planner == 'base':
    base_log_dir = osp.join(planner_res_dir, "{}".format(env_num))
    if not osp.exists(base_log_dir):
        os.mkdir(base_log_dir)

    # Base planner
    base_planner.algo.return_on_path_find = True
    res = base_planner.solve_step_extension(maze, maze.start, maze.goal, max_extension, ext_step_size)
    base_success_res = [tmp[0] for tmp in res]
    base_path_list = [tmp[1] for tmp in res]
    for p in base_path_list:
        if len(p) > 0:
            path = utils.interpolate(p)
            utils.visualize_nodes_global(mesh_path, occ_grid, path, maze.start, maze.goal, show=False, save=True, file_name=osp.join(base_log_dir, "planned_path.png"))

elif args.planner.startswith('neural'):
    learnt_log_dir = osp.join(planner_res_dir, "{}".format(env_num))
    if not osp.exists(learnt_log_dir):
        os.mkdir(learnt_log_dir)

    planner.sampler.log_dir = learnt_log_dir
    planner.log_extension_info = False
    planner.algo.return_on_path_find =True
    res = planner.solve_step_extension(maze, maze.start, maze.goal, max_extension, ext_step_size)
    success_res = [tmp[0] for tmp in res]
    path_list = [tmp[1] for tmp in res]
    for p in path_list:
        if len(p) > 0:
            path = utils.interpolate(p)
            utils.visualize_nodes_global(mesh_path, occ_grid, path, maze.start, maze.goal, show=False, save=True, file_name=osp.join(learnt_log_dir, "planned_path.png"))


