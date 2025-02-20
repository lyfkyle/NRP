import os
import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))

import numpy as np
import argparse
import torch
import datetime
import json

from env.maze import Maze
from rrt_planner import RRTPlanner
from neural_planner import NeuralPlanner
from wbmp8dof.planner.neural_planner_g import CVAENeuralPlanner

CUR_DIR = osp.dirname(osp.abspath(__file__))
LOG_TREE = False

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--name',  default='sl_bias_test')
parser.add_argument('--planner', default='base')
parser.add_argument('--type', default='ext')
args = parser.parse_args()

now = datetime.datetime.now()
date_time = now.strftime("%m-%d-%Y-%H-%M-%S")
res_dir = osp.join(CUR_DIR, "eval_res/{}_{}".format(args.name, args.type))
planner_res_dir = osp.join(res_dir, args.planner)
if not osp.exists(res_dir):
    os.mkdir(res_dir)
if not osp.exists(planner_res_dir):
    os.mkdir(planner_res_dir)

env_num = 250
plan_num = 1
dim = 8
occ_grid_dim = 40

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

maze = Maze(gui=False)

# Neural extension planner
# Hyperparameters:
if args.type == "ext":
    neural_goal_bias = 0.4
elif args.type == "time":
    neural_goal_bias = 0.5
col_checker_path = osp.join(CUR_DIR, "../local_sampler_d/models/model_col_dagger_small.pt")
selector_path = osp.join(CUR_DIR, "../local_sampler_d/models/model_sel_small_v2.pt")
if args.planner.startswith('neural_d'):
    planner = NeuralPlanner(col_checker_path, selector_path)
    planner.algo.goal_bias = neural_goal_bias

# CVAE Planner
if args.type == "ext":
    neural_goal_bias = 0.4
elif args.type == "time":
    neural_goal_bias = 0.4
model_path = osp.join(CUR_DIR, "../local_sampler_g/models/cvae_sel.pt")
if args.planner.startswith('neural_g'):
    planner = CVAENeuralPlanner(model_path)
    planner.algo.goal_bias = neural_goal_bias

max_sl_bias = 100
sl_bias_step_size = 10

base_success_list = [[0 for _ in range(15)] for _ in range(max_sl_bias // sl_bias_step_size + 1)]
success_list = [[0 for _ in range(15)] for _ in range(max_sl_bias // sl_bias_step_size + 1)]
total_extend_time = 0
total_num_extend_called = 0
total_neural_extend_time = 0
total_col_check_time = 0
neural_select_success_rate = 0
extend_success_rate = 0
extend_dist_intended = 0
extend_dist_actual = 0
extend_dist_intended_base = 0
extend_dist_actual_base = 0
extend_col_prop = 0
for i in range(env_num):
    maze.clear_obstacles()
    maze_dir = osp.join(CUR_DIR, "../dataset/test_env/{}".format(i))

    occ_grid = np.loadtxt(osp.join(maze_dir, "occ_grid_small.txt")).astype(np.uint8)
    mesh_path = osp.join(maze_dir, "env_small.obj")
    maze.load_mesh(osp.join(maze_dir, "env_small.obj"))
    maze.load_occupancy_grid(occ_grid, add_enclosing=True)

    for j in range(plan_num):
        print("Loading env {}".format(i * plan_num + j))
        env_dir = osp.join(CUR_DIR, "../dataset/test_env/{}".format(i * plan_num + j))
        # env_dir = osp.join(CUR_DIR, "../eval_env/{}".format(7)) # `18`
        # with open(osp.join(env_dir, "obstacle_dict.json")) as f:
        #     obstacle_dict = json.load(f)

        # maze.default_obstacles(mode=Maze2D.BOX_ONLY)
        occ_grid = maze.get_occupancy_grid()

        with open(osp.join(env_dir, "start_goal.json")) as f:
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
        print(low_bounds, high_bounds)
        planner.sampler.set_robot_bounds(low_bounds, high_bounds)

        if args.planner == 'base':
            pass
        elif args.planner.startswith('neural_d'):
            learnt_log_dir = osp.join(planner_res_dir, "{}".format(i * plan_num + j))
            if not osp.exists(learnt_log_dir):
                os.mkdir(learnt_log_dir)

            if LOG_TREE:
                planner.log_extension_info = True
                planner.log_dir = learnt_log_dir
                planner.algo.return_on_path_find = False
                planner.algo.log_dir = learnt_log_dir

            res = []
            for sl_bias in range(0, max_sl_bias + 1, sl_bias_step_size):
                planner.sl_bias = float(sl_bias) / 100
                if args.type == "ext":
                    r = planner.solve_step_extension(maze, maze.start, maze.goal, 300, 25)
                elif args.type == "time":
                    r = planner.solve_step_time(maze, maze.start, maze.goal, 3, 0.2)
                success_res = [tmp[0] for tmp in r]
                res.append(success_res)

            for idx1 in range(len(res)):
                for idx2 in range(len(res[idx1])):
                    if res[idx1][idx2]:
                        success_list[idx1][idx2] += 1
            print(success_list)

        elif args.planner.startswith('neural_g'):
            learnt_log_dir = osp.join(planner_res_dir, "{}".format(i * plan_num + j))
            if not osp.exists(learnt_log_dir):
                os.mkdir(learnt_log_dir)

            if LOG_TREE:
                planner.log_extension_info = True
                planner.log_dir = learnt_log_dir
                planner.algo.return_on_path_find = False
                planner.algo.log_dir = learnt_log_dir

            res = []
            for sl_bias in range(0, max_sl_bias + 1, sl_bias_step_size):
                planner.sl_bias = float(sl_bias) / 100
                if args.type == "ext":
                    r = planner.solve_step_extension(maze, maze.start, maze.goal, 300, 25)
                elif args.type == "time":
                    r = planner.solve_step_time(maze, maze.start, maze.goal, 3, 0.2)
                success_res = [tmp[0] for tmp in r]
                res.append(success_res)

            for idx1 in range(len(res)):
                for idx2 in range(len(res[idx1])):
                    if res[idx1][idx2]:
                        success_list[idx1][idx2] += 1
            print(success_list)

print("base_success_list", base_success_list)
print("success_list", success_list)

res = {"success_list": success_list}
with open(osp.join(planner_res_dir, "result.json"), "w") as f:
    json.dump(res, f)