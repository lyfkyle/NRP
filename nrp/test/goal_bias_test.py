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
from planner.neural_planner_d import NeuralPlanner
from planner.neural_planner_g import CVAENeuralPlanner

CUR_DIR = osp.dirname(osp.abspath(__file__))
LOG_TREE = False

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--name',  default='goal_bias_test')
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

# Base planner
base_planner = RRTPlanner()

# Neural extension planner
# Hyperparameters:
neural_num_sampels = 1250
neural_goal_bias = 0.4
neural_num_ext = 1
col_checker_path = osp.join(CUR_DIR, "../local_sampler_d/models/model_col_dagger_small.pt")
selector_path = osp.join(CUR_DIR, "../local_sampler_d/models/model_sel_small_v2.pt")
# selector_path = osp.join(CUR_DIR, "../local_sampler_d/models/model_sel.pt")
planner = NeuralPlanner(col_checker_path, selector_path)
planner.max_num_recur = neural_num_ext
planner.algo.goal_bias = neural_goal_bias
planner.sampler.num_of_samples = neural_num_sampels
planner.sampler.use_col_checker = True
planner.sampler.use_selector = True

# CVAE Planner
model_path = osp.join(CUR_DIR, "../local_sampler_g/models/cvae_sel.pt")
cvae_planner = CVAENeuralPlanner(model_path)

max_goal_bias = 90
goal_bias_step_size = 10

base_success_list = [[0 for _ in range(15)] for _ in range(max_goal_bias // goal_bias_step_size)]
success_list = [[0 for _ in range(15)] for _ in range(max_goal_bias // goal_bias_step_size)]
print(success_list)
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

        if args.planner == 'base':
            base_log_dir = osp.join(planner_res_dir, "{}".format(i * plan_num + j))
            if not osp.exists(base_log_dir):
                os.mkdir(base_log_dir)

            # Base planner
            if LOG_TREE:
                base_planner.algo.log_dir = base_log_dir # log tree info
                base_planner.algo.return_on_path_find = False
                base_planner.log_dir = base_log_dir

            res = []
            for goal_bias in range(10, max_goal_bias + 1, goal_bias_step_size):
                base_planner.algo.goal_bias = float(goal_bias) / 100
                if args.type == "ext":
                    r = base_planner.solve_step_extension(maze, maze.start, maze.goal, 300, 25)
                elif args.type == "time":
                    r = base_planner.solve_step_time(maze, maze.start, maze.goal, 3, 0.2)
                success_res = [tmp[0] for tmp in r]
                res.append(success_res)

            # base_success_res = [tmp[0] for tmp in res]
            # base_path_list = [tmp[1] for tmp in res]

            for idx1, goal_res in enumerate(res):
                for idx2, ext_res in enumerate(goal_res):
                    if ext_res:
                        base_success_list[idx1][idx2] += 1
            print(base_success_list)

        elif args.planner.startswith('neural'):
            planner.sampler.set_robot_bounds(low_bounds, high_bounds)
            learnt_log_dir = osp.join(planner_res_dir, "{}".format(i * plan_num + j))
            if not osp.exists(learnt_log_dir):
                os.mkdir(learnt_log_dir)

            if LOG_TREE:
                planner.log_extension_info = True
                planner.log_dir = learnt_log_dir
                planner.algo.return_on_path_find = False
                planner.algo.log_dir = learnt_log_dir

            res = []
            for goal_bias in range(10, max_goal_bias + 1, goal_bias_step_size):
                planner.algo.goal_bias = float(goal_bias) / 100
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

        elif args.planner.startswith('local_sampler_g'):
            cvae_planner.sampler.set_robot_bounds(low_bounds, high_bounds)
            learnt_log_dir = osp.join(planner_res_dir, "{}".format(i * plan_num + j))
            if not osp.exists(learnt_log_dir):
                os.mkdir(learnt_log_dir)

            if LOG_TREE:
                cvae_planner.log_extension_info = True
                cvae_planner.log_dir = learnt_log_dir
                cvae_planner.algo.return_on_path_find = False
                cvae_planner.algo.log_dir = learnt_log_dir

            res = []
            for goal_bias in range(10, max_goal_bias + 1, goal_bias_step_size):
                cvae_planner.algo.goal_bias = float(goal_bias) / 100
                if args.type == "ext":
                    r = cvae_planner.solve_step_extension(maze, maze.start, maze.goal, 300, 25)
                elif args.type == "time":
                    r = cvae_planner.solve_step_time(maze, maze.start, maze.goal, 3, 0.2)
                success_res = [tmp[0] for tmp in r]
                res.append(success_res)

            for idx1 in range(len(res)):
                for idx2 in range(len(res[idx1])):
                    if res[idx1][idx2]:
                        success_list[idx1][idx2] += 1
            print(success_list)

print("base_success_list", base_success_list)
print("success_list", success_list)

# A = np.array(success_list)
# A = np.argsort(A.T)


res = {"success_list": success_list, "base_success_list": base_success_list}
with open(osp.join(planner_res_dir, "result.json"), "w") as f:
    json.dump(res, f)