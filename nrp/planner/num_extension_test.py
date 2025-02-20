import os
import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))

import numpy as np
import argparse
import torch
import datetime
import json
import yaml

from env.maze import Maze
from planner.rrt_planner import RRTPlanner
from planner.neural_planner_d import NRP_d
from planner.neural_planner_g import NRP_g

CUR_DIR = osp.dirname(osp.abspath(__file__))
LOG_TREE = False

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--name',  default='num_extension_test')
parser.add_argument('--planner', default='neural_d')
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
dim = 11
occ_grid_dim = 40

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

maze = Maze(gui=False)

# Hyperparameters:
neural_goal_bias = 0.4
sl_bias = 0.01
col_checker_path = osp.join(CUR_DIR, "../local_sampler_d/models/model_col_final.pt")
selector_path = osp.join(CUR_DIR, "../local_sampler_d/models/model_sel_final.pt")
neural_planner_d = NRP_d(col_checker_path, selector_path)
neural_planner_d.algo.goal_bias = neural_goal_bias
neural_planner_d.algo.add_intermediate_state = False
neural_planner_d.sl_bias = sl_bias

success_list = [[0 for _ in range(15)] for _ in range(5)]
total_neural_extend_time = [0] * 5
total_col_check_time = [0] * 5
total_extend_time = [0] * 5

for i in range(env_num):
    maze.clear_obstacles()
    maze_dir = osp.join(CUR_DIR, "../dataset/test_env/{}".format(i))
    occ_grid = np.loadtxt(osp.join(maze_dir, "occ_grid_small.txt")).astype(np.uint8)
    mesh_path = osp.join(maze_dir, "env_small.obj")

    maze.load_mesh(mesh_path)
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
        neural_planner_d.sampler.set_robot_bounds(low_bounds, high_bounds)

        if args.planner == 'base':
            pass

        else:
            learnt_log_dir = osp.join(planner_res_dir, "{}".format(i * plan_num + j))
            if not osp.exists(learnt_log_dir):
                os.mkdir(learnt_log_dir)

            if LOG_TREE:
                neural_planner_d.log_extension_info = True
                neural_planner_d.log_dir = learnt_log_dir
                neural_planner_d.algo.return_on_path_find = False
                neural_planner_d.algo.log_dir = learnt_log_dir

            res = []
            for num_recur in range(1, 6):
                neural_planner_d.max_num_recur = num_recur
                if args.type == "ext":
                    r = neural_planner_d.solve_step_extension(maze, maze.start, maze.goal, 300, 25)
                elif args.type == "time":
                    r = neural_planner_d.solve_step_time(maze, maze.start, maze.goal, 3, 0.2)
                success_res = [tmp[0] for tmp in r]
                res.append(success_res)

                # if r:
                #     # path = utils.interpolate(p)
                #     # utils.visualize_nodes_global(mesh_path, occ_grid, path, start_goal[0], start_goal[1], show=False, save=True, file_name=osp.join(learnt_log_dir, "path_{}.png".format(num_recur)))
                #     with open(osp.join(learnt_log_dir, 'planned_path_{}.json'.format(num_recur)), 'w') as f:
                #         json.dump(p, f)

                total_extend_time[num_recur - 1] += neural_planner_d.algo.total_running_time
                total_neural_extend_time[num_recur - 1] += neural_planner_d.neural_expansion_time / (neural_planner_d.neural_expansion_cnt + 1e-8)
                total_col_check_time[num_recur - 1] += neural_planner_d.col_check_time / (neural_planner_d.neural_expansion_cnt + 1e-8)

            for idx1 in range(len(res)):
                for idx2 in range(len(res[idx1])):
                    if res[idx1][idx2]:
                        success_list[idx1][idx2] += 1
            print(success_list)

avg_total_extend_time = np.array(total_extend_time) / float(env_num * plan_num)
avg_neural_extend_time = np.array(total_neural_extend_time) / float(env_num * plan_num)
avg_col_check_time = np.array(total_col_check_time) / float(env_num * plan_num)
print("Average total sampling time for learnt: {}, neural: {}, col: {}".format(avg_total_extend_time, avg_neural_extend_time, avg_col_check_time))
print("success_list", success_list)

res = {"success_list": success_list}
with open(osp.join(planner_res_dir, "result.json"), "w") as f:
    json.dump(res, f)