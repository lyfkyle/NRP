# separate the results for each environment
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
from planner.bit_star import BITStar
from planner.rrt_planner import RRTPlanner
from planner.neural_planner_d import NRP_d
from planner.neural_planner_g import NRP_g

CUR_DIR = osp.dirname(osp.abspath(__file__))
LOG_TREE = False

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--name',  default='timing_analysis_v2')
parser.add_argument('--planner', default='rrt')
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

env_num = 10
env_step_size = 25
plan_num = 1
dim = 8
occ_grid_dim = 40

# Base planner
if args.planner.startswith('rrt'):
    rrt_planner = RRTPlanner()
    rrt_planner.algo.goal_bias = 0.1

# BITstar planner
if args.planner.startswith('bit'):
    bit_planner = BITStar()

# Neural extension planner
# Hyperparameters:
if args.planner.startswith('neural_d'):
    neural_goal_bias = 0.5
    sl_bias = 0.4
    col_checker_path = osp.join(CUR_DIR, "../local_sampler_d/models/model_col_final.pt")
    selector_path = osp.join(CUR_DIR, "../local_sampler_d/models/model_sel_final.pt")
    neural_planner_d = NRP_d(col_checker_path, selector_path)
    neural_planner_d.algo.goal_bias = neural_goal_bias
    neural_planner_d.sl_bias = sl_bias

# CVAE Planner
if args.planner.startswith('neural_g'):
    neural_goal_bias = 0.4
    sl_bias = 0.3
    model_path = osp.join(CUR_DIR, "../local_sampler_g/models/cvae_sel.pt")
    neural_planner_g = NRP_g(model_path)
    neural_planner_g.algo.goal_bias = neural_goal_bias
    neural_planner_g.sl_bias = sl_bias

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

maze = Maze(gui=False)

max_time = 3
time_step_size = 3

base_success_list = [0] * int(max_time / time_step_size)
success_list = [0] * int(max_time / time_step_size)
total_extend_time = 0
total_num_extend_called = 0
total_neural_extend_time = 0
total_neural_extend_success_time = 0
total_neural_extend_fail_time = 0
total_col_check_time = 0
total_col_check_success_time = 0
total_col_check_fail_time = 0
total_extend_success = 0
total_neural_select_success = 0
total_neural_select_called = 0
total_neural_extend_called = 0
total_neural_extend_success = 0
extend_success_rate = 0
total_vertex_selection_time = 0
total_vertex_selection_success_time = 0
total_vertex_selection_fail_time = 0
total_vertex_extension_time = 0
total_vertex_extension_success_time = 0
total_vertex_extension_fail_time = 0
avg_time_per_successful_ext = 0
avg_ext_per_successful_ext = 0
for env_idx in range(env_num):
    i = env_idx * env_step_size
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

        occ_grid = maze.get_occupancy_grid()

        with open(osp.join(env_dir, "start_goal.json")) as f:
            start_goal = json.load(f)
        # start_goal = json.load(osp.join(env_dir, "start_goal.json"))
        maze.start = start_goal[0]
        maze.goal = start_goal[1]
        maze.robot.set_state(maze.start)

        # Get robot bounds
        low_bounds = maze.robot.get_joint_lower_bounds()
        high_bounds = maze.robot.get_joint_higher_bounds()
        low_bounds[0] = -2
        low_bounds[1] = -2
        high_bounds[0] = 2
        high_bounds[1] = 2
        # print(low_bounds, high_bounds)

        if args.planner.startswith('rrt'):
            log_dir = osp.join(planner_res_dir, "{}".format(i * plan_num + j))
            if not osp.exists(log_dir):
                os.mkdir(log_dir)

            rrt_planner.algo.return_on_path_find = False
            res = rrt_planner.solve_step_time(maze, maze.start, maze.goal, max_time, time_step_size, mesh_path)
            base_success_res = [tmp[0] for tmp in res]
            base_path_list = [tmp[1] for tmp in res]
            for p in base_path_list:
                if len(p) > 0:
                    # path = utils.interpolate(p)
                    # utils.visualize_nodes_global(occ_grid, path, maze.start, maze.goal, show=False, save=True, file_name=osp.join(base_log_dir, "planned_path.png"))
                    with open(osp.join(log_dir, 'planned_path.json'), 'w') as f:
                        json.dump(p, f)
                    break

            for idx, res in enumerate(base_success_res):
                if res:
                    base_success_list[idx] += 1

            res = {
                "extend_time": rrt_planner.algo.total_running_time,
                "col_check_time": rrt_planner.col_check_time,
                "col_check_success_time": rrt_planner.col_check_success_time,
                "col_check_fail_time": rrt_planner.col_check_fail_time,
                "vertex_selection_time": rrt_planner.algo.total_vertex_selection_time,
                "vertex_selection_success_time": rrt_planner.algo.total_vertex_selection_success_time,
                "vertex_selection_fail_time": rrt_planner.algo.total_vertex_selection_fail_time,
                "vertex_extension_time": rrt_planner.algo.total_vertex_extension_time,
                "vertex_extension_success_time": rrt_planner.algo.total_vertex_extension_success_time,
                "vertex_extension_fail_time": rrt_planner.algo.total_vertex_extension_fail_time,
                "vertex_extension_other_time": rrt_planner.algo.total_vertex_extension_time - rrt_planner.col_check_time,
                "other_time": rrt_planner.algo.total_running_time - rrt_planner.algo.total_vertex_selection_time - rrt_planner.algo.total_vertex_extension_time,
            }

            with open(osp.join(log_dir, "result.json"), "w") as f:
                json.dump(res, f)

            # path_length = utils.calc_path_len(path)
            total_extend_time += rrt_planner.algo.total_running_time
            total_col_check_time += rrt_planner.col_check_time
            total_col_check_success_time += rrt_planner.col_check_success_time
            total_col_check_fail_time += rrt_planner.col_check_fail_time
            total_vertex_selection_time += rrt_planner.algo.total_vertex_selection_time
            total_vertex_selection_success_time += rrt_planner.algo.total_vertex_selection_success_time
            total_vertex_selection_fail_time += rrt_planner.algo.total_vertex_selection_fail_time
            total_vertex_extension_time += rrt_planner.algo.total_vertex_extension_time
            total_vertex_extension_success_time += rrt_planner.algo.total_vertex_extension_success_time
            total_vertex_extension_fail_time += rrt_planner.algo.total_vertex_extension_fail_time
            total_extend_success += rrt_planner.algo.num_extensions - rrt_planner.extension_col_cnt
            total_num_extend_called += rrt_planner.algo.num_extensions

            print(rrt_planner.algo.num_extensions, rrt_planner.algo.total_succcesful_vertex_extension)
            avg_time_per_successful_ext += rrt_planner.algo.total_running_time / rrt_planner.algo.total_succcesful_vertex_extension
            avg_ext_per_successful_ext += rrt_planner.algo.num_extensions / rrt_planner.algo.total_succcesful_vertex_extension

        elif args.planner.startswith('bit'):
            log_dir = osp.join(planner_res_dir, "{}".format(i * plan_num + j))
            if not osp.exists(log_dir):
                os.mkdir(log_dir)

            res = bit_planner.solve_step_time(maze, maze.start, maze.goal, max_time, time_step_size)
            base_success_res = [tmp[0] for tmp in res]
            base_path_list = [tmp[1] for tmp in res]
            for p in base_path_list:
                if len(p) > 0:
                    # path = utils.interpolate(p)
                    # utils.visualize_nodes_global(mesh_path, occ_grid, path, maze.start, maze.goal, show=False, save=True, file_name=osp.join(base_log_dir, "planned_path.png"))
                    with open(osp.join(log_dir, 'planned_path.json'), 'w') as f:
                        json.dump(p, f)
                    break

            for idx, res in enumerate(base_success_res):
                if res:
                    base_success_list[idx] += 1

            res = {
                "extend_time": bit_planner.total_running_time,
                "col_check_time": bit_planner.col_check_time,
                "vertex_selection_time": bit_planner.total_vertex_selection_time,
                "vertex_extension_time": bit_planner.total_vertex_extension_time,
                "vertex_extension_other_time": bit_planner.total_vertex_extension_time - bit_planner.col_check_time,
                "other_time": bit_planner.total_running_time - bit_planner.total_vertex_selection_time - bit_planner.total_vertex_extension_time,
            }

            with open(osp.join(log_dir, "result.json"), "w") as f:
                json.dump(res, f)

            # path_length = utils.calc_path_len(path)
            total_extend_time += bit_planner.total_running_time
            total_col_check_time += bit_planner.col_check_time
            total_vertex_selection_time += bit_planner.total_vertex_selection_time
            total_vertex_extension_time += bit_planner.total_vertex_extension_time
            total_extend_success += bit_planner.num_col_check - bit_planner.num_edge_col
            total_num_extend_called += bit_planner.loop_cnt

        elif args.planner.startswith('neural_d'):
            neural_planner_d.sampler.set_robot_bounds(low_bounds, high_bounds)
            log_dir = osp.join(planner_res_dir, "{}".format(i * plan_num + j))
            if not osp.exists(log_dir):
                os.mkdir(log_dir)

            neural_planner_d.algo.return_on_path_find = False
            res = neural_planner_d.solve_step_time(maze, maze.start, maze.goal, max_time, time_step_size)
            success_res = [tmp[0] for tmp in res]
            path_list = [tmp[1] for tmp in res]
            for p in path_list:
                if len(p) > 0:
                    # path = utils.interpolate(p)
                    # utils.visualize_nodes_global(occ_grid, path, maze.start, maze.goal, show=False, save=True, file_name=osp.join(learnt_log_dir, "planned_path.png"))
                    with open(osp.join(log_dir, 'planned_path.json'), 'w') as f:
                        json.dump(p, f)
                    break

            for idx, res in enumerate(success_res):
                if res:
                    success_list[idx] += 1

            res = {
                "extend_time": neural_planner_d.algo.total_running_time,
                "neural_extend_time": neural_planner_d.neural_expansion_time,
                "neural_extend_success_time": neural_planner_d.neural_expansion_success_time,
                "neural_extend_fail_time": neural_planner_d.neural_expansion_fail_time,
                "col_check_time": neural_planner_d.col_check_time,
                "col_check_success_time": neural_planner_d.col_check_success_time,
                "col_check_fail_time": neural_planner_d.col_check_fail_time,
                "vertex_selection_time": neural_planner_d.algo.total_vertex_selection_time,
                "vertex_selection_success_time": neural_planner_d.algo.total_vertex_selection_success_time,
                "vertex_selection_fail_time": neural_planner_d.algo.total_vertex_selection_fail_time,
                "vertex_extension_time": neural_planner_d.algo.total_vertex_extension_time,
                "vertex_extension_success_time": neural_planner_d.algo.total_vertex_extension_success_time,
                "vertex_extension_fail_time": neural_planner_d.algo.total_vertex_extension_fail_time,
                "vertex_extension_other_time": neural_planner_d.algo.total_vertex_extension_time - neural_planner_d.col_check_time - neural_planner_d.neural_expansion_time,
                "other_time": neural_planner_d.algo.total_running_time - neural_planner_d.algo.total_vertex_selection_time - neural_planner_d.algo.total_vertex_extension_time,
            }

            with open(osp.join(log_dir, "result.json"), "w") as f:
                json.dump(res, f)

            total_extend_time += neural_planner_d.algo.total_running_time
            total_extend_success += neural_planner_d.algo.num_extensions - neural_planner_d.expansion_col_cnt
            total_num_extend_called += neural_planner_d.algo.num_extensions
            total_neural_select_success += neural_planner_d.neural_expansion_cnt - neural_planner_d.selector_col_cnt
            total_neural_select_called += neural_planner_d.neural_expansion_cnt
            total_neural_extend_time += neural_planner_d.neural_expansion_time
            total_neural_extend_success_time += neural_planner_d.neural_expansion_success_time
            total_neural_extend_fail_time += neural_planner_d.neural_expansion_fail_time
            total_col_check_time += neural_planner_d.col_check_time
            total_col_check_success_time += neural_planner_d.col_check_success_time
            total_col_check_fail_time += neural_planner_d.col_check_fail_time
            total_vertex_selection_time += neural_planner_d.algo.total_vertex_selection_time
            total_vertex_selection_success_time += neural_planner_d.algo.total_vertex_selection_success_time
            total_vertex_selection_fail_time += neural_planner_d.algo.total_vertex_selection_fail_time
            total_vertex_extension_time += neural_planner_d.algo.total_vertex_extension_time
            total_vertex_extension_success_time += neural_planner_d.algo.total_vertex_extension_success_time
            total_vertex_extension_fail_time += neural_planner_d.algo.total_vertex_extension_fail_time

            print(neural_planner_d.algo.num_extensions, neural_planner_d.algo.total_succcesful_vertex_extension)
            avg_time_per_successful_ext += neural_planner_d.algo.total_running_time / neural_planner_d.algo.total_succcesful_vertex_extension
            avg_ext_per_successful_ext += neural_planner_d.algo.num_extensions / neural_planner_d.algo.total_succcesful_vertex_extension

        elif args.planner.startswith('neural_g'):
            learnt_log_dir = osp.join(planner_res_dir, "{}".format(i * plan_num + j))
            if not osp.exists(learnt_log_dir):
                os.mkdir(learnt_log_dir)

            neural_planner_g.algo.return_on_path_find = False
            res = neural_planner_g.solve_step_time(maze, maze.start, maze.goal, max_time, time_step_size)
            success_res = [tmp[0] for tmp in res]
            path_list = [tmp[1] for tmp in res]
            for p in path_list:
                if len(p) > 0:
                    # path = utils.interpolate(p)
                    # utils.visualize_nodes_global(mesh_path, occ_grid, path, maze.start, maze.goal, show=False, save=True, file_name=osp.join(learnt_log_dir, "planned_path.png"))
                    with open(osp.join(learnt_log_dir, 'planned_path.json'), 'w') as f:
                        json.dump(p, f)
                    break

            for idx, res in enumerate(success_res):
                if res:
                    success_list[idx] += 1

            print(neural_planner_g.neural_extend_cnt)

            res = {
                "extend_time": neural_planner_g.algo.total_running_time,
                "neural_extend_time": neural_planner_g.neural_extend_time,
                "neural_extend_success_time": neural_planner_g.neural_extend_success_time,
                "neural_extend_fail_time": neural_planner_g.neural_extend_fail_time,
                "col_check_time": neural_planner_g.col_check_time,
                "col_check_success_time": neural_planner_g.col_check_success_time,
                "col_check_fail_time": neural_planner_g.col_check_fail_time,
                "vertex_selection_time": neural_planner_g.algo.total_vertex_selection_time,
                "vertex_selection_success_time": neural_planner_g.algo.total_vertex_selection_success_time,
                "vertex_selection_fail_time": neural_planner_g.algo.total_vertex_selection_fail_time,
                "vertex_extension_time": neural_planner_g.algo.total_vertex_extension_time,
                "vertex_extension_success_time": neural_planner_g.algo.total_vertex_extension_success_time,
                "vertex_extension_fail_time": neural_planner_g.algo.total_vertex_extension_fail_time,
                "vertex_extension_other_time": neural_planner_g.algo.total_vertex_extension_time - neural_planner_g.col_check_time - neural_planner_g.neural_extend_time,
                "other_time": neural_planner_g.algo.total_running_time - neural_planner_g.algo.total_vertex_selection_time - neural_planner_g.algo.total_vertex_extension_time,
            }

            with open(osp.join(learnt_log_dir, "result.json"), "w") as f:
                json.dump(res, f)

            total_extend_time += neural_planner_g.algo.total_running_time
            total_extend_success += neural_planner_g.algo.num_extensions - neural_planner_g.extension_col_cnt
            total_num_extend_called += neural_planner_g.algo.num_extensions
            total_neural_select_success += neural_planner_g.neural_extend_cnt - neural_planner_g.selector_col_cnt
            total_neural_select_called += neural_planner_g.neural_extend_cnt
            total_neural_extend_time += neural_planner_g.neural_extend_time
            total_neural_extend_success_time += neural_planner_g.neural_extend_success_time
            total_neural_extend_fail_time += neural_planner_g.neural_extend_fail_time
            total_col_check_time += neural_planner_g.col_check_time
            total_col_check_success_time += neural_planner_g.col_check_success_time
            total_col_check_fail_time += neural_planner_g.col_check_fail_time
            total_vertex_selection_time += neural_planner_g.algo.total_vertex_selection_time
            total_vertex_selection_success_time += neural_planner_g.algo.total_vertex_selection_success_time
            total_vertex_selection_fail_time += neural_planner_g.algo.total_vertex_selection_fail_time
            total_vertex_extension_time += neural_planner_g.algo.total_vertex_extension_time
            total_vertex_extension_success_time += neural_planner_g.algo.total_vertex_extension_success_time
            total_vertex_extension_fail_time += neural_planner_g.algo.total_vertex_extension_fail_time

            avg_time_per_successful_ext += neural_planner_g.algo.total_running_time / neural_planner_g.algo.total_succcesful_vertex_extension
            avg_ext_per_successful_ext += neural_planner_g.algo.num_extensions / neural_planner_g.algo.total_succcesful_vertex_extension

    print("base_success_list", base_success_list)
    print("success_list", success_list)



# timing
total_vertex_selection_time = total_vertex_selection_time
total_vertex_extension_time = total_vertex_extension_time
print("Timing analysis: neural_total: {}, col_total: {}".format(total_neural_extend_time, total_col_check_time))
print("Timing analysis: neural_success: {}, fail: {}".format(total_neural_extend_success_time, total_neural_extend_fail_time))
print("Timing analysis: col_success: {}, fail: {}".format(total_col_check_success_time, total_col_check_fail_time))
print("Timing analysis: total: {}, vertex_selection: {}, vertex_extension: {}".format(total_extend_time, total_vertex_selection_time, total_vertex_extension_time))
print("Timing analysis: vertex_selection_success: {}, fail: {}".format(total_vertex_selection_success_time, total_vertex_selection_fail_time))
print("Timing analysis: vertex_extension_success: {}, fail: {}".format(total_vertex_extension_success_time, total_vertex_extension_fail_time))
vertex_extension_other_time = total_vertex_extension_time - total_col_check_time - total_neural_extend_time
total_other_time = total_extend_time - total_vertex_selection_time - total_vertex_extension_time
print("Timing analysis: other time: {}, vertex extension other time: {}".format(total_other_time, vertex_extension_other_time))

avg_time_per_successful_ext /= (env_num * plan_num)
print("Timing analysis: avg_time_per_successful_ext: {}".format(avg_time_per_successful_ext))
avg_ext_per_successful_ext /= (env_num * plan_num)
print("Timing analysis: avg_ext_per_successful_ext: {}".format(avg_ext_per_successful_ext))

print("base_success_list", base_success_list)
print("success_list", success_list)

res = {
    "total_extend_time": total_extend_time,
    "total_neural_extend_time": total_neural_extend_time,
    "total_neural_extend_success_time": total_neural_extend_success_time,
    "total_neural_extend_fail_time": total_neural_extend_fail_time,
    "total_col_check_time": total_col_check_time,
    "total_col_check_success_time": total_col_check_success_time,
    "total_col_check_fail_time": total_col_check_fail_time,
    "base_success_list": base_success_list,
    "success_list": success_list,
    "total_vertex_selection_time": total_vertex_selection_time,
    "total_vertex_selection_success_time": total_vertex_selection_success_time,
    "total_vertex_selection_fail_time": total_vertex_selection_fail_time,
    "total_vertex_extension_time": total_vertex_extension_time,
    "total_vertex_extension_success_time": total_vertex_extension_success_time,
    "total_vertex_extension_fail_time": total_vertex_extension_fail_time,
    "vertex_extension_other_time": vertex_extension_other_time,
    "total_other_time": total_other_time,
    "avg_time_per_successful_ext": avg_time_per_successful_ext,
    "avg_ext_per_successful_ext": avg_ext_per_successful_ext
}

with open(osp.join(planner_res_dir, "result.json"), "w") as f:
    json.dump(res, f)