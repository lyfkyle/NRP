import os
import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../../'))

import numpy as np
import argparse
import torch
import datetime
from collections import defaultdict
import json

from env.fetch_11d.maze import Fetch11DEnv as Maze
from env.fetch_11d import utils
from planner.bit_star import BITStar
from planner.rrt_planner import RRTPlanner
# from planner.decoupled_planner import DecoupledRRTPlanner, HybridAStar
from planner.neural_planner_d import NRP_d
from planner.neural_planner_g import NRP_g

CUR_DIR = osp.dirname(osp.abspath(__file__))

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--name',  default='test_ext')
parser.add_argument('--planner', default='rrt')
parser.add_argument('--logtree', action='store_true')
args = parser.parse_args()
LOG_TREE = args.logtree

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

# Constants
env_num = 10
plan_num = 1
dim = 11
occ_grid_dim = [40, 40, 20]
# occ_grid_dim = 40
# occ_grid_dim_z = 20
turning_radius = 0.75
base_radius = 0.3
robot_height = 1.1
occ_grid_resolution = 0.1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
maze = Maze(gui=False)
robot_rest_position = list(maze.robot.rest_position)

# RRT planner
if args.planner.startswith('rrt'):
    rrt_planner = RRTPlanner(optimal="star" in args.planner)
    rrt_planner.algo.goal_bias = 0.3
    if "is" in args.planner:
        rrt_planner.add_intermediate_state = True

# BITstar planner
if args.planner.startswith('bit'):
    bit_planner = BITStar()

# Neural extension planner
# Hyperparameters:
if args.planner.startswith('neural_d'):
    neural_goal_bias = 0.4
    sl_bias = 0.01
    col_checker_path = osp.join(CUR_DIR, "../local_sampler_d/models/model_col_final.pt")
    selector_path = osp.join(CUR_DIR, "../local_sampler_d/models/model_sel_final.pt")
    neural_planner_d = NRP_d(
        maze,
        col_checker_path,
        selector_path,
        optimal="star" in args.planner,
        dim=dim,
        occ_grid_dim=occ_grid_dim,
        no_col_checker="no_col" in args.planner,
    )
    neural_planner_d.algo.goal_bias = neural_goal_bias
    neural_planner_d.sl_bias = sl_bias

# CVAE Planner
if args.planner.startswith('neural_g'):
    neural_goal_bias = 0.4
    sl_bias = 0.01
    model_path = osp.join(CUR_DIR, "../local_sampler_g/models/cvae_sel.pt")
    neural_planner_g = NRP_g(model_path, optimal="star" in args.planner)
    neural_planner_g.algo.goal_bias = neural_goal_bias
    neural_planner_g.sl_bias = sl_bias

# decoupled_rrt_star
if args.planner.startswith('decoupled'):
    decoupled_rrt_planner = DecoupledRRTPlanner(optimal="star" in args.planner)
    decoupled_rrt_planner.algo.goal_bias = 0.1
    hybrid_astar_planner = HybridAStar(br=base_radius, tr=turning_radius, map_res=occ_grid_resolution)

max_extension = 500
ext_step_size = 25

for repeat in range(10):
    base_success_list = [0] * (max_extension // ext_step_size)
    success_list = [0] * (max_extension // ext_step_size)
    total_extend_time = 0
    total_num_extend_called = 0
    total_neural_extend_time = 0
    total_col_check_time = 0
    total_extend_success = 0
    total_neural_select_success = 0
    total_neural_select_called = 0
    total_neural_extend_called = 0
    total_neural_extend_success = 0
    extend_success_rate = 0
    failed_env_idx = defaultdict(list)

    maze.clear_obstacles()
    with open(os.path.join(CUR_DIR, "map/rls_occ_grid.npy"), 'rb') as f:
        occ_grid = np.load(f)
    mesh_path = osp.join(CUR_DIR, "map/rls_mesh.obj")

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
            occ_grid_2d = np.any(occ_grid[:, :, :int(robot_height / occ_grid_resolution)], axis=2).astype(int)

            with open(osp.join(CUR_DIR, "test_path/test_path_{}.json".format(env_idx)), "r") as f:
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

                # Base planner
                if LOG_TREE:
                    rrt_planner.algo.log_dir = log_dir # log tree info
                    rrt_planner.log_dir = log_dir
                rrt_planner.log_extension_info = True
                rrt_planner.algo.return_on_path_find = False
                res = rrt_planner.solve_step_extension(maze, maze.start, maze.goal, max_extension, ext_step_size)
                success_res = [tmp[0] for tmp in res]
                path_list = [tmp[1] for tmp in res]
                for idx, p in enumerate(path_list):
                    # path = utils.interpolate(p)
                    # utils.visualize_nodes_global(mesh_path, occ_grid, path, maze.start, maze.goal, show=False, save=True, file_name=osp.join(base_log_dir, "planned_path.png"))
                    with open(osp.join(log_dir, 'planned_path_{}_{}.json'.format(repeat, idx)), 'w') as f:
                        json.dump(p, f)

                for idx, res in enumerate(success_res):
                    if res:
                        base_success_list[idx] += 1

                # path_length = utils.calc_path_len(path)
                total_extend_time += rrt_planner.algo.total_running_time
                total_col_check_time += rrt_planner.col_check_time / (rrt_planner.algo.num_extensions + 1e-8)
                total_extend_success += rrt_planner.algo.num_extensions - rrt_planner.extension_col_cnt
                total_num_extend_called += rrt_planner.algo.num_extensions

            elif args.planner.startswith('bit'):
                log_dir = osp.join(planner_res_dir, "{}".format(i * plan_num + j))
                if not osp.exists(log_dir):
                    os.mkdir(log_dir)

                res = bit_planner.solve_step_extension(maze, maze.start, maze.goal, max_extension, ext_step_size)
                success_res = [tmp[0] for tmp in res]
                path_list = [tmp[1] for tmp in res]
                for idx, p in enumerate(path_list):
                    # path = utils.interpolate(p)
                    # utils.visualize_nodes_global(mesh_path, occ_grid, path, maze.start, maze.goal, show=False, save=True, file_name=osp.join(base_log_dir, "planned_path.png"))
                    with open(osp.join(log_dir, 'planned_path_{}_{}.json'.format(repeat, idx)), 'w') as f:
                        json.dump(p, f)

                for idx, res in enumerate(success_res):
                    if res:
                        base_success_list[idx] += 1

                # path_length = utils.calc_path_len(path)
                total_extend_time += bit_planner.total_running_time / (bit_planner.loop_cnt + 1e-8)
                total_col_check_time += bit_planner.col_check_time / (bit_planner.num_col_check + 1e-8)
                total_extend_success += bit_planner.num_col_check - bit_planner.num_edge_col
                total_num_extend_called += bit_planner.loop_cnt

            elif args.planner.startswith('neural_d'):
                neural_planner_d.sampler.set_robot_bounds(low_bounds, high_bounds)
                log_dir = osp.join(planner_res_dir, "{}".format(i * plan_num + j))
                if not osp.exists(log_dir):
                    os.mkdir(log_dir)

                if LOG_TREE:
                    neural_planner_d.log_dir = log_dir
                    neural_planner_d.algo.log_dir = log_dir
                neural_planner_d.log_extension_info = True
                neural_planner_d.algo.return_on_path_find = False
                res = neural_planner_d.solve_step_extension(maze, maze.start, maze.goal, max_extension, ext_step_size)
                success_res = [tmp[0] for tmp in res]
                path_list = [tmp[1] for tmp in res]
                for idx, p in enumerate(path_list):
                    # path = utils.interpolate(p)
                    # utils.visualize_nodes_global(mesh_path, occ_grid, path, maze.start, maze.goal, show=False, save=True, file_name=osp.join(base_log_dir, "planned_path.png"))
                    with open(osp.join(log_dir, 'planned_path_{}_{}.json'.format(repeat, idx)), 'w') as f:
                        json.dump(p, f)

                for idx, res in enumerate(success_res):
                    if res:
                        success_list[idx] += 1

                total_extend_time += neural_planner_d.algo.total_running_time
                total_neural_extend_time += neural_planner_d.neural_extend_time / (neural_planner_d.neural_extend_cnt + 1e-8)
                total_col_check_time += neural_planner_d.col_check_time / (neural_planner_d.neural_extend_cnt + 1e-8)
                total_extend_success += neural_planner_d.algo.num_extensions - neural_planner_d.extension_col_cnt
                total_num_extend_called += neural_planner_d.algo.num_extensions
                total_neural_extend_success += neural_planner_d.neural_extend_cnt - neural_planner_d.neural_extend_col_cnt
                total_neural_extend_called += neural_planner_d.neural_extend_cnt
                total_neural_select_success += neural_planner_d.neural_select_cnt - neural_planner_d.selector_col_cnt
                total_neural_select_called += neural_planner_d.neural_select_cnt

            elif args.planner.startswith('neural_g'):
                log_dir = osp.join(planner_res_dir, "{}".format(i * plan_num + j))
                if not osp.exists(log_dir):
                    os.mkdir(log_dir)


                if LOG_TREE:
                    neural_planner_g.log_dir = log_dir
                    neural_planner_g.algo.log_dir = log_dir
                neural_planner_g.log_extension_info = False
                neural_planner_g.algo.return_on_path_find = False
                res = neural_planner_g.solve_step_extension(maze, maze.start, maze.goal, max_extension, ext_step_size)
                success_res = [tmp[0] for tmp in res]
                path_list = [tmp[1] for tmp in res]
                for idx, p in enumerate(path_list):
                    # path = utils.interpolate(p)
                    # utils.visualize_nodes_global(mesh_path, occ_grid, path, maze.start, maze.goal, show=False, save=True, file_name=osp.join(base_log_dir, "planned_path.png"))
                    with open(osp.join(log_dir, 'planned_path_{}_{}.json'.format(repeat, idx)), 'w') as f:
                        json.dump(p, f)

                for idx, res in enumerate(success_res):
                    if res:
                        success_list[idx] += 1

                total_extend_time += neural_planner_g.algo.total_running_time
                total_neural_extend_time += neural_planner_g.neural_extend_time / (neural_planner_g.neural_extend_cnt + 1e-8)
                total_col_check_time += neural_planner_g.col_check_time / (neural_planner_g.neural_extend_cnt + 1e-8)
                total_extend_success += neural_planner_g.algo.num_extensions - neural_planner_g.extension_col_cnt
                total_num_extend_called += neural_planner_g.algo.num_extensions
                total_neural_extend_success += neural_planner_g.neural_extend_cnt - neural_planner_g.neural_extend_col_cnt
                total_neural_extend_called += neural_planner_g.neural_extend_cnt
                total_neural_select_success += neural_planner_g.neural_select_cnt - neural_planner_g.selector_col_cnt
                total_neural_select_called += neural_planner_g.neural_select_cnt

            elif args.planner.startswith('decoupled'):
                log_dir = osp.join(planner_res_dir, "{}".format(i * plan_num + j))
                if not osp.exists(log_dir):
                    os.mkdir(log_dir)

                decoupled_rrt_planner.algo.return_on_path_find = False
                start_stowed = maze.start.copy()
                start_stowed[3:] = robot_rest_position
                goal_stowed = maze.goal.copy()
                goal_stowed[3:] = robot_rest_position
                print(start_stowed[:3], goal_stowed[:3])
                astar_start = (start_stowed[0] / occ_grid_resolution, start_stowed[1] / occ_grid_resolution, start_stowed[2])
                astar_goal = (goal_stowed[0] / occ_grid_resolution, goal_stowed[1] / occ_grid_resolution, goal_stowed[2])
                res_2d, astar_time = hybrid_astar_planner.plan(astar_start, astar_goal, occ_grid_2d)

                # Astar should always succeed
                print("Astar time:", astar_time)
                assert res_2d is not None

                res1 = decoupled_rrt_planner.solve_step_extension(maze, maze.start, start_stowed, max_extension//2, ext_step_size//2)

                total_extend_time += decoupled_rrt_planner.algo.total_running_time
                total_col_check_time += decoupled_rrt_planner.col_check_time / (decoupled_rrt_planner.algo.num_extensions + 1e-8)
                total_extend_success += decoupled_rrt_planner.algo.num_extensions - decoupled_rrt_planner.extension_col_cnt
                total_num_extend_called += decoupled_rrt_planner.algo.num_extensions

                res2 = decoupled_rrt_planner.solve_step_extension(maze, goal_stowed, maze.goal, max_extension//2, ext_step_size//2)

                total_extend_time += decoupled_rrt_planner.algo.total_running_time
                total_col_check_time += decoupled_rrt_planner.col_check_time / (decoupled_rrt_planner.algo.num_extensions + 1e-8)
                total_extend_success += decoupled_rrt_planner.algo.num_extensions - decoupled_rrt_planner.extension_col_cnt
                total_num_extend_called += decoupled_rrt_planner.algo.num_extensions

                res_astar = []
                for i in range(len(res_2d[0])):
                    res_astar.append(tuple([res_2d[0][i] / 10, res_2d[1][i] / 10, res_2d[2][i]] + robot_rest_position))
                success_res = [(tmp1[0] and tmp2[0]) for tmp1, tmp2 in zip(res1, res2)]
                path_list = []
                for i in range(len(res1)):
                    if success_res[i]:
                        path_list.append(res1[i][1] + res_astar + res2[i][1])
                    else:
                        path_list.append([])
                # path_list = [tmp1[1] + res_astar + tmp2[1] for tmp1, tmp2 in zip(res1, res2)]
                for idx, p in enumerate(path_list):
                    # path = utils.interpolate(p)
                    # utils.visualize_nodes_global(mesh_path, occ_grid, p, maze.start, maze.goal, show=False, save=True, file_name="planned_path.png")
                    with open(osp.join(log_dir, 'planned_path_{}_{}.json'.format(repeat, idx)), 'w') as f:
                        json.dump(p, f)

                for idx, res in enumerate(success_res):
                    if res:
                        base_success_list[idx] += 1

                # path_length = utils.calc_path_len(path)

        print("base_success_list", base_success_list)
        print("success_list", success_list)

    # timing
    avg_total_extend_time = total_extend_time / float(env_num * plan_num)
    avg_neural_extend_time = total_neural_extend_time / float(env_num * plan_num)
    avg_col_check_time = total_col_check_time / float(env_num * plan_num)
    print("Average total sampling time for learnt: {}, neural: {}, col: {}".format(avg_total_extend_time, avg_neural_extend_time, avg_col_check_time))

    neural_extend_success_rate = total_neural_extend_success / float(total_neural_extend_called + 1e-8)
    print("neural extend success rate: {}".format(neural_extend_success_rate))

    neural_select_success_rate = total_neural_select_success / float(total_neural_select_called + 1e-8)
    print("neural select success rate: {}".format(neural_select_success_rate))

    extend_success_rate = total_extend_success / float(total_num_extend_called + 1e-8)
    print("extend_success_rate : {}".format(extend_success_rate))

    avg_num_extend_called = total_num_extend_called / (env_num * plan_num)
    print("avg_num_extend_called: {}".format(avg_num_extend_called))

    print("total_neural_extend_cnt", total_neural_select_called / (env_num * plan_num))

    print("base_success_list", base_success_list)
    print("success_list", success_list)

    res = {
        "avg_total_extend_time": avg_total_extend_time,
        "avg_neural_extend_time": avg_neural_extend_time,
        "avg_col_check_time": avg_col_check_time,
        "neural_extend_success_rate": neural_extend_success_rate,
        "neural_select_success_rate": neural_select_success_rate,
        "extend_success_rate": extend_success_rate,
        "avg_num_extend_called": avg_num_extend_called,
        "base_success_list": base_success_list,
        "success_list": success_list,
        "total_neural_extend_cnt": total_neural_select_called / (env_num * plan_num)
    }

    with open(osp.join(planner_res_dir, "result_{}.json".format(repeat)), "w") as f:
        json.dump(res, f)