import os
import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))

import numpy as np
import argparse
import torch
import datetime
import json

from env.snake_8d.maze import Snake8DEnv
from env.fetch_11d.maze import Fetch11DEnv
# from env.snake_8d import utils
from planner.bit_star import BITStar
from planner.rrt_planner import RRTPlanner
from planner.neural_planner_d_global import RRT_NE_D_Global
# from planner.neural_planner_g_global import RRT_NE_G_Global
from planner.vqmpt.vqmpt_rrt_planner import RRT_VQMPT

CUR_DIR = osp.dirname(osp.abspath(__file__))

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--name',  default='test_time_v3')
parser.add_argument('--env',  default='fetch_11d')
parser.add_argument('--planner', default='vqmpt_rrt')
parser.add_argument('--logtree', action='store_true')
args = parser.parse_args()
LOG_TREE = args.logtree

now = datetime.datetime.now()
if args.name == '':
    date_time = now.strftime("%m-%d-%Y-%H-%M-%S")
    res_dir = osp.join(CUR_DIR, "eval_res/{}/{}".format(args.env, date_time))
else:
    res_dir = osp.join(CUR_DIR, "eval_res/{}/{}".format(args.env, args.name))
planner_res_dir = osp.join(res_dir, args.planner)
if not osp.exists(res_dir):
    os.mkdir(res_dir)
if not osp.exists(planner_res_dir):
    os.mkdir(planner_res_dir)

# Constants
env_num = 250
plan_num = 1

if args.env == "snake_8d":
    maze = Snake8DEnv(gui=False)
    dim = 8
    occ_grid_dim = [1, 100, 100]
elif args.env == "fetch_11d":
    maze = Fetch11DEnv(gui=False)
    dim = 11
    occ_grid_dim = [150, 150, 20]

# RRT planner
if args.planner.startswith('rrt'):
    rrt_planner = RRTPlanner()
    rrt_planner.algo.goal_bias = 0.1

# BITstar planner
if args.planner.startswith('bit'):
    bit_planner = BITStar()

# Neural extension planner
if args.planner.startswith('neural_d'):
    neural_goal_bias = 0.5
    sl_bias = 0.4
    col_checker_path = osp.join(CUR_DIR, "../planner/local_sampler_d/models/{}/model_col_global.pt".format(args.env))
    selector_path = osp.join(CUR_DIR, "../planner/local_sampler_d/models/{}/model_sel_global.pt".format(args.env))
    neural_planner_d = RRT_NE_D_Global(
        maze,
        col_checker_path,
        selector_path,
        dim=dim,
        occ_grid_dim=occ_grid_dim,
        no_col_checker="no_col" in args.planner,
    )
    neural_planner_d.algo.goal_bias = neural_goal_bias
    neural_planner_d.algo.add_intermediate_state = False
    neural_planner_d.sl_bias = sl_bias

if args.planner.startswith('neural_g'):
    neural_goal_bias = 0.4
    sl_bias = 0.3
    model_path = osp.join(CUR_DIR, "../local_sampler_g/models/cvae_sel_global.pt")
    neural_planner_g = RRT_NE_G_Global(model_path)
    neural_planner_g.algo.goal_bias = neural_goal_bias
    neural_planner_g.algo.add_intermediate_state = False
    neural_planner_g.sl_bias = sl_bias

if args.planner.startswith('vqmpt'):
    neural_goal_bias = 0.4
    # sl_bias = 0.01
    dim = 8 if args.env == 'snake_8d' else 11
    model_path = osp.join(CUR_DIR, f"../planner/vqmpt/models/{dim}d")
    rrt_vqmpt = RRT_VQMPT(model_path, dim=dim)
    rrt_vqmpt.algo.goal_bias = neural_goal_bias
    rrt_vqmpt.algo.add_intermediate_state = False
    # rrt_vqmpt.sl_bias = sl_bias

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

max_time = 10
time_step_size = 0.5

for repeat in range(10):
    base_success_list = [0] * int(max_time / time_step_size)
    success_list = [0] * int(max_time / time_step_size)
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
    for i in range(env_num):
        maze.clear_obstacles()
        maze_dir = osp.join(CUR_DIR, "../env/{}/dataset/test_env/{}".format(args.env, i))
        occ_grid = maze.utils.get_occ_grid(maze_dir)
        mesh_path = maze.utils.get_mesh_path(maze_dir)

        maze.load_mesh(mesh_path)
        maze.load_occupancy_grid(occ_grid, add_enclosing=True)

        for j in range(plan_num):
            print("Loading env {}".format(i * plan_num + j))
            env_dir = osp.join(CUR_DIR, "../env/{}/dataset/test_env/{}".format(args.env, i * plan_num + j))
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
            # print(low_bounds, high_bounds)

            if args.planner.startswith('rrt'):
                log_dir = osp.join(planner_res_dir, "{}".format(i * plan_num + j))
                if not osp.exists(log_dir):
                    os.mkdir(log_dir)

                rrt_planner.algo.return_on_path_find = False
                res = rrt_planner.solve_step_time(maze, maze.start, maze.goal, max_time, time_step_size, mesh_path)
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

                res = bit_planner.solve_step_time(maze, maze.start, maze.goal, max_time, time_step_size)
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

                neural_planner_d.algo.return_on_path_find = False
                res = neural_planner_d.solve_step_time(maze, maze.start, maze.goal, max_time, time_step_size)
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

                neural_planner_g.algo.return_on_path_find = False
                res = neural_planner_g.solve_step_time(maze, maze.start, maze.goal, max_time, time_step_size)
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

                # print(neural_planner_g.neural_extend_cnt)

                total_extend_time += neural_planner_g.algo.total_running_time
                total_neural_extend_time += neural_planner_g.neural_extend_time / (neural_planner_g.neural_extend_cnt + 1e-8)
                total_col_check_time += neural_planner_g.col_check_time / (neural_planner_g.neural_extend_cnt + 1e-8)
                total_extend_success += neural_planner_g.algo.num_extensions - neural_planner_g.extension_col_cnt
                total_num_extend_called += neural_planner_g.algo.num_extensions
                total_neural_extend_success += neural_planner_g.neural_extend_cnt - neural_planner_g.neural_extend_col_cnt
                total_neural_extend_called += neural_planner_g.neural_extend_cnt
                total_neural_select_success += neural_planner_g.neural_select_cnt - neural_planner_g.selector_col_cnt
                total_neural_select_called += neural_planner_g.neural_select_cnt

            elif args.planner.startswith('vqmpt'):
                log_dir = osp.join(planner_res_dir, "{}".format(i * plan_num + j))
                if not osp.exists(log_dir):
                    os.mkdir(log_dir)

                # Base planner
                if LOG_TREE:
                    rrt_vqmpt.algo.log_dir = log_dir # log tree info
                    rrt_vqmpt.log_dir = log_dir
                rrt_vqmpt.log_extension_info = True
                rrt_vqmpt.algo.return_on_path_find = False
                res = rrt_vqmpt.solve_step_time(maze, maze.start, maze.goal, max_time, time_step_size)
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
                total_extend_time += rrt_vqmpt.algo.total_running_time
                total_col_check_time += rrt_vqmpt.col_check_time / (rrt_vqmpt.algo.num_extensions + 1e-8)
                total_extend_success += rrt_vqmpt.algo.num_extensions - rrt_vqmpt.extension_col_cnt
                total_num_extend_called += rrt_vqmpt.algo.num_extensions

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