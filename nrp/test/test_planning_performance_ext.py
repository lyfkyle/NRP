import os
import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))

import numpy as np
import argparse
import torch
import datetime
from collections import defaultdict
import json

from env.snake_8d.maze import Snake8DEnv
from env.fetch_11d.maze import Fetch11DEnv
from planner.bit_star import BITStar
from planner.rrt_planner import RRTPlanner
from planner.neural_planner_d import NRP_d
from planner.neural_planner_g import NRP_g
from planner.neural_planner_cvae import NeuralPlannerCVAE
from planner.neural_planner_fire import NeuralPlannerFire
from planner.fire.fire import Fire, FireEntry
from planner.vqmpt.vqmpt_rrt_planner import RRT_VQMPT

CUR_DIR = osp.dirname(osp.abspath(__file__))

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--name',  default='test_ext_500')
parser.add_argument('--env',  default='snake_8d')
parser.add_argument('--planner', default='rrt_star')
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
    occ_grid_dim = [1, 40, 40]
elif args.env == "fetch_11d":
    maze = Fetch11DEnv(gui=False)
    dim = 11
    occ_grid_dim = [40, 40, 20]

# RRT planner
if args.planner.startswith('rrt'):
    rrt_planner = RRTPlanner(optimal="star" in args.planner)
    rrt_planner.algo.goal_bias = 0.3
    if "is" in args.planner:
        rrt_planner.add_intermediate_state = True

# BITstar planner
if args.planner.startswith('bit'):
    bit_planner = BITStar(batch_size=200)

if args.planner.startswith('cvae'):
    neural_goal_bias = 0.4
    uniform_bias = 0.5
    model_path = osp.join(CUR_DIR, "../planner/cvae/models/{}/cvae_global.pt".format(args.env))
    cvae_planner = NeuralPlannerCVAE(model_path, optimal="star" in args.planner)
    cvae_planner.algo.goal_bias = neural_goal_bias
    cvae_planner.uniform_bias = uniform_bias

# Neural extension planner
if args.planner.startswith('neural_d'):
    neural_goal_bias = 0.4
    sl_bias = 0.01
    col_checker_path = osp.join(CUR_DIR, "../planner/local_sampler_d/models/{}/model_col_final.pt".format(args.env))
    selector_path = osp.join(CUR_DIR, "../planner/local_sampler_d/models/{}/model_sel_final.pt".format(args.env))
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
    neural_planner_d.algo.add_intermediate_state = False
    neural_planner_d.sl_bias = sl_bias

if args.planner.startswith('neural_g'):
    neural_goal_bias = 0.4
    sl_bias = 0.2
    model_path = osp.join(CUR_DIR, "../planner/local_sampler_g/models/{}/cvae_sel.pt".format(args.env))
    neural_planner_g = NRP_g(
        maze,
        model_path,
        optimal="star" in args.planner,
        dim=dim,
        occ_grid_dim=occ_grid_dim,
    )
    neural_planner_g.algo.goal_bias = neural_goal_bias
    neural_planner_g.algo.add_intermediate_state = False
    neural_planner_g.sl_bias = sl_bias

if args.planner.startswith('fire'):
    model_path = osp.join(CUR_DIR, "../planner/fire/models/fire.pt")
    fire_planner = NeuralPlannerFire(maze, model_path, optimal="star" in args.planner)
    fire_planner.uniform_bias = 0.2

if args.planner.startswith('vqmpt'):
    neural_goal_bias = 0.4
    model_path = osp.join(CUR_DIR, "../planner/vqmpt/models/{}d".format(dim))
    rrt_vqmpt = RRT_VQMPT(model_path, optimal="star" in args.planner, dim=dim)
    rrt_vqmpt.algo.goal_bias = neural_goal_bias
    rrt_vqmpt.algo.add_intermediate_state = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


max_extension = 500
ext_step_size = 25
repeat_num = 1

for repeat in range(repeat_num):
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
    for i in range(env_num):
        maze.clear_obstacles()
        maze_dir = osp.join(CUR_DIR, "../env/{}/dataset/test_env/{}".format(args.env, i))
        occ_grid = maze.utils.get_occ_grid(maze_dir)
        mesh_path = maze.utils.get_mesh_path(maze_dir)

        maze.load_mesh(mesh_path)
        maze.load_occupancy_grid(occ_grid, add_enclosing=True)

        # fire_planner.sampler.synthesize_sampling_distributions()

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

            elif args.planner.startswith('cvae'):
                log_dir = osp.join(planner_res_dir, "{}".format(i * plan_num + j))
                if not osp.exists(log_dir):
                    os.mkdir(log_dir)

                cvae_planner.log_extension_info = True
                cvae_planner.algo.return_on_path_find = False
                res = cvae_planner.solve_step_expansion(maze, maze.start, maze.goal, max_extension, ext_step_size, mesh=mesh_path)
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
                    else:
                        failed_env_idx[idx].append(i * plan_num + j)

                total_extend_time += cvae_planner.algo.total_running_time
                total_extend_success += cvae_planner.algo.num_extensions - cvae_planner.expansion_col_cnt
                total_num_extend_called += cvae_planner.algo.num_extensions
                total_neural_extend_time += cvae_planner.neural_expansion_time / (cvae_planner.extend_cnt + 1e-8)
                total_col_check_time += cvae_planner.col_check_time / (cvae_planner.extend_cnt + 1e-8)

            elif args.planner.startswith('fire'):
                log_dir = osp.join(planner_res_dir, "{}".format(i * plan_num + j))
                if not osp.exists(log_dir):
                    os.mkdir(log_dir)

                fire_planner.log_extension_info = True
                fire_planner.algo.return_on_path_find = False
                res = fire_planner.solve_step_extension(maze, maze.start, maze.goal, max_extension, ext_step_size, mesh=mesh_path)
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
                    else:
                        failed_env_idx[idx].append(i * plan_num + j)

                total_extend_time += fire_planner.algo.total_running_time
                total_extend_success += fire_planner.algo.num_extensions - fire_planner.expansion_col_cnt
                total_num_extend_called += fire_planner.algo.num_extensions
                total_neural_extend_time += fire_planner.neural_expansion_time / (fire_planner.extend_cnt + 1e-8)
                total_col_check_time += fire_planner.col_check_time / (fire_planner.extend_cnt + 1e-8)

            elif args.planner.startswith('neural_d'):
                # Get robot bounds
                low_bounds = maze.robot.get_joint_lower_bounds()
                high_bounds = maze.robot.get_joint_higher_bounds()
                low_bounds[0] = -2
                low_bounds[1] = -2
                high_bounds[0] = 2
                high_bounds[1] = 2
                # print(low_bounds, high_bounds)
                neural_planner_d.sampler.set_robot_bounds(low_bounds, high_bounds)

                log_dir = osp.join(planner_res_dir, "{}".format(i * plan_num + j))
                if not osp.exists(log_dir):
                    os.mkdir(log_dir)

                neural_planner_d.log_extension_info = True
                if LOG_TREE:
                    neural_planner_d.log_dir = log_dir
                    neural_planner_d.algo.log_dir = log_dir
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
                total_neural_extend_time += neural_planner_d.neural_expansion_time / (neural_planner_d.neural_expansion_cnt + 1e-8)
                total_col_check_time += neural_planner_d.col_check_time / (neural_planner_d.neural_expansion_cnt + 1e-8)
                total_extend_success += neural_planner_d.algo.num_extensions - neural_planner_d.expansion_col_cnt
                total_num_extend_called += neural_planner_d.algo.num_extensions
                total_neural_extend_success += neural_planner_d.neural_expansion_cnt - neural_planner_d.neural_expansion_col_cnt
                total_neural_extend_called += neural_planner_d.neural_expansion_cnt
                total_neural_select_success += neural_planner_d.neural_select_cnt - neural_planner_d.selector_col_cnt
                total_neural_select_called += neural_planner_d.neural_select_cnt

            elif args.planner.startswith('neural_g'):
                log_dir = osp.join(planner_res_dir, "{}".format(i * plan_num + j))
                if not osp.exists(log_dir):
                    os.mkdir(log_dir)

                neural_planner_g.log_extension_info = True
                if LOG_TREE:
                    neural_planner_g.log_dir = log_dir
                    neural_planner_g.algo.log_dir = log_dir
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
                    else:
                        failed_env_idx[idx].append(i * plan_num + j)

                # print(neural_planner_g.extension_col_cnt, neural_planner_g.selector_col_cnt)

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
                res = rrt_vqmpt.solve_step_extension(maze, maze.start, maze.goal, max_extension, ext_step_size)
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