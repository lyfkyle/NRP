import os
import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))

import json
from collections import defaultdict
import datetime
import torch
import argparse
import numpy as np

from planner.neural_planner_cvae import NeuralPlannerCVAE
from planner.neural_planner_g import NRP_g
from planner.neural_planner_d import NRP_d
from planner.rrt_planner import RRTPlanner
from planner.bit_star import BITStar
from env.fetch_11d.maze import Fetch11DEnv
from env.snake_8d.maze import Snake8DEnv

CUR_DIR = osp.dirname(osp.abspath(__file__))
DRAW_RESULT = False
LOG_TREE = True

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--env',  default='snake_8d')
parser.add_argument('--name',  default='10_ext')
parser.add_argument('--planner', default='rrt_star')
parser.add_argument('--viz', action="store_true")
args = parser.parse_args()

now = datetime.datetime.now()
if args.name == '':
    date_time = now.strftime("%m-%d-%Y-%H-%M-%S")
    res_dir = osp.join(CUR_DIR, f"eval_res/{args.env}/{date_time}")
else:
    res_dir = osp.join(CUR_DIR, f"eval_res/{args.env}/{args.name}")
planner_res_dir = osp.join(res_dir, args.planner)
if not osp.exists(res_dir):
    os.makedirs(res_dir)
if not osp.exists(planner_res_dir):
    os.makedirs(planner_res_dir)

if args.env == "snake_8d":
    env = Snake8DEnv(gui=False)
    dim = 8
    occ_grid_dim = [1, 40, 40]
elif args.env == "fetch_11d":
    env = Fetch11DEnv(gui=False)
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
    bit_planner = BITStar()

if args.planner.startswith('cvae'):
    neural_goal_bias = 0.4
    uniform_bias = 0.5
    model_path = osp.join(CUR_DIR, f"../planner/cvae/models/{args.env}/cvae_global.pt")
    cvae_planner = NeuralPlannerCVAE(model_path, optimal="star" in args.planner)
    cvae_planner.algo.goal_bias = neural_goal_bias
    cvae_planner.uniform_bias = uniform_bias

# Neural extension planner
if args.planner.startswith('nrp_d'):
    neural_goal_bias = 0.4
    sl_bias = 0.01
    col_checker_path = osp.join(CUR_DIR, f"../planner/local_sampler_d/models/{args.env}/model_col_final.pt")
    selector_path = osp.join(CUR_DIR, f"../planner/local_sampler_d/models/{args.env}/model_sel_final.pt")
    neural_planner_d = NRP_d(env, col_checker_path, selector_path, optimal="star" in args.planner, dim=dim, occ_grid_dim=occ_grid_dim)
    neural_planner_d.algo.goal_bias = neural_goal_bias
    neural_planner_d.algo.add_intermediate_state = False
    neural_planner_d.sl_bias = sl_bias
    neural_planner_d.only_sample_col_free = True  # Hack to enable comparison against ebsa expansion strategy
    neural_planner_d.sample_around_goal = True  # Hack to enable comparison against ebsa expansion strategy

if args.planner.startswith('nrp_g'):
    neural_goal_bias = 0.4
    sl_bias = 0.2
    model_path = osp.join(CUR_DIR, f"../planner/local_sampler_g/models/{args.env}/cvae_sel.pt")
    neural_planner_g = NRP_g(env, model_path, optimal="star" in args.planner, dim=dim, occ_grid_dim=occ_grid_dim)
    neural_planner_g.algo.goal_bias = neural_goal_bias
    neural_planner_g.algo.add_intermediate_state = False
    neural_planner_g.sl_bias = sl_bias
    neural_planner_g.only_sample_col_free = True  # Hack to enable comparison against ebsa expansion strategy
    neural_planner_g.sample_around_goal = True  # Hack to enable comparison against ebsa expansion strategy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Constants
dataset_name = "ext_test"

# envs = [i for i in range(250)]
# envs = [i for i in range(150)
envs = [3]
plan_num = 20
env_num = len(envs)

max_extension = 1
ext_step_size = 1

base_success_list = [0] * (max_extension // ext_step_size)
success_list = [0] * (max_extension // ext_step_size)
total_extend_time = 0
total_num_extend_called = 0
total_neural_extend_time = 0
total_neural_extend_success_time = 0
total_neural_extend_fail_time = 0
total_col_check_time = 0
total_extend_success = 0
total_neural_select_success = 0
total_neural_select_called = 0
total_neural_extend_called = 0
total_neural_extend_success = 0
extend_success_rate = 0
failed_env_idx = defaultdict(list)
for i in envs:
    env.clear_obstacles()
    env_dir = osp.join(CUR_DIR, f"../env/{args.env}/dataset/{dataset_name}/{i}")
    occ_grid = env.utils.get_occ_grid(env_dir)
    mesh_path = env.utils.get_mesh_path(env_dir)

    env.load_mesh(mesh_path)
    env.load_occupancy_grid(occ_grid, add_enclosing=True)

    for j in range(plan_num):
        print("Loading env {}".format(i))
        # env_dir = osp.join(CUR_DIR, f"../env/{args.env}/dataset/{dataset_name}/{i}")
        # env_dir = osp.join(CUR_DIR, "../dataset/test_env/{}".format(219))
        # with open(osp.join(env_dir, "obstacle_dict.json")) as f:
        #     obstacle_dict = json.load(f)

        occ_grid = env.get_occupancy_grid()

        with open(osp.join(env_dir, "start_goal.json")) as f:
            start_goal = json.load(f)
        # start_goal = json.load(osp.join(env_dir, "start_goal.json"))
        env.start = start_goal[0]
        env.goal = start_goal[1]
        env.robot.set_state(env.start)

        log_dir = osp.join(planner_res_dir, f"{i}/{j}")
        if not osp.exists(log_dir):
            os.makedirs(log_dir)

        if args.planner.startswith('rrt'):
            # Base planner
            if LOG_TREE:
                rrt_planner.algo.log_dir = log_dir  # log tree info
                rrt_planner.log_dir = log_dir
            rrt_planner.log_extension_info = True
            rrt_planner.algo.return_on_path_find = False
            res = rrt_planner.solve_step_extension(env, env.start, env.goal, max_extension, ext_step_size)
            success_res = [tmp[0] for tmp in res]
            path_list = [tmp[1] for tmp in res]
            for idx, p in enumerate(path_list):
                # path = env.utils.interpolate(p)
                # env.utils.visualize_nodes_global(mesh_path, occ_grid, path, maze.start, maze.goal, show=False, save=True, file_name=osp.join(base_log_dir, "planned_path.png"))
                with open(osp.join(log_dir, 'planned_path_{}.json'.format(idx)), 'w') as f:
                    json.dump(p, f)

            for idx, res in enumerate(success_res):
                if res:
                    base_success_list[idx] += 1

            # path_length = env.utils.calc_path_len(path)
            total_extend_time += rrt_planner.algo.total_running_time
            total_col_check_time += rrt_planner.col_check_time / (rrt_planner.algo.num_extensions + 1e-8)
            total_extend_success += rrt_planner.algo.num_extensions - rrt_planner.extension_col_cnt
            total_num_extend_called += rrt_planner.algo.num_extensions

        elif args.planner.startswith('bit'):
            res = bit_planner.solve_step_extension(env, env.start, env.goal, max_extension, ext_step_size)
            success_res = [tmp[0] for tmp in res]
            path_list = [tmp[1] for tmp in res]
            for idx, p in enumerate(path_list):
                # path = env.utils.interpolate(p)
                # env.utils.visualize_nodes_global(mesh_path, occ_grid, path, maze.start, maze.goal, show=False, save=True, file_name=osp.join(base_log_dir, "planned_path.png"))
                with open(osp.join(log_dir, 'planned_path_{}.json'.format(idx)), 'w') as f:
                    json.dump(p, f)

            for idx, res in enumerate(success_res):
                if res:
                    base_success_list[idx] += 1

            # path_length = env.utils.calc_path_len(path)
            total_extend_time += bit_planner.total_running_time / (bit_planner.loop_cnt + 1e-8)
            total_col_check_time += bit_planner.col_check_time / (bit_planner.num_col_check + 1e-8)
            total_extend_success += bit_planner.num_col_check - bit_planner.num_edge_col
            total_num_extend_called += bit_planner.loop_cnt

        elif args.planner.startswith('cvae'):
            cvae_planner.log_extension_info = True
            cvae_planner.algo.return_on_path_find = False
            res = cvae_planner.solve_step_expansion(env, env.start, env.goal, max_extension, ext_step_size, mesh=mesh_path)
            success_res = [tmp[0] for tmp in res]
            path_list = [tmp[1] for tmp in res]
            for idx, p in enumerate(path_list):
                # path = env.utils.interpolate(p)
                # env.utils.visualize_nodes_global(mesh_path, occ_grid, path, maze.start, maze.goal, show=False, save=True, file_name=osp.join(base_log_dir, "planned_path.png"))
                with open(osp.join(log_dir, 'planned_path_{}.json'.format(idx)), 'w') as f:
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

        elif args.planner.startswith('nrp_d'):
            # Get robot bounds
            low_bounds = env.robot.get_joint_lower_bounds()
            high_bounds = env.robot.get_joint_higher_bounds()
            low_bounds[0] = -2
            low_bounds[1] = -2
            high_bounds[0] = 2
            high_bounds[1] = 2
            # print(low_bounds, high_bounds)
            neural_planner_d.sampler.set_robot_bounds(low_bounds, high_bounds)

            neural_planner_d.log_extension_info = True
            if LOG_TREE:
                neural_planner_d.log_dir = log_dir
                neural_planner_d.algo.log_dir = log_dir
            neural_planner_d.algo.return_on_path_find = False
            res = neural_planner_d.solve_step_extension(env, env.start, env.goal, max_extension, ext_step_size)
            success_res = [tmp[0] for tmp in res]
            path_list = [tmp[1] for tmp in res]
            for idx, p in enumerate(path_list):
                # path = env.utils.interpolate(p)
                # env.utils.visualize_nodes_global(mesh_path, occ_grid, path, maze.start, maze.goal, show=False, save=True, file_name=osp.join(base_log_dir, "planned_path.png"))
                with open(osp.join(log_dir, 'planned_path_{}.json'.format(idx)), 'w') as f:
                    json.dump(p, f)

            for idx, res in enumerate(success_res):
                if res:
                    success_list[idx] += 1

            total_extend_time += neural_planner_d.algo.total_running_time
            total_neural_extend_time += neural_planner_d.neural_expansion_time / (neural_planner_d.neural_expansion_cnt + 1e-8)
            total_neural_extend_success_time += neural_planner_d.neural_expansion_success_time / (neural_planner_d.neural_expansion_cnt + 1e-8)
            total_neural_extend_fail_time += neural_planner_d.neural_expansion_fail_time / (neural_planner_d.neural_expansion_cnt + 1e-8)
            total_col_check_time += neural_planner_d.col_check_time / (neural_planner_d.neural_expansion_cnt + 1e-8)
            total_extend_success += neural_planner_d.algo.num_extensions - neural_planner_d.expansion_col_cnt
            total_num_extend_called += neural_planner_d.algo.num_extensions
            total_neural_extend_success += neural_planner_d.neural_expansion_cnt - neural_planner_d.neural_expansion_col_cnt
            total_neural_extend_called += neural_planner_d.neural_expansion_cnt
            total_neural_select_success += neural_planner_d.neural_select_cnt - neural_planner_d.selector_col_cnt
            total_neural_select_called += neural_planner_d.neural_select_cnt

        elif args.planner.startswith('nrp_g'):
            neural_planner_g.log_extension_info = True
            if LOG_TREE:
                neural_planner_g.log_dir = log_dir
                neural_planner_g.algo.log_dir = log_dir
            neural_planner_g.algo.return_on_path_find = False
            res = neural_planner_g.solve_step_extension(env, env.start, env.goal, max_extension, ext_step_size)
            success_res = [tmp[0] for tmp in res]
            path_list = [tmp[1] for tmp in res]
            for idx, p in enumerate(path_list):
                # path = env.utils.interpolate(p)
                # env.utils.visualize_nodes_global(mesh_path, occ_grid, path, maze.start, maze.goal, show=False, save=True, file_name=osp.join(base_log_dir, "planned_path.png"))
                with open(osp.join(log_dir, 'planned_path_{}.json'.format(idx)), 'w') as f:
                    json.dump(p, f)

            for idx, res in enumerate(success_res):
                if res:
                    success_list[idx] += 1
                else:
                    failed_env_idx[idx].append(i * plan_num + j)

            # print(neural_planner_g.extension_col_cnt, neural_planner_g.selector_col_cnt)

            total_extend_time += neural_planner_g.algo.total_running_time
            total_neural_extend_time += neural_planner_g.neural_extend_time / (neural_planner_g.neural_extend_cnt + 1e-8)
            total_neural_extend_success_time += neural_planner_g.neural_extend_success_time / (neural_planner_g.neural_extend_cnt + 1e-8)
            total_neural_extend_fail_time += neural_planner_g.neural_extend_fail_time / (neural_planner_g.neural_extend_cnt + 1e-8)
            total_col_check_time += neural_planner_g.col_check_time / (neural_planner_g.neural_extend_cnt + 1e-8)
            total_extend_success += neural_planner_g.algo.num_extensions - neural_planner_g.extension_col_cnt
            total_num_extend_called += neural_planner_g.algo.num_extensions
            total_neural_extend_success += neural_planner_g.neural_extend_cnt - neural_planner_g.neural_extend_col_cnt
            total_neural_extend_called += neural_planner_g.neural_extend_cnt
            total_neural_select_success += neural_planner_g.neural_select_cnt - neural_planner_g.selector_col_cnt
            total_neural_select_called += neural_planner_g.neural_select_cnt

    print("base_success_list", base_success_list)
    print("success_list", success_list)
    print("failed_env_idx", failed_env_idx)

# timing
avg_total_extend_time = total_extend_time / float(env_num * plan_num)
avg_neural_extend_time = total_neural_extend_time / float(env_num * plan_num)
avg_neural_extend_success_time = total_neural_extend_success_time / float(env_num * plan_num)
avg_neural_extend_fail_time = total_neural_extend_fail_time / float(env_num * plan_num)
avg_col_check_time = total_col_check_time / float(env_num * plan_num)
print("Average total sampling time for learnt: {}, neural: {}, col: {}".format(avg_total_extend_time, avg_neural_extend_time, avg_col_check_time))
print("Average neural extending time: success: {}, fail: {}".format(avg_neural_extend_success_time, avg_neural_extend_fail_time))

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
    "avg_neural_extend_success_time": avg_neural_extend_success_time,
    "avg_neural_extend_fail_time": avg_neural_extend_fail_time,
    "avg_col_check_time": avg_col_check_time,
    "neural_extend_success_rate": neural_extend_success_rate,
    "neural_select_success_rate": neural_select_success_rate,
    "extend_success_rate": extend_success_rate,
    "avg_num_extend_called": avg_num_extend_called,
    "base_success_list": base_success_list,
    "success_list": success_list,
    "total_neural_extend_cnt": total_neural_select_called / (env_num * plan_num)
}

with open(osp.join(planner_res_dir, "result.json"), "w") as f:
    json.dump(res, f)
