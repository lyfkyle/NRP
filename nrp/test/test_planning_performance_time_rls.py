import os
import os.path as osp
import numpy as np
import argparse
import torch
import datetime
import json
from collections import defaultdict

from nrp.env.fetch_11d.env import Fetch11DEnv
from nrp.env.fetch_11d import utils
from nrp.planner.bit_star import BITStar
from nrp.planner.rrt_planner import RRTPlanner
from nrp.planner.decoupled_planner import DecoupledRRTPlanner, HybridAStar
from nrp.planner.neural_planner_d import NRP_d
from nrp.planner.neural_planner_g import NRP_g

CUR_DIR = osp.dirname(osp.abspath(__file__))

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("--name", default="test_time")
parser.add_argument("--planner", default="rrt")
parser.add_argument("--repeat", default=1, type=int)
args = parser.parse_args()

# args.planner = 'decoupled_rrt_star'

now = datetime.datetime.now()
if args.name == "":
    date_time = now.strftime("%m-%d-%Y-%H-%M-%S")
    res_dir = osp.join(CUR_DIR, "eval_res/{}/{}".format(args.env, date_time))
else:
    res_dir = osp.join(CUR_DIR, "eval_res/{}/{}".format(args.env, args.name))
planner_res_dir = osp.join(res_dir, args.planner)
if not osp.exists(res_dir):
    os.makedirs(res_dir)
if not osp.exists(planner_res_dir):
    os.makedirs(planner_res_dir)

# Constants
env = Fetch11DEnv(gui=False)
dim = 11
occ_grid_dim = [40, 40, 20]

turning_radius = 0.1
base_radius = 0.3
robot_height = 1.1
occ_grid_resolution = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
robot_rest_position = list(env.robot.rest_position)

# RRT planner
if args.planner.startswith("rrt"):
    rrt_planner = RRTPlanner(optimal="star" in args.planner)
    rrt_planner.algo.goal_bias = 0.1
    if "is" in args.planner:
        rrt_planner.add_intermediate_state = True

# BITstar planner
if args.planner.startswith("bit"):
    bit_planner = BITStar()

# Neural extension planner
# Hyperparameters:
if args.planner.startswith("nrp_d"):
    neural_goal_bias = 0.3
    sl_bias = 0.1
    selector_path = osp.join(ROOT_DIR, "train/{}/models/sampler_d_01_critical_v2.pt".format(args.env))
    neural_planner_d = NRP_d(
        env,
        selector_path,
        optimal="star" in args.planner,
        dim=dim,
        occ_grid_dim=occ_grid_dim,
    )
    neural_planner_d.algo.goal_bias = neural_goal_bias
    neural_planner_d.algo.add_intermediate_state = False
    neural_planner_d.sl_bias = sl_bias

if args.planner.startswith("nrp_g"):
    neural_goal_bias = 0.3
    sl_bias = 0.1
    # model_path = osp.join(CUR_DIR, "../planner/local_sampler_g/models/{}/cvae_sel.pt".format(args.env))
    model_path = osp.join(ROOT_DIR, "train/{}/models/sampler_g_01_critical_v2.pt".format(args.env))
    neural_planner_g = NRP_g(
        env,
        model_path,
        optimal="star" in args.planner,
        dim=dim,
        occ_grid_dim=occ_grid_dim,
    )
    neural_planner_g.algo.goal_bias = neural_goal_bias
    neural_planner_g.algo.add_intermediate_state = False
    neural_planner_g.sl_bias = sl_bias

# decoupled_rrt_star
if args.planner.startswith("decoupled"):
    decoupled_rrt_planner = DecoupledRRTPlanner(optimal="star" in args.planner)
    decoupled_rrt_planner.algo.goal_bias = 0.1
    hybrid_astar_planner = HybridAStar(br=base_radius, tr=turning_radius, map_res=occ_grid_resolution)
    if "is" in args.planner:
        decoupled_rrt_planner.add_intermediate_state = True

max_time = 10
time_step_size = 0.5
num_repeat = args.repeat
env_num = 10
for repeat in range(num_repeat):
    success_list = [0] * int(max_time / time_step_size)
    decoupled_success_list1 = [0] * int(max_time / time_step_size)
    decoupled_success_list2 = [0] * int(max_time / time_step_size)
    total_loop_time = 0
    total_num_expansion_called = 0
    total_neural_expansion_time = 0
    total_col_check_time = 0
    total_expansion_success = 0
    total_neural_select_success = 0
    total_neural_select_called = 0
    total_neural_expansion_called = 0
    total_neural_expansion_success = 0
    expansion_success_rate = 0
    total_vertex_selection_time = 0
    total_vertex_expansion_time = 0

    env.clear_obstacles()
    with open(os.path.join(CUR_DIR, "map/rls_occ_grid.npy"), "rb") as f:
        occ_grid = np.load(f)
    mesh_path = osp.join(CUR_DIR, "map/rls_mesh.obj")

    env.load_mesh(mesh_path)
    env.load_occupancy_grid(occ_grid, add_enclosing=True)
    # env.load_occupancy_grid(np.zeros_like(occ_grid), add_enclosing=True)

    for i in range(env_num):
        print("Loading env {}".format(i))
        # env_dir = osp.join(CUR_DIR, "../eval_env/{}".format(7)) # `18`
        # with open(osp.join(env_dir, "obstacle_dict.json")) as f:
        #     obstacle_dict = json.load(f)

        occ_grid = env.get_occupancy_grid()
        occ_grid_2d = np.any(occ_grid[:, :, : int(robot_height / occ_grid_resolution)], axis=2).astype(int)

        with open(osp.join(CUR_DIR, "test_path/test_path_{}.json".format(env_idx)), "r") as f:
            start_goal = json.load(f)
        # start_goal = json.load(osp.join(env_dir, "start_goal.json"))
        env.start = start_goal[0]
        env.goal = start_goal[1]
        env.robot.set_state(env.start)

        if args.planner.startswith("rrt"):
            log_dir = osp.join(planner_res_dir, "{}".format(i))
            if not osp.exists(log_dir):
                os.mkdir(log_dir)

            rrt_planner.algo.return_on_path_find = False
            res = rrt_planner.solve_step_time(env, env.start, env.goal, max_time, time_step_size, mesh_path)
            success_res = [tmp[0] for tmp in res]
            path_list = [tmp[1] for tmp in res]
            for idx, p in enumerate(path_list):
                # path = utils.interpolate(p)
                # utils.visualize_nodes_global(mesh_path, occ_grid, path, env.start, env.goal, show=False, save=True, file_name=osp.join(base_log_dir, "planned_path.png"))
                with open(osp.join(log_dir, "planned_path_{}_{}.json".format(repeat, idx)), "w") as f:
                    json.dump(p, f)

            for idx, res in enumerate(success_res):
                if res:
                    success_list[idx] += 1

            # path_length = utils.calc_path_len(path)
            total_loop_time += rrt_planner.algo.total_running_time / (rrt_planner.algo.num_expansions + 1e-8)
            total_col_check_time += rrt_planner.col_check_time / (rrt_planner.algo.num_expansions + 1e-8)
            total_expansion_success += rrt_planner.algo.num_expansions - rrt_planner.expansion_col_cnt
            total_num_expansion_called += rrt_planner.algo.num_expansions

        elif args.planner.startswith("bit"):
            log_dir = osp.join(planner_res_dir, "{}".format(i))
            if not osp.exists(log_dir):
                os.mkdir(log_dir)

            res = bit_planner.solve_step_time(env, env.start, env.goal, max_time, time_step_size)
            success_res = [tmp[0] for tmp in res]
            path_list = [tmp[1] for tmp in res]
            for idx, p in enumerate(path_list):
                # path = utils.interpolate(p)
                # utils.visualize_nodes_global(mesh_path, occ_grid, path, env.start, env.goal, show=False, save=True, file_name=osp.join(base_log_dir, "planned_path.png"))
                with open(osp.join(log_dir, "planned_path_{}_{}.json".format(repeat, idx)), "w") as f:
                    json.dump(p, f)

            for idx, res in enumerate(success_res):
                if res:
                    success_list[idx] += 1

            # path_length = utils.calc_path_len(path)
            total_loop_time += bit_planner.total_running_time / (bit_planner.loop_cnt + 1e-8)
            total_col_check_time += bit_planner.col_check_time / (bit_planner.num_col_check + 1e-8)
            total_expansion_success += bit_planner.num_col_check - bit_planner.num_edge_col
            total_num_expansion_called += bit_planner.loop_cnt

        elif args.planner.startswith("nrp_d"):
            # Get robot bounds
            low_bounds = env.robot.get_joint_lower_bounds()
            high_bounds = env.robot.get_joint_higher_bounds()
            low_bounds[0] = -2
            low_bounds[1] = -2
            high_bounds[0] = 2
            high_bounds[1] = 2
            # print(low_bounds, high_bounds)
            neural_planner_d.sampler.set_sample_bounds(low_bounds, high_bounds)

            log_dir = osp.join(planner_res_dir, "{}".format(i))
            if not osp.exists(log_dir):
                os.mkdir(log_dir)

            neural_planner_d.algo.return_on_path_find = False
            res = neural_planner_d.solve_step_time(env, env.start, env.goal, max_time, time_step_size)
            success_res = [tmp[0] for tmp in res]
            path_list = [tmp[1] for tmp in res]
            for idx, p in enumerate(path_list):
                # path = utils.interpolate(p)
                # utils.visualize_nodes_global(mesh_path, occ_grid, path, env.start, env.goal, show=False, save=True, file_name=osp.join(base_log_dir, "planned_path.png"))
                with open(osp.join(log_dir, "planned_path_{}_{}.json".format(repeat, idx)), "w") as f:
                    json.dump(p, f)

            for idx, res in enumerate(success_res):
                if res:
                    success_list[idx] += 1

            total_loop_time += neural_planner_d.algo.total_running_time / (neural_planner_d.algo.num_expansions + 1e-8)
            total_neural_expansion_time += neural_planner_d.neural_expansion_time / (neural_planner_d.neural_expansion_cnt + 1e-8)
            total_col_check_time += neural_planner_d.col_check_time / (neural_planner_d.neural_expansion_cnt + 1e-8)
            total_expansion_success += neural_planner_d.algo.num_expansions - neural_planner_d.expansion_col_cnt
            total_num_expansion_called += neural_planner_d.algo.num_expansions
            total_neural_expansion_success += neural_planner_d.neural_expansion_cnt - neural_planner_d.neural_expansion_col_cnt
            total_neural_expansion_called += neural_planner_d.neural_expansion_cnt
            print(neural_planner_d.algo.num_expansions)

        elif args.planner.startswith("nrp_g"):
            log_dir = osp.join(planner_res_dir, "{}".format(i))
            if not osp.exists(log_dir):
                os.mkdir(log_dir)

            neural_planner_g.algo.return_on_path_find = False
            res = neural_planner_g.solve_step_time(env, env.start, env.goal, max_time, time_step_size)
            success_res = [tmp[0] for tmp in res]
            path_list = [tmp[1] for tmp in res]
            for idx, p in enumerate(path_list):
                # path = utils.interpolate(p)
                # utils.visualize_nodes_global(mesh_path, occ_grid, path, env.start, env.goal, show=False, save=True, file_name=osp.join(base_log_dir, "planned_path.png"))
                with open(osp.join(log_dir, "planned_path_{}_{}.json".format(repeat, idx)), "w") as f:
                    json.dump(p, f)

            for idx, res in enumerate(success_res):
                if res:
                    success_list[idx] += 1

            # print(neural_planner_g.extension_col_cnt, neural_planner_g.selector_col_cnt)

            total_loop_time += neural_planner_g.algo.total_running_time / (neural_planner_g.algo.num_expansions + 1e-8)
            total_neural_expansion_time += neural_planner_g.neural_expansion_time / (neural_planner_g.neural_expansion_cnt + 1e-8)
            total_col_check_time += neural_planner_g.col_check_time / (neural_planner_g.neural_expansion_cnt + 1e-8)
            total_expansion_success += neural_planner_g.algo.num_expansions - neural_planner_g.expansion_col_cnt
            total_num_expansion_called += neural_planner_g.algo.num_expansions
            total_neural_expansion_success += neural_planner_g.neural_expansion_cnt - neural_planner_g.neural_expansion_col_cnt
            total_neural_expansion_called += neural_planner_g.neural_expansion_cnt

        elif args.planner.startswith("decoupled"):
            log_dir = osp.join(planner_res_dir, "{}".format(i))
            if not osp.exists(log_dir):
                os.mkdir(log_dir)

            decoupled_rrt_planner.algo.return_on_path_find = False
            start_stowed = env.start.copy()
            start_stowed[3:] = robot_rest_position
            goal_stowed = env.goal.copy()
            goal_stowed[3:] = robot_rest_position
            print(start_stowed[:3], goal_stowed[:3])
            astar_start = (
                start_stowed[0] / occ_grid_resolution,
                start_stowed[1] / occ_grid_resolution,
                start_stowed[2],
            )
            astar_goal = (
                goal_stowed[0] / occ_grid_resolution,
                goal_stowed[1] / occ_grid_resolution,
                goal_stowed[2],
            )
            res_2d, astar_time = hybrid_astar_planner.plan(astar_start, astar_goal, occ_grid_2d)

            # Astar should always succeed
            print("Astar time:", astar_time)
            assert res_2d is not None

            res1 = decoupled_rrt_planner.solve_step_time(
                env, start_stowed, env.start, max_time, time_step_size, mesh_path, reverse=True
            )

            total_loop_time += decoupled_rrt_planner.algo.total_running_time
            total_col_check_time += decoupled_rrt_planner.col_check_time / (
                decoupled_rrt_planner.algo.num_extensions + 1e-8
            )
            total_vertex_selection_time += decoupled_rrt_planner.algo.total_vertex_selection_time
            total_vertex_expansion_time += decoupled_rrt_planner.algo.total_vertex_extension_time
            total_expansion_success += (
                decoupled_rrt_planner.algo.num_extensions - decoupled_rrt_planner.extension_col_cnt
            )
            total_num_expansion_called += decoupled_rrt_planner.algo.num_extensions

            res2 = decoupled_rrt_planner.solve_step_time(
                env, goal_stowed, env.goal, max_time, time_step_size, mesh_path
            )

            total_loop_time += decoupled_rrt_planner.algo.total_running_time
            total_col_check_time += decoupled_rrt_planner.col_check_time / (
                decoupled_rrt_planner.algo.num_extensions + 1e-8
            )
            total_vertex_selection_time += decoupled_rrt_planner.algo.total_vertex_selection_time
            total_vertex_expansion_time += decoupled_rrt_planner.algo.total_vertex_extension_time
            total_expansion_success += (
                decoupled_rrt_planner.algo.num_extensions - decoupled_rrt_planner.extension_col_cnt
            )
            total_num_expansion_called += decoupled_rrt_planner.algo.num_extensions

            res_astar = []
            for i in range(len(res_2d[0])):
                res_astar.append(tuple([res_2d[0][i] / 10, res_2d[1][i] / 10, res_2d[2][i]] + robot_rest_position))
            success_res = [(tmp1[0] and tmp2[0]) for tmp1, tmp2 in zip(res1, res2)]
            success_res1 = [tmp[0] for tmp in res1]
            success_res2 = [tmp[0] for tmp in res2]
            path_list = []
            for i in range(len(res1)):
                if success_res[i]:
                    path_list.append(res1[i][1] + res_astar + res2[i][1])
                else:
                    path_list.append([])
            # path_list = [tmp1[1] + res_astar + tmp2[1] for tmp1, tmp2 in zip(res1, res2)]
            for idx, p in enumerate(path_list):
                # path = utils.interpolate(p)
                # utils.visualize_nodes_global(mesh_path, occ_grid, p, env.start, env.goal, show=False, save=True, file_name="planned_path.png")
                with open(osp.join(log_dir, "planned_path_{}_{}.json".format(repeat, idx)), "w") as f:
                    json.dump(p, f)

            for idx, res in enumerate(success_res):
                if res:
                    success_list[idx] += 1

            for idx, res in enumerate(success_res1):
                if res:
                    decoupled_success_list1[idx] += 1
            for idx, res in enumerate(success_res2):
                if res:
                    decoupled_success_list2[idx] += 1

            # path_length = utils.calc_path_len(path)

        print("base_success_list", success_list)
        print("base_success_list1", decoupled_success_list1)
        print("base_success_list2", decoupled_success_list2)
        print("success_list", success_list)

    # timing
    avg_total_expand_time = total_loop_time / float(env_num)
    avg_neural_expand_time = total_neural_expansion_time / float(env_num)
    avg_col_check_time = total_col_check_time / float(env_num)
    print(
        "Average total sampling time for learnt: {}, neural: {}, col: {}".format(
            avg_total_expand_time, avg_neural_expand_time, avg_col_check_time
        )
    )

    neural_expand_success_rate = total_neural_expansion_success / float(total_neural_expansion_called + 1e-8)
    print("neural expand success rate: {}".format(neural_expand_success_rate))

    # neural_select_success_rate = total_neural_select_success / float(total_neural_select_called + 1e-8)
    # print("neural select success rate: {}".format(neural_select_success_rate))

    expand_success_rate = total_expansion_success / float(total_num_expansion_called + 1e-8)
    print("expand_success_rate : {}".format(expand_success_rate))

    avg_num_expand_called = total_num_expansion_called / (env_num)
    print("avg_num_expand_called: {}".format(avg_num_expand_called))

    print("total_neural_expand_cnt", total_neural_select_called / (env_num))

    # avg_offline_process_time = total_offline_process_time / (env_num)
    # print("avg_offline_process_time: {}".format(avg_offline_process_time))

    print("success_list", success_list)

    res = {
        "avg_total_expand_time": avg_total_expand_time,
        "avg_neural_expand_time": avg_neural_expand_time,
        "avg_col_check_time": avg_col_check_time,
        "neural_expand_success_rate": neural_expand_success_rate,
        # "neural_select_success_rate": neural_select_success_rate,
        "expand_success_rate": expand_success_rate,
        "avg_num_expand_called": avg_num_expand_called,
        "success_list": success_list,
        "total_neural_expand_cnt": total_neural_select_called / (env_num),
    }

    with open(osp.join(planner_res_dir, "result_{}.json".format(repeat)), "w") as f:
        json.dump(res, f)
