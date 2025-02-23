import os
import os.path as osp
import numpy as np
import argparse
import torch
import datetime
from collections import defaultdict
import json

from nrp.env.snake_8d.env import Snake8DEnv
from nrp.env.fetch_11d.env import Fetch11DEnv
from nrp.planner.bit_star import BITStar
from nrp.planner.rrt_planner import RRTPlanner
from nrp.planner.neural_planner_d import NRP_d, NRPGlobal_d
from nrp.planner.neural_planner_g import NRP_g, NRPGlobal_g
from nrp.planner.neural_planner_cvae import NeuralPlannerCVAE
from nrp.planner.neural_planner_fire import NeuralPlannerFire
from nrp.planner.fire.fire import Fire, FireEntry
from nrp import ROOT_DIR

CUR_DIR = osp.dirname(osp.abspath(__file__))

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("--name", default="")
parser.add_argument("--env", default="fetch_11d")
parser.add_argument("--testset", default="test_env")
parser.add_argument("--planner", default="nrp_g")
parser.add_argument("--logtree", action="store_true")
parser.add_argument("--drawtree", action="store_true")
parser.add_argument("--repeat", default=1, type=int)
args = parser.parse_args()
LOG_TREE = args.logtree

now = datetime.datetime.now()
if args.name == "":
    date_time = now.strftime("%m-%d-%Y-%H-%M-%S")
    res_dir = osp.join(ROOT_DIR, "results/{}/{}".format(args.env, date_time))
else:
    res_dir = osp.join(ROOT_DIR, "results/{}/{}".format(args.env, args.name))
planner_res_dir = osp.join(res_dir, args.planner)
if not osp.exists(res_dir):
    os.makedirs(res_dir)
if not osp.exists(planner_res_dir):
    os.makedirs(planner_res_dir)

# Constants
if args.env == "snake_8d":
    from nrp.env.snake_8d import utils

    env_num = 250
    env = Snake8DEnv(gui=False)
    dim = 8
    occ_grid_dim = [1, 40, 40]
    global_occ_grid_dim = [1, 100, 100]

elif args.env == "fetch_11d":
    from nrp.env.fetch_11d import utils

    env_num = 250
    env = Fetch11DEnv(gui=False)
    dim = 11
    occ_grid_dim = [40, 40, 20]
    global_occ_grid_dim = [150, 150, 20]

elif args.env == "rls":
    from nrp.env.rls import utils

    env_num = 10
    env = Fetch11DEnv(gui=False)
    dim = 11
    occ_grid_dim = [40, 40, 20]
    global_occ_grid_dim = [150, 150, 20]
    turning_radius = 0.1
    base_radius = 0.3
    robot_height = 1.1
    occ_grid_resolution = 0.1
    robot_rest_position = list(env.robot.rest_position)

# RRT planner
if args.planner.startswith("rrt"):
    rrt_planner = RRTPlanner(optimal="star" in args.planner)
    rrt_planner.algo.goal_bias = 0.1
    if "is" in args.planner:
        rrt_planner.add_intermediate_state = True

# BITstar planner
elif args.planner.startswith("bit"):
    bit_planner = BITStar(batch_size=50)

elif args.planner.startswith("cvae"):
    neural_goal_bias = 0.4
    uniform_bias = 0.5
    model_path = osp.join(ROOT_DIR, "models/cvae/{}/cvae_global.pt".format(args.env))
    cvae_planner = NeuralPlannerCVAE(
        env, model_path, optimal="star" in args.planner, dim=dim, occ_grid_dim=global_occ_grid_dim
    )
    cvae_planner.algo.goal_bias = neural_goal_bias
    cvae_planner.uniform_bias = uniform_bias

# Neural extension planner
elif args.planner.startswith("nrp_d"):
    neural_goal_bias = 0.3
    sl_bias = 0.1

    if "global" in args.planner:
        model_path = osp.join(ROOT_DIR, "models/nrp_d/{}/nrp_d_global.pt".format(args.env))
        neural_planner_d = NRPGlobal_d(
            env,
            model_path,
            optimal="star" in args.planner,
            dim=dim,
            occ_grid_dim=global_occ_grid_dim,
        )
    else:
        model_path = osp.join(ROOT_DIR, "models/nrp_d/{}/nrp_d_critical.pt".format(args.env))
        neural_planner_d = NRP_d(
            env,
            model_path,
            optimal="star" in args.planner,
            dim=dim,
            occ_grid_dim=occ_grid_dim,
        )

    neural_planner_d.algo.goal_bias = neural_goal_bias
    neural_planner_d.algo.add_intermediate_state = False
    neural_planner_d.sl_bias = sl_bias

elif args.planner.startswith("nrp_g"):
    neural_goal_bias = 0.3
    sl_bias = 0.1

    if "global" in args.planner:
        model_path = osp.join(ROOT_DIR, "models/nrp_g/{}/nrp_g_global.pt".format(args.env))
        neural_planner_g = NRPGlobal_g(
            env,
            model_path,
            optimal="star" in args.planner,
            dim=dim,
            occ_grid_dim=global_occ_grid_dim,
        )
    else:
        model_path = osp.join(ROOT_DIR, "models/nrp_g/{}/nrp_g_critical.pt".format(args.env))
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


elif args.planner.startswith("fire"):
    assert args.env == "fetch_11d", "Fire only works in fetch_11d"
    model_path = osp.join(ROOT_DIR, "models/fire/fetch_11d/fire.pt")
    fire_planner = NeuralPlannerFire(env, model_path, optimal="star" in args.planner)
    fire_planner.uniform_bias = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

max_expansion = 500
expansion_step_size = 25
num_repeat = args.repeat

for repeat in range(num_repeat):
    success_list = [0] * (max_expansion // expansion_step_size)
    success_list_by_env = defaultdict(lambda: defaultdict(int))
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
    total_offline_process_time = 0

    for i in range(env_num):
        env.clear_obstacles()
        env_dir = osp.join(ROOT_DIR, "dataset/{}/{}/{}".format(args.env, args.testset, i))
        occ_grid = utils.get_occ_grid(env_dir)
        mesh_path = utils.get_mesh_path(env_dir)
        print(mesh_path)

        env.load_mesh(mesh_path)
        env.load_occupancy_grid(occ_grid, add_enclosing=True)

        # fire_planner.sampler.synthesize_sampling_distributions()
        print("Loading env {} from {}".format(i, args.testset))

        occ_grid = env.get_occupancy_grid()
        start_goal = utils.get_start_goal(env_dir)
        env.start = start_goal[0]
        env.goal = start_goal[1]
        env.robot.set_state(env.start)

        if args.planner.startswith("rrt"):
            log_dir = osp.join(planner_res_dir, "{}".format(i))
            if not osp.exists(log_dir):
                os.mkdir(log_dir)

            # Base planner
            if LOG_TREE:
                rrt_planner.algo.log_dir = log_dir  # log tree info
                rrt_planner.log_dir = log_dir

            rrt_planner.algo.return_on_path_find = False
            res = rrt_planner.solve_step_expansion(env, env.start, env.goal, max_expansion, expansion_step_size)
            success_res = [tmp[0] for tmp in res]
            path_list = [tmp[1] for tmp in res]
            for idx, p in enumerate(path_list):
                with open(osp.join(log_dir, "planned_path_{}_{}.json".format(repeat, idx)), "w") as f:
                    json.dump(p, f)

            for idx, res in enumerate(success_res):
                if res:
                    success_list[idx] += 1
                    success_list_by_env[i % 50][idx] += 1

            # path_length = utils.calc_path_len(path)
            total_loop_time += rrt_planner.algo.total_running_time / (rrt_planner.algo.num_expansions + 1e-8)
            total_col_check_time += rrt_planner.col_check_time / (rrt_planner.algo.num_expansions + 1e-8)
            total_expansion_success += rrt_planner.algo.num_expansions - rrt_planner.expansion_col_cnt
            total_num_expansion_called += rrt_planner.algo.num_expansions

        elif args.planner.startswith("bit"):
            log_dir = osp.join(planner_res_dir, "{}".format(i))
            if not osp.exists(log_dir):
                os.mkdir(log_dir)

            res = bit_planner.solve_step_expansion(env, env.start, env.goal, max_expansion, expansion_step_size)
            success_res = [tmp[0] for tmp in res]
            path_list = [tmp[1] for tmp in res]
            for idx, p in enumerate(path_list):
                with open(osp.join(log_dir, "planned_path_{}_{}.json".format(repeat, idx)), "w") as f:
                    json.dump(p, f)

            for idx, res in enumerate(success_res):
                if res:
                    success_list[idx] += 1
                    success_list_by_env[int(i / 50)][idx] += 1

            # path_length = utils.calc_path_len(path)
            total_loop_time += bit_planner.total_running_time / (bit_planner.loop_cnt + 1e-8)
            total_col_check_time += bit_planner.col_check_time / (bit_planner.num_col_check + 1e-8)
            total_expansion_success += bit_planner.num_col_check - bit_planner.num_edge_col
            total_num_expansion_called += bit_planner.loop_cnt

        elif args.planner.startswith("cvae"):
            log_dir = osp.join(planner_res_dir, "{}".format(i))
            if not osp.exists(log_dir):
                os.mkdir(log_dir)

            cvae_planner.log_extension_info = True
            cvae_planner.algo.return_on_path_find = False
            res = cvae_planner.solve_step_expansion(
                env, env.start, env.goal, max_expansion, expansion_step_size, mesh=mesh_path
            )
            success_res = [tmp[0] for tmp in res]
            path_list = [tmp[1] for tmp in res]
            for idx, p in enumerate(path_list):
                with open(osp.join(log_dir, "planned_path_{}_{}.json".format(repeat, idx)), "w") as f:
                    json.dump(p, f)

            for idx, res in enumerate(success_res):
                if res:
                    success_list[idx] += 1
                    success_list_by_env[int(i / 50)][idx] += 1

            total_loop_time += cvae_planner.algo.total_running_time
            total_expansion_success += cvae_planner.algo.num_expansions - cvae_planner.expansion_col_cnt
            total_num_expansion_called += cvae_planner.algo.num_expansions
            # total_neural_expansion_time += cvae_planner.neural_expansion_time / (cvae_planner.neural_expansion_cnt + 1e-8)
            # total_col_check_time += cvae_planner.col_check_time / (cvae_planner.neural_expansion_cnt + 1e-8)

        elif args.planner.startswith("fire"):
            log_dir = osp.join(planner_res_dir, "{}".format(i))
            if not osp.exists(log_dir):
                os.mkdir(log_dir)

            # fire_planner.log_extension_info = True
            fire_planner.algo.return_on_path_find = False
            res = fire_planner.solve_step_expansion(
                env, env.start, env.goal, max_expansion, expansion_step_size, mesh=mesh_path
            )
            success_res = [tmp[0] for tmp in res]
            path_list = [tmp[1] for tmp in res]
            for idx, p in enumerate(path_list):
                with open(osp.join(log_dir, "planned_path_{}_{}.json".format(repeat, idx)), "w") as f:
                    json.dump(p, f)

            for idx, res in enumerate(success_res):
                if res:
                    success_list[idx] += 1
                    success_list_by_env[i % 50][idx] += 1

            total_offline_process_time += fire_planner.offline_preprocess_time
            total_loop_time += fire_planner.algo.total_running_time
            total_expansion_success += fire_planner.algo.num_expansions - fire_planner.expansion_col_cnt
            total_num_expansion_called += fire_planner.algo.num_expansions
            # total_neural_expansion_time += fire_planner.neural_expansion_time / (fire_planner.expand_cnt + 1e-8)
            # total_col_check_time += fire_planner.col_check_time / (fire_planner.expand_cnt + 1e-8)

        elif args.planner.startswith("nrp_d"):
            # Get robot bounds
            low_bounds = env.robot.get_joint_lower_bounds()
            high_bounds = env.robot.get_joint_higher_bounds()
            if "global" in args.planner:  # In global mode, we normalize before passing through neural network
                if args.env == "snake_8d":
                    low_bounds[0] = -1
                    low_bounds[1] = -1
                    high_bounds[0] = 1
                    high_bounds[1] = 1
                elif args.env == "fetch_11d":
                    low_bounds[0] = -15
                    low_bounds[1] = -15
                    high_bounds[0] = 15
                    high_bounds[1] = 15
            else:  # In local mode, we use local environment size
                low_bounds[0] = -2
                low_bounds[1] = -2
                high_bounds[0] = 2
                high_bounds[1] = 2

            neural_planner_d.sampler.set_sample_bounds(low_bounds, high_bounds)

            log_dir = osp.join(planner_res_dir, "{}".format(i))
            if not osp.exists(log_dir):
                os.mkdir(log_dir)

            neural_planner_d.algo.return_on_path_find = False
            res = neural_planner_d.solve_step_expansion(env, env.start, env.goal, max_expansion, expansion_step_size)
            success_res = [tmp[0] for tmp in res]
            path_list = [tmp[1] for tmp in res]
            for idx, p in enumerate(path_list):
                with open(osp.join(log_dir, "planned_path_{}_{}.json".format(repeat, idx)), "w") as f:
                    json.dump(p, f)

            for idx, res in enumerate(success_res):
                if res:
                    success_list[idx] += 1
                    success_list_by_env[int(i / 50)][idx] += 1

            total_loop_time += neural_planner_d.algo.total_running_time / (neural_planner_d.algo.num_expansions + 1e-8)
            total_neural_expansion_time += neural_planner_d.neural_expansion_time / (
                neural_planner_d.neural_expansion_cnt + 1e-8
            )
            total_col_check_time += neural_planner_d.col_check_time / (neural_planner_d.neural_expansion_cnt + 1e-8)
            total_expansion_success += neural_planner_d.algo.num_expansions - neural_planner_d.expansion_col_cnt
            total_num_expansion_called += neural_planner_d.algo.num_expansions
            total_neural_expansion_success += (
                neural_planner_d.neural_expansion_cnt - neural_planner_d.neural_expansion_col_cnt
            )
            total_neural_expansion_called += neural_planner_d.neural_expansion_cnt
            print(neural_planner_d.algo.num_expansions)

        elif args.planner.startswith("nrp_g"):
            log_dir = osp.join(planner_res_dir, "{}".format(i))
            if not osp.exists(log_dir):
                os.mkdir(log_dir)

            neural_planner_g.algo.return_on_path_find = False
            res = neural_planner_g.solve_step_expansion(env, env.start, env.goal, max_expansion, expansion_step_size)
            success_res = [tmp[0] for tmp in res]
            path_list = [tmp[1] for tmp in res]
            for idx, p in enumerate(path_list):
                with open(osp.join(log_dir, "planned_path_{}_{}.json".format(repeat, idx)), "w") as f:
                    json.dump(p, f)

            for idx, res in enumerate(success_res):
                if res:
                    success_list[idx] += 1
                    success_list_by_env[int(i / 50)][idx] += 1

            total_loop_time += neural_planner_g.algo.total_running_time / (neural_planner_g.algo.num_expansions + 1e-8)
            total_neural_expansion_time += neural_planner_g.neural_expansion_time / (
                neural_planner_g.neural_expansion_cnt + 1e-8
            )
            total_col_check_time += neural_planner_g.col_check_time / (neural_planner_g.neural_expansion_cnt + 1e-8)
            total_expansion_success += neural_planner_g.algo.num_expansions - neural_planner_g.expansion_col_cnt
            total_num_expansion_called += neural_planner_g.algo.num_expansions
            total_neural_expansion_success += (
                neural_planner_g.neural_expansion_cnt - neural_planner_g.neural_expansion_col_cnt
            )
            total_neural_expansion_called += neural_planner_g.neural_expansion_cnt

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

    expansion_success_rate = total_expansion_success / float(total_num_expansion_called + 1e-8)
    print("expand_success_rate : {}".format(expansion_success_rate))

    avg_num_expand_called = total_num_expansion_called / (env_num)
    print("avg_num_expand_called: {}".format(avg_num_expand_called))

    print("total_neural_expand_cnt", total_neural_select_called / (env_num))

    avg_offline_process_time = total_offline_process_time / (env_num)
    print("avg_offline_process_time: {}".format(avg_offline_process_time))

    print("success_list", success_list)

    res = {
        "avg_total_expand_time": avg_total_expand_time,
        "avg_neural_expand_time": avg_neural_expand_time,
        "avg_col_check_time": avg_col_check_time,
        "neural_expand_success_rate": neural_expand_success_rate,
        # "neural_select_success_rate": neural_select_success_rate,
        "expand_success_rate": expansion_success_rate,
        "avg_num_expand_called": avg_num_expand_called,
        "success_list": success_list,
        "success_list_by_env": success_list_by_env,
        "total_neural_expand_cnt": total_neural_select_called / (env_num),
        "avg_offline_process_time": avg_offline_process_time,
    }

    with open(osp.join(planner_res_dir, "result_{}.json".format(repeat)), "w") as f:
        json.dump(res, f)
