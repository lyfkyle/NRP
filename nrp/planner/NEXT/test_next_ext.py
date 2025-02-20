import os
import os.path as osp
import sys

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), "../"))

import torch
import os.path as osp
import argparse
import numpy as np
import json

# from env.maze_2d import Maze2D
import utils
from NEXT.model import Model
from NEXT.algorithm import NEXT_plan, RRTS_plan
from NEXT.environment.maze_env import MyMazeEnv
from NEXT.algorithm.search_tree import SearchTree

CUR_DIR = osp.dirname(osp.abspath(__file__))

def extract_path(search_tree):
    goal_id = search_tree.goal_idx
    if goal_id == -1:
        return []

    path = [search_tree.states[goal_id].tolist()]
    id = goal_id
    while id:
        parent_id = search_tree.rewired_parents[id]
        if parent_id:
            path.append(search_tree.states[parent_id].tolist())

        id = parent_id

    path.append(search_tree.non_terminal_states[0].tolist())  # append the init state
    path.reverse()

    return path

def solve_step_extension(env, model, max_extensions, step_size):
    search_tree = SearchTree(env=env, root=env.init_state, model=model, dim=env.dim)

    res = []
    i = 0
    for _ in range(step_size, max_extensions + 1, step_size):
        search_tree, success = NEXT_plan(env, model, T = step_size, search_tree=search_tree)
        path = extract_path(search_tree)
        res.append((success, path))
        i += 1
    return res

def solve_step_time(env, model, max_time, step_size):
    search_tree = SearchTree(env=env, root=env.init_state, model=model, dim=env.dim)

    res = []
    t = step_size
    while t < max_time + 1e-4:
        search_tree, success = NEXT_plan(env, model, max_allowed_time=step_size, search_tree=search_tree)
        path = extract_path(search_tree)
        res.append((success, path))
        t += step_size
    return res

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("--name", default="next")
parser.add_argument("--checkpoint", default="next_v3")
args = parser.parse_args()

# Constatns
maze_dir = osp.join(CUR_DIR, "../dataset/test_env")
model_path = osp.join(CUR_DIR, "models/next_v3.pt")
# best_model_path = osp.join(CUR_DIR, "models/next_v2_best.pt")
res_dir = osp.join(CUR_DIR, "../planner/eval_res/test_ext_500/{}".format(args.name))
if not osp.exists(res_dir):
    os.makedirs(res_dir)

max_extension = 500
ext_step_size = 25

# Hyperparameters:
visualize = False
cuda = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UCB_type = "kde"
robot_dim = 8
bs = 256
occ_grid_dim = 100
train_step_cnt = 2000
lr = 0.001
start_epoch = 0
# sigma = torch.tensor([0.5, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]).to(device)

env = MyMazeEnv(robot_dim, maze_dir, test=True)
model = Model(env, cuda=cuda, dim=robot_dim, env_width=occ_grid_dim)
mse_loss = torch.nn.MSELoss()

if args.checkpoint != "":
    print("Loading checkpoint {}.pt".format(args.checkpoint))
    model.net.load_state_dict(
        torch.load(osp.join(CUR_DIR, "models/{}.pt".format(args.checkpoint)))
    )

batch_num = 0
best_loss = float("inf")

test_num = 250
for repeat in range(10):
    success_rate = 0
    success_list = [0] * (max_extension // ext_step_size)
    for env_idx in range(test_num):
        p_res_dir = osp.join(res_dir, "{}".format(env_idx))
        if not osp.exists(p_res_dir):
            os.mkdir(p_res_dir)

        model.net.eval()
        problem = env.init_new_problem(index=env_idx, use_start_goal=True)
        model.set_problem(problem)

        g_explore_eps = 0.1

        # Get path
        print("Planning... with explore_eps: {}".format(g_explore_eps))
        path = None

        res = solve_step_extension(env, model, max_extension, ext_step_size)
        success_res = [tmp[0] for tmp in res]
        path_list = [tmp[1] for tmp in res]
        for idx, p in enumerate(path_list):
            # path = utils.interpolate(p)
            # utils.visualize_nodes_global(mesh_path, occ_grid, path, maze.start, maze.goal, show=False, save=True, file_name=osp.join(base_log_dir, "planned_path.png"))
            with open(osp.join(p_res_dir, 'planned_path_{}_{}.json'.format(repeat, idx)), 'w') as f:
                json.dump(p, f)

        for idx, res in enumerate(success_res):
            if res:
                success_list[idx] += 1

        # search_tree, done = NEXT_plan(
        #     env=env,
        #     model=model,
        #     T=300,
        #     g_explore_eps=g_explore_eps,
        #     stop_when_success=True,
        #     UCB_type=UCB_type,
        # )
        # if done:
        #     success_rate += 1
        #     path = extract_path(search_tree)

        # if path is not None:
        #     print("Get path, saving to data")
        #     print(path[0], env.init_state, path[-1], env.goal_state)
        #     # assert np.allclose(np.array(path[0]), np.array(env.init_state))
        #     # assert np.allclose(np.array(path[-1]), np.array(env.goal_state))
        #     with open('planned_path.json', 'w') as f:
        #         json.dump(path, f)
        #     # path_tmp = utils.interpolate(path)
        #     # utils.visualize_nodes_global(
        #     #     env.map,
        #     #     path_tmp,
        #     #     env.init_state,
        #     #     env.goal_state,
        #     #     show=False,
        #     #     save=True,
        #     #     file_name=osp.join(p_res_dir, "next_path.png")
        #     # )

    print(success_list)

    res = {
        "success_list": success_list,
    }

    with open(osp.join(res_dir, "result_{}.json".format(repeat)), "w") as f:
        json.dump(res, f)