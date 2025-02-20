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
    leaf_id = search_tree.states.shape[0] - 1

    path = [search_tree.states[leaf_id].tolist()]
    id = leaf_id
    while id:
        parent_id = search_tree.rewired_parents[id]
        if parent_id:
            path.append(search_tree.states[parent_id].tolist())

        id = parent_id

    path.append(search_tree.non_terminal_states[0].tolist())  # append the init state
    path.reverse()

    return path

def solve_step_extension(env, model, max_extensions, step_size, sl_bias=0.1):
    search_tree = SearchTree(env=env, root=env.init_state, model=model, dim=env.dim)

    res = []
    i = 0
    for _ in range(step_size, max_extensions + 1, step_size):
        search_tree, success = NEXT_plan(env, model, g_explore_eps=sl_bias, T = step_size, search_tree=search_tree)
        path = extract_path(search_tree)
        res.append((success, path))
        i += 1
    return res

def solve_step_time(env, model, max_time, step_size, sl_bias=0.1):
    search_tree = SearchTree(env=env, root=env.init_state, model=model, dim=env.dim)

    res = []
    i = 0
    for _ in range(step_size, max_time + 1, step_size):
        search_tree, success = NEXT_plan(env, model, g_explore_eps=sl_bias, max_allowed_time=step_size, search_tree=search_tree)
        path = extract_path(search_tree)
        res.append((success, path))
        i += 1
    return res

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("--name", default="sl_bias_test")
parser.add_argument('--planner', default='next')
parser.add_argument('--type', default='ext')
args = parser.parse_args()

# Constatns
maze_dir = osp.join(CUR_DIR, "../dataset/test_env")
model_path = osp.join(CUR_DIR, "models/next_v3.pt")
# best_model_path = osp.join(CUR_DIR, "models/next_v2_best.pt")
planner_res_dir = osp.join(CUR_DIR, "../planner/res/{}_{}/{}".format(args.name, args.type, args.planner))
if not osp.exists(planner_res_dir):
    os.makedirs(planner_res_dir)

max_extension = 300
ext_step_size = 25
max_time = 3
time_step_size = 0.2
max_sl_bias = 100
sl_bias_step_size = 10

# Hyperparameters:
visualize = False
cuda = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UCB_type = "kde"
robot_dim = 8
occ_grid_dim = 100

env = MyMazeEnv(robot_dim, maze_dir, test=True)
model = Model(env, cuda=cuda, dim=robot_dim, env_width=occ_grid_dim)
model.net.load_state_dict(torch.load(model_path))

success_rate = 0
success_list = [[0 for _ in range(15)] for _ in range(max_sl_bias // sl_bias_step_size + 1)]

test_num = 250
for env_idx in range(test_num):
    p_res_dir = osp.join(planner_res_dir, "{}".format(env_idx))
    if not osp.exists(p_res_dir):
        os.mkdir(p_res_dir)

    model.net.eval()
    problem = env.init_new_problem(use_start_goal=True)
    model.set_problem(problem)

    g_explore_eps = 0.1

    # Get path
    print("Planning... with explore_eps: {}".format(g_explore_eps))
    path = None

    res = []
    for sl_bias in range(0, max_sl_bias + 1, sl_bias_step_size):
        sl_bias = float(sl_bias) / 100
        if args.type == "ext":
            r = solve_step_extension(env, model, max_extension, ext_step_size, sl_bias)
        elif args.type == "time":
            r = solve_step_time(env, model, max_time, time_step_size, sl_bias)
        success_res = [tmp[0] for tmp in r]
        res.append(success_res)

    for idx1 in range(len(res)):
        for idx2 in range(len(res[idx1])):
            if res[idx1][idx2]:
                success_list[idx1][idx2] += 1
    print(success_list)

res = {"success_list": success_list}
with open(osp.join(planner_res_dir, "result.json"), "w") as f:
    json.dump(res, f)