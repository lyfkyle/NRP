import os
import os.path as osp

import random
from datetime import datetime
import numpy as np
import argparse
import networkx as nx
import math
import itertools
from sklearn.neighbors import NearestNeighbors
import random
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

from nrp import ROOT_DIR
from nrp.env.fetch_11d.env import Fetch11DEnv
from nrp.env.fetch_11d import utils
from nrp.env.fetch_11d import prm_utils


def convert(pos):
    pos[0] += 2
    pos[1] += 2


CUR_DIR = osp.dirname(osp.abspath(__file__))

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("--model", default="")
parser.add_argument("--env", default="fetch_11d")
parser.add_argument("--name", default="local_tree_viz")
args = parser.parse_args()

# env_name = args.env
# env_dir = osp.join(CUR_DIR, f"../env/fetch_11d/dataset/gibson/mytest/Markleeville")
# env_dir = osp.join(CUR_DIR, f"../env/fetch_11d/dataset/gibson/mytest/{env_name}")

env_num = 25
train_env_dir = osp.join(ROOT_DIR, f"env/{args.env}/dataset/gibson_01/train")
train_env_dirs = []
for p in Path(train_env_dir).rglob("env.obj"):
    train_env_dirs.append(p.parent)
assert len(train_env_dirs) == env_num
test_env_dir = osp.join(ROOT_DIR, f"env/{args.env}/dataset/gibson_01/test")
test_env_dirs = []
for p in Path(test_env_dir).rglob("env.obj"):
    test_env_dirs.append(p.parent)
assert len(test_env_dirs) == 5

output_dir = osp.join(CUR_DIR, args.name)
if not osp.exists(output_dir):
    os.makedirs(output_dir)

env = Fetch11DEnv(gui=False)

dim = 11
occ_grid_dim = [40, 40, 20]
local_env_size = 2

name = datetime.now().timestamp()
env_obj_dict = defaultdict(int)
num_local_envs = 1
num_goal_per_local_env = 1
num_env = 25
for env_idx in range(num_env):
    env_dir = train_env_dirs[env_idx]

    occ_grid = utils.get_occ_grid(env_dir)
    orig_G = utils.get_prm(env_dir)
    mesh_path = utils.get_mesh_path(env_dir)

    env.clear_obstacles()
    # env.load_occupancy_grid(occ_grid)

    data_idx = 0
    local_env_idx = 0
    while local_env_idx < num_local_envs:
        env.clear_obstacles()
        env.load_mesh(mesh_path)
        env.load_occupancy_grid(occ_grid)

        # sample start_pos
        free_nodes = utils.get_free_nodes(orig_G)
        start_node = random.choice(free_nodes)
        start_pos = utils.node_to_numpy(orig_G, start_node)
        assert env.pb_ompl_interface.is_state_valid(start_pos)

        # start_pos = utils.node_to_numpy(orig_G, start)
        # goal_pos = utils.node_to_numpy(orig_G, goal)
        # local_start_pos = utils.global_to_local(start_pos, start_pos)
        # assert math.fabs(local_goal_pos[0]) > local_env_size or math.fabs(local_goal_pos[1]) > local_env_size

        obj_idx = env_obj_dict[env_idx]
        tmp_mesh_file_name = "tmp_{}_{}.obj".format(env_idx, obj_idx)
        env_obj_dict[env_idx] += 1
        new_occ_grid, new_mesh_path = env.clear_obstacles_outside_local_occ_grid(
            start_pos, local_env_size, tmp_mesh_file_name
        )
        # utils.visualize_nodes_global(osp.join(env_dir, "env_large.obj"), occ_grid, [], None, None, show=False, save=True, file_name=osp.join(CUR_DIR, "res/global_{}_{}.png".format(env_idx, idx)))
        # utils.visualize_nodes_global(new_mesh_path, global_occ_grid, [], None, None, show=False, save=True, file_name=osp.join(CUR_DIR, "res/local_{}_{}.png".format(env_idx, idx)))

        # utils.visualize_tree(global_occ_grid, G, start_pos, goal_pos, show=False, save=True, file_name=os.path.join(CUR_DIR, "res/tree_viz.png"))
        G_without_goal, outside_nodes = prm_utils.generate_new_prm(
            orig_G,
            env,
            start_node,
            local_env_size=2,
        )

        # for theta in range(0, 360, 5):
        for goal_idx in range(num_goal_per_local_env):
            # sample goal_pos
            # while True:
            #     random_state = [0] * env.robot.num_dim
            #     for i in range(env.robot.num_dim):
            #         random_state[i] = random.uniform(low[i], high[i])
            #     if env.pb_ompl_interface.is_state_valid(random_state):
            #         goal_pos = np.array(random_state)
            #         local_goal_pos = utils.global_to_local(goal_pos, start_pos)
            #         if math.fabs(local_goal_pos[0]) > local_env_size or math.fabs(local_goal_pos[1]) > local_env_size:
            #             break
            orig_goal_node = random.choice(free_nodes)
            goal_pos = utils.node_to_numpy(orig_G, orig_goal_node)
            if not env.pb_ompl_interface.is_state_valid(goal_pos):
                utils.visualize_nodes_global(
                    new_mesh_path,
                    new_occ_grid,
                    [],
                    start_pos,
                    goal_pos,
                    show=False,
                    save=True,
                    file_name=os.path.join(
                        output_dir,
                        "invalid_goal_{}_{}_{}.png".format(env_idx, local_env_idx, goal_idx),
                    ),
                )
                utils.visualize_nodes_global(
                    mesh_path,
                    occ_grid,
                    [],
                    start_pos,
                    goal_pos,
                    show=False,
                    save=True,
                    file_name=os.path.join(
                        output_dir,
                        "invalid_goal_{}_{}_{}_original.png".format(env_idx, local_env_idx, goal_idx),
                    ),
                )
                continue

            # Generate new prm with global obstacle removed
            cur_G = G_without_goal.copy()
            cur_G, goal_node = prm_utils.add_goal_node_to_prm(cur_G, env, start_node, goal_pos, local_env_size=2)
            utils.visualize_tree_simple(
                new_occ_grid,
                cur_G,
                start_pos,
                goal_pos,
                show=False,
                save=True,
                file_name=osp.join(output_dir, f"local_tree_{env_idx}_{local_env_idx}_{goal_idx}.png"),
            )

            # Get expert path
            try:
                expert_node_path = nx.shortest_path(cur_G, start_node, goal_node)
                expert_next_node = expert_node_path[1]
                expert_next_node_pos = utils.node_to_numpy(cur_G, expert_next_node)
                local_expert_pos = utils.global_to_local(expert_next_node_pos, start_pos)
            except:
                print("No path to sampled goal")
                utils.visualize_nodes_global(
                    new_mesh_path,
                    new_occ_grid,
                    [],
                    start_pos,
                    goal_pos,
                    show=False,
                    save=True,
                    file_name=osp.join(
                        output_dir,
                        "no_path_{}_{}_{}.png".format(env_idx, local_env_idx, goal_idx),
                    ),
                )
                utils.visualize_tree_simple(
                    new_occ_grid,
                    cur_G,
                    start_pos,
                    goal_pos,
                    show=False,
                    save=True,
                    file_name=osp.join(
                        output_dir,
                        "no_path_tree_{}_{}_{}.png".format(env_idx, local_env_idx, goal_idx),
                    ),
                )
                continue

            if math.fabs(local_expert_pos[0]) > local_env_size or math.fabs(local_expert_pos[1]) > local_env_size:
                print("Local expert pos outside local environment!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                continue

            expert_path = [utils.node_to_numpy(cur_G, n) for n in expert_node_path]
            expert_wp_pos = expert_path[1]
            expert_path_dense = utils.interpolate(expert_path, step_size=0.5)
            utils.visualize_nodes_global(
                new_mesh_path,
                new_occ_grid,
                expert_path_dense,
                start_pos,
                goal_pos,
                sample_pos=expert_wp_pos,
                show=False,
                save=True,
                file_name=osp.join(
                    output_dir,
                    "expert_path_without_global_information_{}_{}_{}.png".format(env_idx, local_env_idx, goal_idx),
                ),
            )

            try:
                orig_expert_node_path = nx.shortest_path(orig_G, start_node, orig_goal_node)
                orig_expert_next_node = orig_expert_node_path[1]
                orig_expert_wp_pos = utils.node_to_numpy(orig_G, orig_expert_next_node)
                orig_expert_path = [utils.node_to_numpy(orig_G, n) for n in orig_expert_node_path]
                orig_expert_path = utils.interpolate(orig_expert_path, step_size=0.5)
                utils.visualize_nodes_global(
                    mesh_path,
                    occ_grid,
                    orig_expert_path,
                    start_pos,
                    goal_pos,
                    sample_pos=orig_expert_wp_pos,
                    show=False,
                    save=True,
                    file_name=osp.join(
                        output_dir,
                        "expert_path_with_global_information_{}_{}_{}.png".format(env_idx, local_env_idx, goal_idx),
                    ),
                )
            except:
                utils.visualize_nodes_global(
                    mesh_path,
                    occ_grid,
                    [],
                    start_pos,
                    goal_pos,
                    show=False,
                    save=True,
                    file_name=osp.join(
                        output_dir,
                        "expert_path_with_global_information_{}_{}_{}.png".format(env_idx, local_env_idx, goal_idx),
                    ),
                )

            local_occ_grid = env.get_local_occ_grid(start_pos, local_env_size=2)
            local_start_pos = utils.global_to_local(start_pos, start_pos)
            local_goal_pos = utils.global_to_local(goal_pos, start_pos)
            local_path = [utils.global_to_local(p, start_pos) for p in expert_path]
            local_wp_pos = local_path[1]
            convert(local_start_pos)
            convert(local_goal_pos)
            convert(local_wp_pos)
            utils.visualize_nodes_global(
                None,
                local_occ_grid,
                [],
                local_start_pos,
                local_goal_pos,
                local_wp_pos,
                show=False,
                save=True,
                file_name=osp.join(
                    output_dir, "expert_path_local_{}_{}_{}.png".format(env_idx, local_env_idx, goal_idx)
                ),
            )

        if os.path.exists(new_mesh_path):
            os.remove(new_mesh_path)
        assert not os.path.exists(new_mesh_path)

        local_env_idx += 1
