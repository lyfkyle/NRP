import os.path as osp
import os
import networkx as nx
import random
import numpy as np
import math
import pickle
import argparse
import torch.multiprocessing as mp
import json

from nrp.env.fetch_11d import utils
from nrp.env.fetch_11d import prm_utils
from nrp.env.fetch_11d.env import Fetch11DEnv

CUR_DIR = osp.dirname(osp.abspath(__file__))

PRM_CONNECT_RADIUS = 2.0
PROCESS_NUM = 40


def collect_gt(
    data_dir,
    train_env_dirs,
    env_idx,
    env_obj_dict,
    local_env_size,
    num_local_envs=10,
    num_goal_per_local_env=50,
):
    env_dir = train_env_dirs[env_idx]
    env = Fetch11DEnv(gui=False)

    # print("Evaluation on {}".format(env_dir))

    occ_grid = utils.get_occ_grid(env_dir)
    orig_G = utils.get_prm(env_dir)
    mesh_path = utils.get_mesh_path(env_dir)

    env.clear_obstacles()
    # env.load_occupancy_grid(occ_grid)

    # local_env_idx = 0
    with open(osp.join(env_dir, "criticality.json"), "r") as f:
        criticality = json.load(f)

    n_with_criticality = [(criticality[n], n) for n in orig_G.nodes()]
    n_with_criticality.sort(key=lambda item: item[0], reverse=True)
    critical_nodes = [x[1] for x in n_with_criticality]

    # assert num_local_envs < len(criticality)
    for local_env_idx in range(num_local_envs):
        # Check dataset sanity
        valid = True
        prev_start_pos = None
        for goal_idx in range(num_goal_per_local_env):
            file_path = osp.join(data_dir, "data_{}_{}_{}.pkl".format(env_idx, local_env_idx, goal_idx))
            if not os.path.exists(file_path):
                continue

            with open(file_path, "rb") as f:
                expert_path = pickle.load(f)

            start_pos = expert_path[0]
            if prev_start_pos is not None and tuple(start_pos) != tuple(prev_start_pos):
                valid = False
                break

            prev_start_pos = start_pos

        if prev_start_pos is None:
            valid = False

        # If it is ok, skip
        if valid:
            print("Skipping data_{}_{}".format(env_idx, local_env_idx))
            local_env_idx += 1
            continue
        # else re-generate
        else:
            print(f"Generating data {env_idx}_{local_env_idx}")
            for goal_idx in range(num_goal_per_local_env):
                file_path = osp.join(data_dir, "data_{}_{}_{}.pkl".format(env_idx, local_env_idx, goal_idx))
                if os.path.exists(file_path):
                    os.remove(file_path)

        env.clear_obstacles()
        env.load_mesh(mesh_path)
        env.load_occupancy_grid(occ_grid)

        # use critical point as start_pos
        start_node = critical_nodes[local_env_idx]
        start_pos = utils.node_to_numpy(orig_G, start_node)
        assert env.pb_ompl_interface.is_state_valid(start_pos)

        free_nodes = utils.get_free_nodes(orig_G)

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

        # Generate new prm with global obstacle removed
        G_without_goal, outside_nodes = prm_utils.generate_new_prm(
            orig_G,
            env,
            start_node,
            local_env_size=2,
        )
        # This could happen if start_node is not connectable to anything
        if not G_without_goal.has_node(start_node):
            continue

        for goal_idx in range(num_goal_per_local_env):
            file_path = osp.join(data_dir, "data_{}_{}_{}.pkl".format(env_idx, local_env_idx, goal_idx))
            if os.path.exists(file_path):
                print("Skipping data_{}_{}_{}".format(env_idx, local_env_idx, goal_idx))
                continue

            # sample goal_pos
            orig_goal_node = random.choice(free_nodes)
            goal_pos = utils.node_to_numpy(orig_G, orig_goal_node)
            if not env.pb_ompl_interface.is_state_valid(goal_pos):
                # utils.visualize_nodes_global(
                #     new_mesh_path,
                #     new_occ_grid,
                #     [],
                #     start_pos,
                #     goal_pos,
                #     show=False,
                #     save=True,
                #     file_name=os.path.join(
                #         CUR_DIR,
                #         "tmp/invalid_goal_{}_{}_{}.png".format(env_idx, local_env_idx, goal_idx),
                #     ),
                # )
                # utils.visualize_nodes_global(
                #     mesh_path,
                #     occ_grid,
                #     [],
                #     start_pos,
                #     goal_pos,
                #     show=False,
                #     save=True,
                #     file_name=os.path.join(
                #         CUR_DIR,
                #         "tmp/invalid_goal_{}_{}_{}_original.png".format(env_idx, local_env_idx, goal_idx),
                #     ),
                # )
                continue

            # Add goal to PRM
            cur_G = G_without_goal.copy()
            cur_G, goal_node = prm_utils.add_goal_node_to_prm(cur_G, env, start_node, goal_pos, local_env_size=2)

            # Get expert path
            try:
                expert_node_path = nx.shortest_path(cur_G, start_node, goal_node)
                expert_next_node = expert_node_path[1]
                expert_next_node_pos = utils.node_to_numpy(cur_G, expert_next_node)
                local_expert_pos = utils.global_to_local(expert_next_node_pos, start_pos)
            except:
                print("No path to sampled goal")
                # utils.visualize_nodes_global(
                #     new_mesh_path,
                #     new_occ_grid,
                #     [],
                #     start_pos,
                #     goal_pos,
                #     show=False,
                #     save=True,
                #     file_name=osp.join(
                #         output_dir,
                #         "no_path_{}_{}_{}.png".format(env_idx, local_env_idx, goal_idx),
                #     ),
                # )
                # utils.visualize_tree_simple(
                #     new_occ_grid,
                #     cur_G,
                #     start_pos,
                #     goal_pos,
                #     show=False,
                #     save=True,
                #     file_name=osp.join(
                #         output_dir,
                #         "no_path_tree_{}_{}_{}.png".format(env_idx, local_env_idx, goal_idx),
                #     ),
                # )
                continue

            if math.fabs(local_expert_pos[0]) > local_env_size or math.fabs(local_expert_pos[1]) > local_env_size:
                print("Local expert pos outside local environment!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                continue

            expert_path = [utils.node_to_numpy(cur_G, n) for n in expert_node_path]
            if local_env_idx < 2 and goal_idx < 2:
                expert_wp_pos = expert_path[1]
                expert_path = utils.interpolate(expert_path, step_size=0.5)
                utils.visualize_nodes_global(
                    new_mesh_path,
                    new_occ_grid,
                    expert_path,
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

            file_path = osp.join(data_dir, "data_{}_{}_{}.pkl".format(env_idx, local_env_idx, goal_idx))
            with open(file_path, "wb") as f:
                # print("Dumping to {}".format(file_path))
                pickle.dump(expert_path, f)

            nx.write_graphml(cur_G, osp.join(data_dir, f"dense_g_{env_idx}_{local_env_idx}_{goal_idx}.graphml"))

        if os.path.exists(new_mesh_path):
            os.remove(new_mesh_path)
        assert not os.path.exists(new_mesh_path)

        local_env_idx += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--env", default="fetch_11d")
    args = parser.parse_args()

    # constants
    robot_dim = 11
    linkpos_dim = 24
    occ_grid_dim = 40
    occ_grid_dim_z = 20
    goal_dim = robot_dim + linkpos_dim + 1
    model_name = "train_01_critical_v2"
    local_env_size = 2.0

    # env dirs
    train_env_dirs = utils.TRAIN_ENV_DIRS

    output_dir = osp.join(CUR_DIR, "dataset/path_dataset_viz")

    # Run in train env
    # print("----------- Collecting from train env -------------")
    print("Collecting gt")
    process_num = 25
    manager = mp.Manager()
    # dataset_dict = manager.dict()

    data_dir = osp.join(CUR_DIR, "./dataset/{}".format(model_name))
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    env_obj_dict = manager.dict()
    for env_idx in range(len(train_env_dirs)):
        env_obj_dict[env_idx] = 0

    j = 0
    while j < len(train_env_dirs):
        processes = []
        print("Running on env {} to {}".format(j, min(len(train_env_dirs), j + process_num)))
        for i in range(j, min(len(train_env_dirs), j + process_num)):
            p = mp.Process(
                target=collect_gt,
                args=(data_dir, train_env_dirs, i, env_obj_dict, local_env_size, 100),
                daemon=True,
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        j += process_num
