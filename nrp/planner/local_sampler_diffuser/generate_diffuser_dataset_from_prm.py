import os.path as osp
import sys
import os
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))
import copy

import torch
import torch.multiprocessing as mp
import pickle
import math
import random
import numpy as np
import utils
from utils import interpolate_to_fixed_horizon, clip_path, k_shortest_paths, visualize_nodes_local
import networkx as nx

import diffuser.utils as diffuser_utils

from env.maze import Maze
from pathlib import Path

CUR_DIR = osp.dirname(osp.abspath(__file__))
LOCAL_ENV_SIZE = 2.0
HORIZON = 20
NUM_OF_RANDOM_LOCAL_SAMPLE = 20
TOTAL_NUM_OF_SAMPLES = 20
OCC_GRID_DIM = 40
DIFFUSION_REPEAT = 20

JOINT_BOUNDS = torch.asarray(([2.0] * 2 + [math.radians(180)] * 6), dtype=torch.float32)

# TODO:  nagative sample with collision: generate from current diffusion model + straight line to goal

def clip_and_interpolate_path(path):
    # clip the trajectory in local range
    clipped_path = clip_path(path, local_env_size=LOCAL_ENV_SIZE)
    interpolated_path = interpolate_to_fixed_horizon(clipped_path, HORIZON)

    return interpolated_path

def generate_samples(data):
    local_occ_grid, G, start_pos, goal_pos, expert_node_path, idx = data
    samples = []

    local_start_pos = utils.global_to_local(start_pos, start_pos)
    local_goal_pos = utils.global_to_local(goal_pos, start_pos)
    g_node = expert_node_path[-1]
    s_node = expert_node_path[0]

    new_expert_path = [utils.node_to_numpy(G, n) for n in expert_node_path]
    local_expert_path = [utils.global_to_local(n, start_pos) for n in new_expert_path]
    expert_path_len = utils.calc_path_len(new_expert_path)

    expert_next_node = expert_node_path[1]
    # expert_next_node_pos = utils.node_to_numpy(G, expert_next_node)
    # local_expert_pos = utils.global_to_local(expert_next_node_pos, start_pos)

    # samples.append([local_occ_grid, local_start_pos, local_goal_pos, local_expert_path, expert_path_len, expert_path_len])
    interpolated_path = clip_and_interpolate_path(local_expert_path)
    samples.append([local_occ_grid, local_start_pos, local_goal_pos, interpolated_path, expert_path_len, expert_path_len])

    local_sample_pos = []
    for node in G.nodes():
        # node_pos = utils.node_to_numpy(G, node)
        if G.has_edge(node, s_node) and node != expert_next_node and len(local_sample_pos) < NUM_OF_RANDOM_LOCAL_SAMPLE:
            # local_sample_pos.append(utils.global_to_local(node_pos, start_pos))
            local_sample_pos.append(node)

    # append near optimal samples
    for node in local_sample_pos:
        # node_pos = utils.node_to_numpy(G, node)
        # local_v_pos = utils.global_to_local(node_pos, start_pos)
        # v_pos = utils.local_to_global(local_v_pos, start_pos)
        try:
            node_path = nx.shortest_path(G, node, g_node)
        except:
            print("No path found!! This should not happen")
            continue
        node_path.insert(0, s_node)
        path = [utils.node_to_numpy(G, node) for node in node_path]
        path_len = utils.calc_path_len(path)
        local_path = [utils.global_to_local(n, start_pos) for n in path]

        # samples.append([local_occ_grid, local_start_pos, local_goal_pos, local_path, path_len, expert_path_len])
        interpolated_path = clip_and_interpolate_path(local_path)
        samples.append([local_occ_grid, local_start_pos, local_goal_pos, interpolated_path, path_len, expert_path_len])

    # append random simples path samples
    # all_node_paths = nx.all_simple_paths(G, s_node, g_node, cutoff=len(expert_node_path))  # only return simple paths no longer than shortest path
    # for i, node_path in enumerate(all_node_paths):
    #     new_path = [utils.node_to_numpy(G, n) for n in node_path]
    #     local_path = [utils.global_to_local(n, start_pos) for n in new_path]
    #     path_len = utils.calc_path_len(new_path)

    #     samples.append([local_occ_grid, local_start_pos, local_goal_pos, local_path, path_len, expert_path_len])
    #     # interpolated_path = clip_and_interpolate_path(local_path)
    #     # samples.append([local_occ_grid, local_start_pos, local_goal_pos, interpolated_path, path_len, expert_path_len])
    #     # print(path_len)
    #     if len(samples) >= TOTAL_NUM_OF_SAMPLES:
    #         break

    return samples

def get_k_shortest_samples(data):
    local_occ_grid, G, start_pos, goal_pos, expert_node_path, idx = data
    local_start_pos = utils.global_to_local(start_pos, start_pos)
    local_goal_pos = utils.global_to_local(goal_pos, start_pos)

    g_node = expert_node_path[-1]
    s_node = expert_node_path[0]

    new_expert_path = [utils.node_to_numpy(G, n) for n in expert_node_path]
    expert_path_len = utils.calc_path_len(new_expert_path)

    samples = []

    path_lengths, clipped_paths = k_shortest_paths(G, s_node, g_node, k=TOTAL_NUM_OF_SAMPLES, local_env_size=LOCAL_ENV_SIZE)
    # idx = 0
    for path_len, clipped_path in zip(path_lengths, clipped_paths):
        interpolated_path = interpolate_to_fixed_horizon(clipped_path, horizon=HORIZON)
        samples.append([local_occ_grid, local_start_pos, local_goal_pos, interpolated_path, path_len, expert_path_len])
        # visualize_nodes_local(local_occ_grid, interpolated_path, start_pos, goal_pos, show=False, save=True, file_name=os.path.join(CUR_DIR, "dataset/viz/k_shortest_{}.png".format(idx)))
        # idx += 1
    return samples

def collect_gt(prm_dir, env_idx, data_output_dir, num_samples_per_env=100):

    idx = 0
    while idx < num_samples_per_env:
        file_path = osp.join(data_output_dir, "data_{}_{}.pkl".format(env_idx, idx))
        if os.path.exists(file_path):
            idx += 1
            print("Skipping data_{}_{}".format(env_idx, idx))
            continue

        print("Processing data_{}_{}".format(env_idx, idx))
        prm_file_path = osp.join(prm_dir, "data_{}_{}.pkl".format(env_idx, idx))

        if not os.path.exists(prm_file_path):
            print("Warning! PRM not found at {}".format(prm_file_path))
            continue

        with open(prm_file_path, 'rb') as f:
            datas = pickle.load(f)

        dataset_list = generate_samples(datas[0])

        print("total num of samples: {}".format(len(dataset_list)))
        print("{}: Adding gt {}/{}".format(env_idx, idx, num_samples_per_env))

        file_path = osp.join(data_output_dir, "data_{}_{}.pkl".format(env_idx, idx))
        with open(file_path, 'wb') as f:
            # print("Dumping to {}".format(file_path))
            pickle.dump(dataset_list, f)

        idx += 1

def generate_k_shortest_datas(prm_dir, data_dir, env_num, dataset_num):
    # all_samples = []
    # for env_idx in range(env_num):
    env_idx = env_num
    print("Starting collection for env {}".format(env_idx))
    for dataset_idx in range(dataset_num):
        file_path = osp.join(prm_dir, "data_{}_{}.pkl".format(env_idx, dataset_idx))
        new_file_path = osp.join(data_dir, "data_{}_{}.pkl".format(env_idx, dataset_idx))
        if os.path.exists(new_file_path):
            print("skipping for data_{}_{}.pkl".format(env_idx, dataset_idx))
            continue

        if not os.path.exists(file_path):
            continue

        # print(f"Opening {file_path}")

        with open(file_path, 'rb') as f:
            datas = pickle.load(f)

        # samples = generate_samples(datas[0])
        samples = get_k_shortest_samples(datas[0])
        with open(new_file_path, 'wb') as f:
            # print("Dumping to {}".format(file_path))
            pickle.dump(samples, f)

        # all_samples += samples
        print(len(samples))
        print(file_path)

    # return all_samples

def generate_straight_line_datas(prm_dir, data_dir, env_num, dataset_num, env_dirs):
    env_idx = env_num
    maze = Maze(gui=False)
    print("Starting collection for env {}".format(env_idx))
    for dataset_idx in range(dataset_num):
        file_path = osp.join(prm_dir, "data_{}_{}.pkl".format(env_idx, dataset_idx))
        new_file_path = osp.join(data_dir, "data_{}_{}.pkl".format(env_idx, dataset_idx))
        if os.path.exists(new_file_path):
            print("skipping for data_{}_{}.pkl".format(env_idx, dataset_idx))
            continue

        if not os.path.exists(file_path):
            continue

        # print(f"Opening {file_path}")

        with open(file_path, 'rb') as f:
            datas = pickle.load(f)

        local_occ_grid, G, start_pos, goal_pos, expert_node_path, env_idx = datas[0]
        local_start_pos = utils.global_to_local(start_pos, start_pos)
        local_goal_pos = utils.global_to_local(goal_pos, start_pos)

        new_expert_path = [utils.node_to_numpy(G, n) for n in expert_node_path]
        expert_path_len = utils.calc_path_len(new_expert_path)

        local_path = [local_start_pos, local_goal_pos]
        localc_path_len = utils.calc_path_len(local_path)
        interpolated_path = clip_and_interpolate_path(local_path)
        global_interpolated_path = [utils.local_to_global(local_pos, start_pos) for local_pos in interpolated_path]

        # construct maze and check collision
        maze_dir = env_dirs[env_idx]
        global_occ_grid = np.loadtxt(
            osp.join(maze_dir, "occ_grid_small.txt")).astype(np.uint8)
        maze.clear_obstacles()
        maze.load_mesh(osp.join(maze_dir, "env_small.obj"))
        maze.load_occupancy_grid(global_occ_grid)

        obj_idx = env_obj_dict[env_idx]
        tmp_mesh_file_name = "tmp_{}_{}.obj".format(env_idx, obj_idx)
        env_obj_dict[env_idx] += 1
        global_occ_grid, new_mesh_path = maze.clear_obstacles_outside_local_occ_grid(start_pos, tmp_mesh_file_name)

        collision_vector = utils.get_collision_vector(maze, global_interpolated_path)

        samples = [[local_occ_grid, local_start_pos, local_goal_pos, interpolated_path, localc_path_len, expert_path_len, collision_vector]]
        with open(new_file_path, 'wb') as f:
            # print("Dumping to {}".format(file_path))
            pickle.dump(samples, f)

        # all_samples += samples
        print("straight line:", len(samples))
        print(file_path)

def generate_diffusion_datas(prm_dir, data_dir, env_num, dataset_num, env_dirs):
    diffusion, torch_device = get_duffusion_model()
    env_idx = env_num
    maze = Maze(gui=False)
    print("Starting collection for env {}".format(env_idx))
    for dataset_idx in range(dataset_num):
        all_samples = []
        file_path = osp.join(prm_dir, "data_{}_{}.pkl".format(env_idx, dataset_idx))
        new_file_path = osp.join(data_dir, "data_{}_{}.pkl".format(env_idx, dataset_idx))
        if os.path.exists(new_file_path):
            print("skipping for data_{}_{}.pkl".format(env_idx, dataset_idx))
            continue

        if not os.path.exists(file_path):
            continue

        # print(f"Opening {file_path}")

        with open(file_path, 'rb') as f:
            datas = pickle.load(f)

        local_occ_grid, G, start_pos, goal_pos, expert_node_path, env_idx = datas[0]
        local_start_pos = utils.global_to_local(start_pos, start_pos)
        local_goal_pos = utils.global_to_local(goal_pos, start_pos)

        new_expert_path = [utils.node_to_numpy(G, n) for n in expert_node_path]
        expert_path_len = utils.calc_path_len(new_expert_path)

        local_start_pos_normed = torch.Tensor(local_start_pos).unsqueeze(0) / JOINT_BOUNDS
        cond = {0: local_start_pos_normed.repeat(DIFFUSION_REPEAT, 1).to(torch_device)}
        occ_grid = torch.Tensor(local_occ_grid).view(1, OCC_GRID_DIM, OCC_GRID_DIM).unsqueeze(0).repeat(DIFFUSION_REPEAT, 1, 1, 1).to(torch_device)
        local_goal_pos_normed = torch.Tensor(local_goal_pos) / JOINT_BOUNDS
        local_goal_direction = torch.atan2(local_goal_pos_normed[1], local_goal_pos_normed[0]).view(1)
        local_goal_pos_normed = torch.concat([local_goal_pos_normed, local_goal_direction]).unsqueeze(0).repeat(DIFFUSION_REPEAT, 1).to(torch_device)
        collision = torch.ones((DIFFUSION_REPEAT, 1), device=torch_device).to(torch.float32)

        samples = diffusion.conditional_sample(cond, occ_grid, local_goal_pos_normed, collision)
        for i in range(DIFFUSION_REPEAT):
            local_path = (samples[i, :, :].detach().cpu() * JOINT_BOUNDS).numpy()
            localc_path_len = utils.calc_path_len(local_path)
            global_path = [utils.local_to_global(local_pos, start_pos) for local_pos in local_path]

            # construct maze and check collision
            maze_dir = env_dirs[env_idx]
            global_occ_grid = np.loadtxt(
                osp.join(maze_dir, "occ_grid_small.txt")).astype(np.uint8)
            maze.clear_obstacles()
            maze.load_mesh(osp.join(maze_dir, "env_small.obj"))
            maze.load_occupancy_grid(global_occ_grid)

            obj_idx = env_obj_dict[env_idx]
            tmp_mesh_file_name = "tmp_{}_{}.obj".format(env_idx, obj_idx)
            env_obj_dict[env_idx] += 1
            global_occ_grid, new_mesh_path = maze.clear_obstacles_outside_local_occ_grid(start_pos, tmp_mesh_file_name)

            collision_vector = utils.get_collision_vector(maze, global_path)

            all_samples.append([local_occ_grid, local_start_pos, local_goal_pos, local_path, localc_path_len, expert_path_len, collision_vector])

        with open(new_file_path, 'wb') as f:
            # print("Dumping to {}".format(file_path))
            pickle.dump(all_samples, f)

        # all_samples += samples
        print("diffusion:", len(all_samples))
        print(file_path)

def get_datas_train(data_dir, env_num, dataset_num, col_label=None, only_col=False):
    all_samples = []
    for env_idx in range(env_num):
        for dataset_idx in range(dataset_num):
            samples = []

            file_path = osp.join(data_dir, "data_{}_{}.pkl".format(env_idx, dataset_idx))

            if not os.path.exists(file_path):
                continue

            # print(f"Opening {file_path}")

            with open(file_path, 'rb') as f:
                datas = pickle.load(f)

            for i, data in enumerate(datas):
                if col_label is None:
                    occ_grid, start_pos, goal_pos, path, path_len, expert_path_len, collision = data
                    # if only getting collision data, skip those without collision
                    if only_col and sum(collision) == 0:
                        continue
                else:
                    occ_grid, start_pos, goal_pos, path, path_len, expert_path_len = data
                    collision = [col_label] * HORIZON
                samples.append([occ_grid, start_pos, goal_pos, path, path_len, expert_path_len, collision])

            all_samples += samples
            print(len(samples))
            print(file_path)

    return all_samples

def get_datas_eval(data_dir, env_num, dataset_num, num_to_sample, col_label=None, only_col=False):
    all_samples = []
    for env_idx in range(env_num):
        for _ in range(num_to_sample):
            dataset_idx = random.randint(0, dataset_num)
            samples = []

            file_path = osp.join(data_dir, "data_{}_{}.pkl".format(env_idx, dataset_idx))

            if not os.path.exists(file_path):
                continue

            # print(f"Opening {file_path}")

            with open(file_path, 'rb') as f:
                datas = pickle.load(f)

            for i, data in enumerate(datas):
                if col_label is None:
                    occ_grid, start_pos, goal_pos, path, path_len, expert_path_len, collision = data
                    # if only getting collision data, skip those without collision
                    if only_col and sum(collision) == 0:
                        continue
                else:
                    occ_grid, start_pos, goal_pos, path, path_len, expert_path_len = data
                    collision = [col_label] * HORIZON
                samples.append([occ_grid, start_pos, goal_pos, path, path_len, expert_path_len, collision])

            all_samples += samples
            print(len(samples))
            print(file_path)

    return all_samples

def get_datas_test(data_dir, env_num, dataset_num, col_label=None, only_col=False):
    all_samples = []
    for env_idx in range(env_num):
        for dataset_idx in range(dataset_num):
            samples = []

            file_path = osp.join(data_dir, "data_{}_{}.pkl".format(env_idx, dataset_idx))

            if not os.path.exists(file_path):
                continue

            # print(f"Opening {file_path}")

            with open(file_path, 'rb') as f:
                datas = pickle.load(f)

            for i, data in enumerate(datas):
                if col_label is None:
                    occ_grid, start_pos, goal_pos, path, path_len, expert_path_len, collision = data
                    # if only getting collision data, skip those without collision
                    if only_col and sum(collision) == 0:
                        continue
                else:
                    occ_grid, start_pos, goal_pos, path, path_len, expert_path_len = data
                    collision = [col_label] * HORIZON
                samples.append([occ_grid, start_pos, goal_pos, path, path_len, expert_path_len, collision])

            all_samples += samples
            print(len(samples))
            print(file_path)

    return all_samples

def get_duffusion_model():
    # from weights.checkpoint_normed_small.config import Config
    from weights.checkpoint_collision_label.config import Config
    diffuser_utils.set_seed(Config.seed)
    observation_dim = 8
    action_dim = 0
    # Config.device = 'cpu'

    if Config.diffusion == 'models.GaussianInvDynDiffusion':
        transition_dim = observation_dim
    else:
        transition_dim = observation_dim + action_dim

    model_config = diffuser_utils.Config(
        Config.model,
        savepath='model_config.pkl',
        horizon=Config.horizon,
        transition_dim=transition_dim,
        cond_dim=observation_dim,
        dim_mults=Config.dim_mults,
        dim=Config.dim,
        returns_condition=Config.returns_condition,
        device=Config.device,
    )

    diffusion_config = diffuser_utils.Config(
        Config.diffusion,
        savepath='diffusion_config.pkl',
        horizon=Config.horizon,
        observation_dim=observation_dim,
        action_dim=action_dim,
        n_timesteps=Config.n_diffusion_steps,
        loss_type=Config.loss_type,
        clip_denoised=Config.clip_denoised,
        predict_epsilon=Config.predict_epsilon,
        # hidden_dim=Config.hidden_dim,
        ## loss weighting
        action_weight=Config.action_weight,
        loss_weights=Config.loss_weights,
        loss_discount=Config.loss_discount,
        returns_condition=Config.returns_condition,
        device=Config.device,
        condition_guidance_w=Config.condition_guidance_w,
        inference=True,
    )

    # loadpath = os.path.join(Config.bucket, 'checkpoint_normed_small')
    loadpath = os.path.join(Config.bucket, 'checkpoint_collision_label')
    loadpath = os.path.join(loadpath, 'state_500000.pt')
    state_dict = torch.load(loadpath, map_location=Config.device)

    model = model_config()
    diffusion = diffusion_config(model)
    diffusion.load_state_dict(state_dict['ema'])

    return diffusion, Config.device


if __name__ == '__main__':

    # constants
    model_name = "model_prm"
    dataset_name = "k_shortest"
    prm_data_dir = osp.join(CUR_DIR, "./dataset/prm/{}".format(model_name))
    prm_data_dir_t = osp.join(CUR_DIR, "./dataset/prm/{}_t".format(model_name))
    compressed_data_dir = osp.join(CUR_DIR, "./dataset/{}".format(dataset_name))
    compressed_data_dir_t = osp.join(CUR_DIR, "./dataset/{}_t".format(dataset_name))
    straight_line_data_dir = osp.join(CUR_DIR, "./dataset/straight_line")
    straight_line_data_dir_t = osp.join(CUR_DIR, "./dataset/straight_line_t")
    if not os.path.exists(straight_line_data_dir):
        os.makedirs(straight_line_data_dir)
    if not os.path.exists(straight_line_data_dir_t):
        os.makedirs(straight_line_data_dir_t)

    diffusion_data_dir = osp.join(CUR_DIR, "./dataset/diffusion")
    diffusion_data_dir_t = osp.join(CUR_DIR, "./dataset/diffusion_t")
    diffusion_data_col_dir = osp.join(CUR_DIR, "./dataset/diffusion_col")
    diffusion_data_col_dir_t = osp.join(CUR_DIR, "./dataset/diffusion_col_t")
    if not os.path.exists(diffusion_data_dir):
        os.makedirs(diffusion_data_dir)
    if not os.path.exists(diffusion_data_dir_t):
        os.makedirs(diffusion_data_dir_t)
    if not os.path.exists(diffusion_data_col_dir):
        os.makedirs(diffusion_data_col_dir)
    if not os.path.exists(diffusion_data_col_dir_t):
        os.makedirs(diffusion_data_col_dir_t)

    mixed_dataset_name = "mixed"
    sel_train_data_dir = osp.join(CUR_DIR, "./dataset/{}_train".format(mixed_dataset_name))
    if not os.path.exists(sel_train_data_dir):
        os.makedirs(sel_train_data_dir)
    eval_data_dir = osp.join(CUR_DIR, "./dataset/{}_eval".format(mixed_dataset_name))
    if not os.path.exists(eval_data_dir):
        os.makedirs(eval_data_dir)
    test_data_dir = osp.join(CUR_DIR, "./dataset/{}_test".format(mixed_dataset_name))
    if not os.path.exists(test_data_dir):
        os.makedirs(test_data_dir)

    # hyperparameters
    # data_cnt = 1258258
    sel_train_data_cnt = 0
    eval_data_cnt = 0
    test_data_cnt = 0
    train_col_cnt = 0

    # diffusion, torch_device = get_duffusion_model()

    print("----------- Collecting from train env -------------")
    print("Collecting gt")
    process_num = 5
    env_num = 25
    manager = mp.Manager()

    # collect_gt(train_env_dirs, 0, env_obj_dict, 1)

    train_env_dir = osp.join(CUR_DIR, "../dataset/gibson/train")
    train_env_dirs = []
    for p in Path(train_env_dir).rglob('env_small.obj'):
        train_env_dirs.append(p.parent)
    assert len(train_env_dirs) == env_num

    env_obj_dict = manager.dict()
    for env_idx in range(len(train_env_dirs)):
        env_obj_dict[env_idx] = 0

    # for i in range(env_num):
        # generate_diffusion_datas(prm_data_dir, diffusion_data_dir, i, 400, train_env_dirs)
        # generate_diffusion_datas(prm_data_dir, diffusion_data_col_dir, i, 400, train_env_dirs)
        # generate_straight_line_datas(prm_data_dir, straight_line_data_dir, i, 400, train_env_dirs)

    j = 0
    while j < env_num:
        processes = []
        print("Running on env {} to {}".format(j, min(env_num, j + process_num)))
        for i in range(j, min(env_num, j + process_num)):
            # p = mp.Process(target=generate_k_shortest_datas, args=(prm_data_dir, compressed_data_dir, i, 400), daemon=True)
            # p = mp.Process(target=generate_straight_line_datas, args=(prm_data_dir, straight_line_data_dir, i, 400, train_env_dirs), daemon=True)
            p = mp.Process(target=generate_diffusion_datas, args=(prm_data_dir, diffusion_data_col_dir, i, 400, train_env_dirs), daemon=True)
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        j += process_num

    print("----------- Collecting from test env -------------")
    print("Collecting gt")
    process_num = 5
    test_env_num = 5
    manager = mp.Manager()

    test_env_dir = osp.join(CUR_DIR, "../dataset/gibson/mytest")
    test_env_dirs = []
    for p in Path(test_env_dir).rglob('env_small.obj'):
        test_env_dirs.append(p.parent)
    assert len(test_env_dirs) == test_env_num

    env_obj_dict = manager.dict()
    for env_idx in range(len(test_env_dirs)):
        env_obj_dict[env_idx] = 0

    # for i in range(test_env_num):
        # generate_diffusion_datas(prm_data_dir_t, diffusion_data_dir_t, i, 10, test_env_dirs)
        # generate_diffusion_datas(prm_data_dir_t, diffusion_data_col_dir_t, i, 10, test_env_dirs)
        # generate_straight_line_datas(prm_data_dir_t, straight_line_data_dir_t, i, 10, test_env_dirs)

    j = 0
    while j < test_env_num:
        processes = []
        print("Running on env {} to {}".format(j, min(test_env_num, j + process_num)))
        for i in range(j, min(test_env_num, j + process_num)):
            # p = mp.Process(target=generate_k_shortest_datas, args=(prm_data_dir_t, compressed_data_dir_t, i, 10), daemon=True)
            # p = mp.Process(target=generate_straight_line_datas, args=(prm_data_dir_t, straight_line_data_dir_t, i, 10, test_env_dirs), daemon=True)
            p = mp.Process(target=generate_diffusion_datas, args=(prm_data_dir_t, diffusion_data_col_dir_t, i, 10, test_env_dirs), daemon=True)
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        j += process_num

    # Convert compressed data into individual trajectories
    # selection
    all_samples = get_datas_train(compressed_data_dir, 25, 400, col_label=0)
    straight_line_samples = get_datas_train(straight_line_data_dir, 25, 400, col_label=None)
    all_samples += straight_line_samples
    diffusion_samples = get_datas_train(diffusion_data_dir, 25, 400, col_label=None)
    all_samples += diffusion_samples
    diffusion_col_samples = get_datas_train(diffusion_data_col_dir, 25, 400, col_label=None, only_col=True)
    all_samples += diffusion_col_samples
    print(len(all_samples))
    for sample in all_samples:
        if sum(sample[-1]) > 0:
            train_col_cnt += 1
        new_file_path = osp.join(sel_train_data_dir, "data_{}.pkl".format(sel_train_data_cnt))
        with open(new_file_path, 'wb') as f:
            # print("Dumping to {}".format(file_path))
            pickle.dump(sample, f)
        sel_train_data_cnt += 1

    # Evaluation dataset. No need to do balance
    all_samples = get_datas_eval(compressed_data_dir, 25, 400, 10, col_label=0)
    straight_line_samples = get_datas_eval(straight_line_data_dir, 25, 400, 10, col_label=None)
    all_samples += straight_line_samples
    diffusion_samples = get_datas_eval(diffusion_data_dir, 25, 400, 10, col_label=None)
    all_samples += diffusion_samples
    diffusion_col_samples = get_datas_eval(diffusion_data_col_dir, 25, 400, 10, col_label=None, only_col=True)
    all_samples += diffusion_col_samples
    print(len(all_samples))
    for sample in all_samples:
        new_file_path = osp.join(eval_data_dir, "data_{}.pkl".format(eval_data_cnt))
        with open(new_file_path, 'wb') as f:
            # print("Dumping to {}".format(file_path))
            pickle.dump(sample, f)
        eval_data_cnt += 1

    # Test dataset. No need to do balance
    all_samples = get_datas_test(compressed_data_dir_t, 5, 10, col_label=0)
    straight_line_samples = get_datas_test(straight_line_data_dir_t, 5, 10, col_label=None)
    all_samples += straight_line_samples
    diffusion_samples = get_datas_test(diffusion_data_dir_t, 5, 10, col_label=None)
    all_samples += diffusion_samples
    diffusion_col_samples = get_datas_test(diffusion_data_col_dir_t, 5, 10, col_label=None, only_col=True)
    all_samples += diffusion_col_samples
    print(len(all_samples))
    for sample in all_samples:
        new_file_path = osp.join(test_data_dir, "data_{}.pkl".format(test_data_cnt))
        with open(new_file_path, 'wb') as f:
            # print("Dumping to {}".format(file_path))
            pickle.dump(sample, f)
        test_data_cnt += 1

    print("Train: {}, Eval: {}, Test: {}".format(sel_train_data_cnt, eval_data_cnt, test_data_cnt))
    print("In train, collision: {}, no-collision: {}".format(train_col_cnt, sel_train_data_cnt - train_col_cnt))

    # mixed:
    # train: 403152, eval: 10160, test: 1969
    # In train, collision: 163555, no-collision: 239597
