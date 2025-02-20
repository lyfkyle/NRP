import sys
import os.path as osp
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))

from env.maze import Maze
import utils
import argparse
import json
import itertools
import torch.multiprocessing as mp
from pathlib import Path
import math
import numpy as np
import random
import networkx as nx
import torch
import os
import time
import pickle

# from path_dataset import MyDataset
import diffuser.utils as diffuser_utils

# from config.wbmp8dof_config import Config
# from weights.checkpoint_collision_label.config import Config
from weights.checkpoint_collision_array.config import Config


CUR_DIR = osp.dirname(osp.abspath(__file__))

NUM_OF_RANDOM_LOCAL_SAMPLE = 100
LOCAL_ENV_SIZE = 2.0
PATH_LEN_DIFF_THRESHOLD = 2.0
OCC_GRID_DIM = 40
JOINT_BOUNDS = torch.asarray(([2.0] * 2 + [math.radians(180)] * 6), dtype=torch.float32)

DISCRIMINATOR = False


def generate_new_prm(orig_G, maze, start_pos, goal_node, mesh=None, occ_g=None, size=LOCAL_ENV_SIZE + 1):
    dense_G = nx.create_empty_copy(orig_G)  # remove all edges
    # print(dense_G.number_of_edges(), dense_G.number_of_nodes())

    goal_pos = utils.node_to_numpy(dense_G, goal_node)
    # print(start_pos, goal_pos)

    # print("Connecting outside nodes to goal")
    outside_nodes = []
    for node in dense_G.nodes():
        node_pos = utils.node_to_numpy(dense_G, node)
        if (math.fabs(node_pos[0] - start_pos[0]) > LOCAL_ENV_SIZE or math.fabs(node_pos[1] - start_pos[1]) > LOCAL_ENV_SIZE) and \
                (math.fabs(node_pos[0] - start_pos[0]) <= size and math.fabs(node_pos[1] - start_pos[1]) <= size):
            nx.set_node_attributes(
                dense_G, {node: {"col": False, "coords": ','.join(map(str, node_pos))}})
            outside_nodes.append(node)
    # print(len(outside_nodes))

    # check valid outside nodes. path from valid outside nodes to goal should not pass through the local environment
    valid_outside_nodes = []
    for node in outside_nodes:
        node_pos = utils.node_to_numpy(dense_G, node)
        path_to_goal = utils.interpolate([goal_pos, node_pos])

        valid = True
        for p in path_to_goal:
            if math.fabs(p[0] - start_pos[0]) <= LOCAL_ENV_SIZE and math.fabs(p[1] - start_pos[1]) <= LOCAL_ENV_SIZE:
                valid = False
                break

        if valid:
            valid_outside_nodes.append(node)
            dense_G.add_edge(
                node, goal_node, weight=utils.calc_edge_len(node_pos, goal_pos))

    # print("Connecting inside nodes using the original graph")
    inside_nodes = []
    for node in dense_G.nodes():
        node_pos = utils.node_to_numpy(dense_G, node)
        if math.fabs(node_pos[0] - start_pos[0]) <= LOCAL_ENV_SIZE and math.fabs(node_pos[1] - start_pos[1]) <= LOCAL_ENV_SIZE:
            if not dense_G.nodes[node]['col']:
                inside_nodes.append(node)

    # use the original graph to connect inside nodes
    node_pairs = itertools.combinations(inside_nodes, 2)
    for node_pair in node_pairs:
        if orig_G.has_edge(node_pair[0], node_pair[1]):
            s1 = utils.node_to_numpy(dense_G, node_pair[0])
            s2 = utils.node_to_numpy(dense_G, node_pair[1])
            dense_G.add_edge(node_pair[0], node_pair[1],
                             weight=utils.calc_edge_len(s1, s2))

    # print("Connecting start_pos to inside nodes")
    s_node = "n{}".format(dense_G.number_of_nodes() + 1)
    dense_G.add_node(s_node, coords=','.join(map(str, start_pos)), col=False)
    s_cnt = 0
    for node in inside_nodes:
        node_pos = utils.node_to_numpy(dense_G, node)

        # ignore edges far apart
        if s_cnt < 50 and math.fabs(node_pos[0] - start_pos[0]) < 1.5 and math.fabs(node_pos[1] - start_pos[1]) < 1.5:
            if utils.is_edge_free(maze, start_pos, node_pos):
                dense_G.add_edge(
                    s_node, node, weight=utils.calc_edge_len(start_pos, node_pos))

            s_cnt += 1

    # print("Connecting outside nodes using the original graph")
    pairs_to_check = []
    node_pairs = itertools.combinations(outside_nodes, 2)
    for node_pair in node_pairs:
        s1 = utils.node_to_numpy(dense_G, node_pair[0])
        s2 = utils.node_to_numpy(dense_G, node_pair[1])
        if orig_G.has_edge(node_pair[0], node_pair[1]):
            dense_G.add_edge(node_pair[0], node_pair[1],
                             weight=utils.calc_edge_len(s1, s2))
        else:
            if math.fabs(s2[0] - s1[0]) < 1.0 and math.fabs(s2[1] - s1[1]) < 1.0:
                pairs_to_check.append((s1, s2, node_pair))

    # print("num edges to check = {}".format(len(pairs_to_check)))
    random.shuffle(pairs_to_check)
    if len(pairs_to_check) > 2500:
        pairs_to_check = pairs_to_check[:2500]

    for s1, s2, node_pair in pairs_to_check:
        if dense_G.has_edge(node_pair[0], node_pair[1]):
            continue
        else:
            path = utils.interpolate([s1, s2])

            valid = True
            for p in path:
                if math.fabs(p[0] - start_pos[0]) <= LOCAL_ENV_SIZE and math.fabs(p[1] - start_pos[1]) <= LOCAL_ENV_SIZE:
                    valid = False
                    break

            if valid:
                dense_G.add_edge(
                    node_pair[0], node_pair[1], weight=utils.calc_edge_len(s1, s2))

    # print("Connecting inside nodes to outside nodes")
    node_pairs_to_check = []
    for node in inside_nodes:
        for node2 in valid_outside_nodes:
            node_pairs_to_check.append([node, node2])
    random.shuffle(node_pairs_to_check)

    pairs_to_check = []
    for node_pair in node_pairs_to_check:
        if dense_G.nodes[node_pair[0]]['col'] or dense_G.nodes[node_pair[1]]['col']:
            continue

        if orig_G.has_edge(node_pair[0], node_pair[1]):
            dense_G.add_edge(node_pair[0], node_pair[1],
                             weight=utils.calc_edge_len(s1, s2))
            continue

        s1 = utils.node_to_numpy(dense_G, node_pair[0])
        s2 = utils.node_to_numpy(dense_G, node_pair[1])

        # ignore edges far apart
        if np.allclose(s1, s2) or math.fabs(s2[0] - s1[0]) > 1.0 or math.fabs(s2[1] - s1[1]) > 1.0:
            continue

        pairs_to_check.append((s1, s2, node_pair))

    # print("num edges to check = {}".format(len(pairs_to_check)))
    random.shuffle(pairs_to_check)
    if len(pairs_to_check) > 2500:
        pairs_to_check = pairs_to_check[:2500]

    for s1, s2, node_pair in pairs_to_check:
        if dense_G.has_edge(node_pair[0], node_pair[1]):
            continue
        else:
            if utils.is_edge_free(maze, s1, s2):
                dense_G.add_edge(
                    node_pair[0], node_pair[1], weight=utils.calc_edge_len(s1, s2))

    # print("Connecting expert node path")
    # for i in range(1, len(expert_node_path)):
    #     node1 = expert_node_path[i - 1]
    #     node2 = expert_node_path[i]
    #     s1 = utils.node_to_numpy(dense_G, node1)
    #     s2 = utils.node_to_numpy(dense_G, node2)
    #     dense_G.add_edge(node1, node2, weight=utils.calc_edge_len(s1, s2))

    dense_G.remove_nodes_from(list(nx.isolates(dense_G)))
    # print(dense_G.number_of_edges(), dense_G.number_of_nodes())

    return s_node, dense_G


def plan_using_PRM(maze, G, v_pos, s_node, g_node):
    free_nodes = [node for node in G.nodes() if not G.nodes[node]['col']]
    random.shuffle(free_nodes)

    # special case where v_pos is already in G.
    for node in free_nodes:
        node_pos = utils.node_to_numpy(G, node)
        if np.allclose(node_pos, v_pos):
            try:
                node_path = nx.shortest_path(G, node, g_node)
                path = [utils.node_to_numpy(G, n) for n in node_path]
            except:
                print("No path found!!")
                node_path = None
                path = None

            return node_path, path

    # Add v_node to G
    number_of_nodes = G.number_of_nodes()
    s_pos = utils.node_to_numpy(G, s_node)
    g_pos = utils.node_to_numpy(G, g_node)
    v_node = "n{}".format(number_of_nodes + 1)
    # g_node = "n{}".format(number_of_nodes + 2)
    G.add_node(v_node, coords=','.join(map(str, v_pos)), col=False)
    # G.add_node(g_node, coords=','.join(map(str, g)), col=False)

    # Connect v_node to nearby nodes
    # add connection to start_node
    # if utils.is_edge_free(maze, s_pos, v_pos):
    G.add_edge(s_node, v_node, weight=utils.calc_edge_len(v_pos, s_pos))
    # else:
    #     print("Edge not free!!")
    #     print(s_pos, v_pos)
    #     return None

    s_cnt = 0
    # g_cnt = 0
    for node in free_nodes:
        if node == s_node:
            continue

        node_pos = utils.node_to_numpy(G, node)

        # ignore edges far apart
        if math.fabs(node_pos[0] - v_pos[0]) < LOCAL_ENV_SIZE and math.fabs(node_pos[1] - v_pos[1]) < LOCAL_ENV_SIZE:
            if utils.is_edge_free(maze, v_pos, node_pos):
                G.add_edge(
                    v_node, node, weight=utils.calc_edge_len(v_pos, node_pos))

    try:
        node_path = nx.shortest_path(G, v_node, g_node)
        path = [utils.node_to_numpy(G, node) for node in node_path]
    except:
        print("No path found!!")
        path = None

    G.remove_node(v_node)

    return node_path, path


def is_path_differ(maze, G, start_pos, expert_node_path, selected_pos):
    g_node = expert_node_path[-1]
    s_node = expert_node_path[0]
    expert_path = [utils.node_to_numpy(G, node) for node in expert_node_path]
    g_pos = expert_path[-1]

    # path_to_boundary.append(g_pos)
    # path_to_boundary.pop(0)
    expert_waypoint_to_goal = expert_path[1:]  # exclude start
    expert_path_len = utils.calc_path_len(expert_waypoint_to_goal)
    # expert_path_base_len = utils.calc_path_len_base(expert_waypoint_to_goal)

    # Find boundary node
    local_expert_waypoint_to_goal = [utils.global_to_local(v, start_pos) for v in expert_waypoint_to_goal]
    local_expert_waypoint_to_goal = utils.interpolate(local_expert_waypoint_to_goal)
    for i, pos in enumerate(local_expert_waypoint_to_goal):
        if not utils.is_robot_within_local_env(pos, 2.0):  # local env size = 2.0
            break
    assert i > 0
    expert_path_len = expert_path_len * (len(local_expert_waypoint_to_goal) - (i - 1)) / len(local_expert_waypoint_to_goal)

    # make sure the selected position is far enough way from start
    # dist_expert_to_start = utils.calc_edge_len(start_pos, expert_path[1])
    # dist_selected_pos_to_start = utils.calc_edge_len(start_pos, selected_pos)
    # if dist_selected_pos_to_start - dist_expert_to_start < -0.5:
    #     return True, None, None

    # print("expert_path:", expert_waypont_to_goal)
    # print("expert_path_len:", expert_path_len)
    # utils.visualize_nodes_global(global_occ_grid, expert_path, start_pos, g_pos, show=False, save=True, file_name=osp.join(CUR_DIR, "res/gt_path_to_boundary_{}.png".format(i)))

    selected_node_path, selected_path = plan_using_PRM(
        maze, G, selected_pos, s_node, g_node)
    if selected_path is None:  # there is no path from selected_pos to goal
        print("No path from selected_pos to g_pos")
        # print(selected_pos, expert_path[0])
        assert utils.is_edge_free(maze, expert_path[0], selected_pos)
        node_path = nx.shortest_path(G, s_node, g_node)
        assert node_path is not None
        return -1, expert_path_len

    # path_to_boundary.append(g_pos)
    selected_waypont_to_goal = selected_path
    selected_path_len = utils.calc_path_len(selected_waypont_to_goal)
    # selected_path_base_len = utils.calc_path_len_base(selected_waypont_to_goal)

    # print("selected_path:", selected_waypont_to_goal)
    # print("selected_path_len:", selected_path_len)
    # print("selected_path_len_base:", selected_path_base_len)
    # utils.visualize_nodes_global(global_occ_grid, selected_waypont_to_goal[:-1], start_pos, g_pos, show=False, save=True, file_name=osp.join(CUR_DIR, "res/selected_path_to_boundary_{}_{}.png".format(i, path_differ)))

    return selected_path_len, expert_path_len


def sample_problems(G):
    # path = dict(nx.all_pairs_shortest_path(G))
    free_nodes = [n for n in G.nodes() if not G.nodes[n]['col']]
    random.shuffle(free_nodes)

    for i, s_name in enumerate(free_nodes):
        s_name = free_nodes[i]
        start_pos = utils.node_to_numpy(G, s_name).tolist()

        for g_name in free_nodes[i:]:
            goal_pos = utils.node_to_numpy(G, g_name).tolist()

            try:
                path = nx.shortest_path(G, source=s_name, target=g_name)
            except:
                continue

            # p = [utils.node_to_numpy(G, n).tolist() for n in path]
            # for x in p:
            #     x[0] += 2
            #     x[1] += 2

            if len(path) > 2 and \
                    math.fabs(goal_pos[0] - start_pos[0]) > LOCAL_ENV_SIZE and \
                    math.fabs(goal_pos[1] - start_pos[1]) > LOCAL_ENV_SIZE:
                return s_name, g_name, path

    return None, None, []


def collect_gt(test_dataset, env_dirs):
    maze = Maze(gui=False)

    # print("Evaluation on {}".format(maze_dir))
    spl = 0
    collision_count = 0
    for idx, test_data in enumerate(test_dataset):
        local_occ_grid, G, start_pos, goal_pos, expert_node_path, env_idx = test_data

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
        # local_occ_grid_from_maze = maze.get_local_occ_grid(start_pos)
        # print(np.all(local_occ_grid == local_occ_grid_from_maze))

        print("Calling diffusion..")
        local_start_pos = utils.global_to_local(start_pos, start_pos)
        local_goal_pos = utils.global_to_local(goal_pos, start_pos)
        local_start_pos_normed = utils.normalize_sample(torch.as_tensor(local_start_pos, dtype=torch.float32).unsqueeze(0), -JOINT_BOUNDS, JOINT_BOUNDS)
        conditions = {0: local_start_pos_normed.to(device)}
        occ_grid = torch.as_tensor(local_occ_grid, dtype=torch.float32, device=device).view([1, 1, OCC_GRID_DIM, OCC_GRID_DIM])
        local_goal_pos_normed = utils.normalize_sample(torch.as_tensor(local_goal_pos, dtype=torch.float32), -JOINT_BOUNDS, JOINT_BOUNDS)
        local_goal_direction = torch.atan2(local_goal_pos_normed[1], local_goal_pos_normed[0]).view(1)
        local_goal_pos_normed = torch.concat([local_goal_pos_normed, local_goal_direction]).unsqueeze(0).to(device)
        collision = torch.ones((1, Config.collision_dim), device=device)
        local_path = model.conditional_sample(conditions, occ_grid, local_goal_pos_normed, collision)
        local_path = utils.unnormalize_sample(local_path[0].cpu(), -JOINT_BOUNDS, JOINT_BOUNDS).numpy()
        global_path = [utils.local_to_global(v, start_pos) for v in local_path]
        # global_path = [start_pos]

        file_path = osp.join(CUR_DIR, "eval_res/sampling_viz/d_global/")
        if not osp.exists(file_path):
            os.makedirs(file_path)
        # if idx % 10 == 0:
        #     utils.visualize_nodes_global(global_occ_grid, global_path, start_pos, goal_pos, show=False, save=True, file_name=osp.join(
        #         file_path, "global_{}_{}.png".format(env_idx, idx)))
        #     utils.visualize_nodes_local(local_occ_grid, local_path, local_start_pos, local_goal_pos, show=False, save=True, file_name=osp.join(
        #         file_path, "local_{}_{}.png".format(env_idx, idx)))

        print("calculating accuracy..")
        # path_free = utils.is_path_free(maze, global_path)  # TODO: Always return False now
        free_path = utils.rrt_extend_path(maze, global_path)
        last_point = free_path[-1]
        selected_dist_to_goal, expert_dist_to_goal = is_path_differ(maze, G, start_pos, expert_node_path, last_point)
        cur_spl = expert_dist_to_goal / max(selected_dist_to_goal, expert_dist_to_goal)
        # cur_spl = expert_dist_to_goal / selected_dist_to_goal  # NOTE: now with diffusion, selected_dist_to_goal should be smaller than expert_dist_to_goal.

        # if path_free:
        #     last_point = global_path[-1]
        #     selected_dist_to_goal, expert_dist_to_goal = is_path_differ(maze, G, start_pos, expert_node_path, last_point)
        #     # cur_spl = expert_dist_to_goal / max(selected_dist_to_goal, expert_dist_to_goal)
        #     cur_spl = expert_dist_to_goal / selected_dist_to_goal  # NOTE: now with diffusion, selected_dist_to_goal should be smaller than expert_dist_to_goal.

        #     # TODO: A more principled way here is to select expert node as the node in the local environment that does have shortest distance to goal and calculate
        #     # expert_dist_to_goal as that.
        # else:
        #     print('PATH NOT FREE')
        #     cur_spl = 0
        #     collision_count += 1

        print('env_id: {}, idx: {}, spl: {}'.format(env_idx, idx, cur_spl))
        print('collision: {}'.format(collision_count))

        spl += cur_spl
        # idx += 1

    return spl

def get_diffusion_model():
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
        collision_dim=Config.collision_dim,
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
        returns_condition=True,  # Config.returns_condition,
        device=Config.device,
        condition_guidance_w=1.25,  # Config.condition_guidance_w,
        inference=True,
    )

    # loadpath = os.path.join(Config.bucket, 'checkpoint')
    # loadpath = os.path.join(Config.bucket, 'checkpoint_collision_label')
    loadpath = os.path.join(Config.bucket, 'checkpoint_collision_array')
    loadpath = os.path.join(loadpath, 'state_500000.pt')
    state_dict = torch.load(loadpath, map_location=Config.device)

    model = model_config()
    diffusion = diffusion_config(model)
    diffusion.load_state_dict(state_dict['ema'])

    return diffusion


if __name__ == '__main__':
    total_start_time = time.time()
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data', default='all')
    args = parser.parse_args()

    # constants
    robot_dim = 8
    linkpos_dim = 12
    occ_grid_dim = 40
    goal_dim = robot_dim + linkpos_dim + 1

    # env dirs
    env_num = 25
    train_env_dir = osp.join(CUR_DIR, "../dataset/gibson/train")
    train_env_dirs = []
    for p in Path(train_env_dir).rglob('env_small.obj'):
        train_env_dirs.append(p.parent)
    assert len(train_env_dirs) == env_num
    test_env_dir = osp.join(CUR_DIR, "../dataset/gibson/mytest")
    test_env_dirs = []
    for p in Path(test_env_dir).rglob('env_small.obj'):
        test_env_dirs.append(p.parent)
    assert len(test_env_dirs) == 5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # diffusion model
    model = get_diffusion_model()

    # Warm up
    start_time = time.time()
    for _ in range(0):
        conditions = {0: torch.zeros((10, 8), device=device)}
        occ_grid = torch.zeros((10, 1, 40, 40), device=device)
        goal_pos = torch.zeros((10, 9), device=device)
        collision = torch.zeros((10, Config.collision_dim), device=device)
        samples = model.conditional_sample(conditions, occ_grid, goal_pos, collision)
    end_time = time.time()
    print("warmup takes {}".format(end_time - start_time))

    # load the testing dataset
    # test data should contain local_occ_grid, G, start_pos, goal_pos, expert_node_path, env_idx
    dataset_num = 10
    test_env_num = 5
    data_dir = osp.join(CUR_DIR, "./dataset/prm/model_prm_t")
    test_dataset = []
    for env_idx in range(test_env_num):
        for dataset_idx in range(dataset_num):

            file_path = osp.join(data_dir, "data_{}_{}.pkl".format(env_idx, dataset_idx))

            if not os.path.exists(file_path):
                continue

            # print(f"Opening {file_path}")

            with open(file_path, 'rb') as f:
                datas = pickle.load(f)

            for i, data in enumerate(datas):
                test_dataset.append(data)

    print('size of test dataset: {}'.format(len(test_dataset)))

    iter_num = 0
    data_cnt = 0
    fk = utils.FkTorch(device)
    train_spl = 0
    test_spl = 0

    # if args.data == "train" or args.data == "all":
    #     print("----------- Collecting from train env -------------")

    #     manager = mp.Manager()
    #     env_obj_dict = manager.dict()
    #     for env_idx in range(len(train_env_dirs)):
    #         env_obj_dict[env_idx] = 0

    #     for i in range(25):
    #         train_spl += collect_gt(train_env_dirs, i, env_obj_dict, 4)

    #     train_spl /= 100
    #     print(train_spl)

    if args.data == "test" or args.data == "all":
        print("----------- Collecting from test env -------------")
        manager = mp.Manager()
        env_obj_dict = manager.dict()
        for env_idx in range(len(test_env_dirs)):
            env_obj_dict[env_idx] = 0

        # for i in range(5):
        test_spl += collect_gt(test_dataset, test_env_dirs)

        test_spl /= len(test_dataset)
        print(test_spl)

    all_res = {
        "train_spl": train_spl,
        "test_spl": test_spl
    }

    with open("eval_res/eval_sampler_res.json", "w") as f:
        json.dump(all_res, f)

    # print("time:", time.time() - total_start_time)
