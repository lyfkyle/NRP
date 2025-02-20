import os.path as osp
import sys
import os
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../../'))

import torch
import networkx as nx
import random
import numpy as np
import math
from pathlib import Path
import torch.multiprocessing as mp
import json
import argparse

from planner.local_sampler_d.model_8d import DiscriminativeSampler
from planner.local_sampler_g.model_8d import VAEInference
from env.snake_8d import utils
from env.snake_8d.maze import Snake8DEnv

CUR_DIR = osp.dirname(osp.abspath(__file__))

NUM_OF_RANDOM_LOCAL_SAMPLE = 100
LOCAL_ENV_SIZE = 2.0
PATH_LEN_DIFF_THRESHOLD = 2.0

DISCRIMINATOR = True

def plan_using_PRM_2(maze, orig_G, v_pos, g_node):
    G = orig_G.copy()

    free_nodes = [node for node in G.nodes() if not G.nodes[node]['col']]
    random.shuffle(free_nodes)

    # Add v_node to G
    number_of_nodes = G.number_of_nodes()
    g_pos = utils.node_to_numpy(G, g_node)
    v_node = "n{}".format(number_of_nodes + 1)
    G.add_node(v_node, coords=','.join(map(str, v_pos)), col=False)

    for node in free_nodes:
        node_pos = utils.node_to_numpy(G, node)

        # ignore edges far apart
        if math.fabs(node_pos[0] - v_pos[0]) < LOCAL_ENV_SIZE and math.fabs(node_pos[1] - v_pos[1]) < LOCAL_ENV_SIZE:
            if utils.is_edge_free(maze, v_pos, node_pos):
                G.add_edge(v_node, node, weight=utils.calc_edge_len(v_pos, node_pos))

    try:
        node_path = nx.shortest_path(G, v_node, g_node)
        path = [utils.node_to_numpy(G, node) for node in node_path]
    except:
        print("No path found!!")
        node_path = None
        path = None

    return G, v_node, node_path, path

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
    G.add_node(v_node, coords=','.join(map(str, v_pos)), col=False)
    G.add_edge(s_node, v_node, weight=utils.calc_edge_len(v_pos, s_pos))

    # Connect v_node to nearby nodes
    s_cnt = 0
    for node in free_nodes:
        if node == s_node:
            continue

        node_pos = utils.node_to_numpy(G, node)

        # ignore edges far apart
        if math.fabs(node_pos[0] - v_pos[0]) < LOCAL_ENV_SIZE and math.fabs(node_pos[1] - v_pos[1]) < LOCAL_ENV_SIZE:
            if utils.is_edge_free(maze, v_pos, node_pos):
                G.add_edge(v_node, node, weight=utils.calc_edge_len(v_pos, node_pos))

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
    expert_waypoint_to_goal = expert_path[1:] # exclude start
    expert_path_len = utils.calc_path_len(expert_waypoint_to_goal)
    # expert_path_base_len = utils.calc_path_len_base(expert_waypoint_to_goal)

    # make sure the selected position is far enough way from start
    # dist_expert_to_start = utils.calc_edge_len(start_pos, expert_path[1])
    # dist_selected_pos_to_start = utils.calc_edge_len(start_pos, selected_pos)
    # if dist_selected_pos_to_start - dist_expert_to_start < -0.5:
    #     return True, None, None

    # print("expert_path:", expert_waypont_to_goal)
    # print("expert_path_len:", expert_path_len)
    # utils.visualize_nodes_global(global_occ_grid, expert_path, start_pos, g_pos, show=False, save=True, file_name=osp.join(CUR_DIR, "res/gt_path_to_boundary_{}.png".format(i)))

    selected_node_path, selected_path = plan_using_PRM(maze, G, selected_pos, s_node, g_node)
    if selected_path  is None:  # there is no path from selected_pos to goal
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

def run_selector_d(maze, selector, occ_grid, start_pos, goal_pos):
    high = maze.robot.get_joint_higher_bounds()
    high[0] = 5
    high[1] = 5
    low = maze.robot.get_joint_lower_bounds()
    low[0] = -5
    low[1] = -5
    low = torch.tensor(low, device=device)
    high = torch.tensor(high, device=device)

    with torch.no_grad():
        samples = torch.rand((1250, robot_dim), device=device, dtype=torch.float)
        samples_t = samples * (high - low) + low

        occ_grid_t = torch.tensor(occ_grid, device=device, dtype=torch.float).view(1, occ_grid_dim, occ_grid_dim)
        start_t = torch.tensor(start_pos, device=device, dtype=torch.float)
        goal_t = torch.tensor(goal_pos, device=device, dtype=torch.float)

        linkinfo = fk.get_link_positions(start_t.view(1, -1)).view(-1)
        start_t = torch.cat((start_t, linkinfo))
        linkinfo = fk.get_link_positions(goal_t.view(1, -1)).view(-1)
        goal_direction = torch.atan2(goal_t[1], goal_t[0]).view(1)
        goal_t = torch.cat((goal_t, linkinfo, goal_direction))
        linkinfo = fk.get_link_positions(samples_t)
        samples_t = torch.cat((samples_t, linkinfo), dim=-1)

        occ_grid_batch = occ_grid_t.unsqueeze(0) # 1 x 4 x occ_grid_dim x occ_grid_dim x occ_grid_dim_z
        start_batch = start_t.unsqueeze(0) # 1 x dim
        goal_batch = goal_t.unsqueeze(0) # 1 x dim
        samples_batch = samples_t.unsqueeze(0) # 1 x N x dim
        # _, sel_scores = selector(occ_grid_batch, start_batch, goal_batch, samples_batch, fixed_env=True)
        sel_scores = selector.get_sel_score(occ_grid_batch, start_batch, goal_batch, samples_batch).view(-1)
        sel_scores = sel_scores.view(-1)

        selected_idx = torch.argmax(sel_scores)
        selected_sample = samples_t[selected_idx][:robot_dim].cpu().numpy().tolist()

    return selected_sample

def run_selector_g(maze, generator, occ_grid, start_pos, goal_pos):
    with torch.no_grad():
        # convert to GPU
        start = torch.tensor(start_pos, device=device, dtype=torch.float)
        occ_grid_t = torch.tensor(occ_grid, device=device, dtype=torch.float).view(1, occ_grid_dim, occ_grid_dim)
        goal = torch.tensor(goal_pos, device=device, dtype=torch.float)

        # select
        tmp = torch.cat((start.view(1, -1), goal.view(1,-1)), dim=0)
        all_linkpos = fk.get_link_positions(tmp)
        start_linkpos = all_linkpos[0].view(-1)
        goal_linkpos = all_linkpos[1].view(-1)
        start_t = torch.cat((start, start_linkpos))

        goal_direction = torch.atan2(goal[1], goal[0]).view(1)
        goal_t = torch.cat((goal, goal_linkpos, goal_direction))
        context_t = torch.cat((start_t, goal_t), dim=-1)
        samples = generator(1, occ_grid_t, context_t)[:, :robot_dim]

        selected_sample = samples[0].cpu().numpy().tolist()

    return selected_sample

def collect_gt(train_env_dirs, env_idx, env_obj_dict, num_samples_per_env=10):
    maze_dir = train_env_dirs[env_idx]
    maze = Snake8DEnv(gui=False)

    # print("Evaluation on {}".format(maze_dir))

    occ_grid = np.loadtxt(osp.join(maze_dir, "occ_grid_small.txt")).astype(np.uint8)
    orig_G = nx.read_graphml(osp.join(maze_dir, "dense_g_small.graphml.xml"))

    maze.clear_obstacles()
    maze.load_mesh(osp.join(maze_dir, "env_small.obj"))
    maze.load_occupancy_grid(occ_grid)

    idx = 0
    spl = 0
    while idx < num_samples_per_env:
        print("{}: Sampling start and goal..".format(idx))
        high = maze.robot.get_joint_higher_bounds()
        low = maze.robot.get_joint_lower_bounds()
        # sample v_pos
        while True:
            random_state = [0] * maze.robot.num_dim
            for i in range(maze.robot.num_dim):
                random_state[i] = random.uniform(low[i], high[i])
            if maze.pb_ompl_interface.is_state_valid(random_state):
                start_pos = np.array(random_state)
                break

        # sample goal position
        free_nodes = [node for node in orig_G.nodes() if not orig_G.nodes[node]['col']]
        g_node = random.choice(free_nodes)
        goal_pos = utils.node_to_numpy(orig_G, g_node)

        # Get expert path
        G, s_node, expert_node_path, path = plan_using_PRM_2(maze, orig_G, start_pos, g_node)
        if expert_node_path is None:
            print("not path exists between sampled start and goal position")
            continue

        print("Calling selector..")
        global_start_pos = utils.normalize_state(start_pos)
        global_goal_pos = utils.normalize_state(goal_pos)
        if DISCRIMINATOR:
            global_sample = run_selector_d(maze, selector, occ_grid, global_start_pos, global_goal_pos)
        else:
            global_sample = run_selector_g(maze, generator, occ_grid, global_start_pos, global_goal_pos)
        v_pos = utils.unnormalize_state(global_sample)

        # if idx % 10 == 0:
        #     if DISCRIMINATOR:
        #         print(start_pos, goal_pos, v_pos)
        #         utils.visualize_nodes_global(occ_grid, [v_pos], start_pos, goal_pos, show=False, save=True, file_name=osp.join(CUR_DIR, "eval_res/sampling_viz/d_global/{}_{}.png".format(env_idx, idx)))
        #     else:
        #         utils.visualize_nodes_global(occ_grid, [v_pos], start_pos, goal_pos, show=False, save=True, file_name=osp.join(CUR_DIR, "eval_res/sampling_viz/g_global/{}_{}.png".format(env_idx, idx)))

        print("calculating accuracy..")
        edge_col_free = utils.is_edge_free(maze, start_pos, v_pos)
        if edge_col_free:
            selected_path_len, expert_path_len = is_path_differ(maze, G, start_pos, expert_node_path, v_pos)
            cur_spl = expert_path_len / max(selected_path_len, expert_path_len)
        else:
            cur_spl = 0

        print(cur_spl)

        spl += cur_spl
        idx += 1

    return spl

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data', default='all')
    args = parser.parse_args()

    # constants
    robot_dim = 8
    linkpos_dim = 12
    occ_grid_dim = 100
    goal_dim = robot_dim + linkpos_dim + 1

    # env dirs
    env_num = 25
    train_env_dir = osp.join(CUR_DIR, "../../env/snake_8d/dataset/gibson/train")
    train_env_dirs = []
    for p in Path(train_env_dir).rglob('env_small.obj'):
        train_env_dirs.append(p.parent)
    assert len(train_env_dirs) == env_num
    test_env_dir = osp.join(CUR_DIR, "../../env/snake_8d/dataset/gibson/mytest")
    test_env_dirs = []
    for p in Path(test_env_dir).rglob('env_small.obj'):
        test_env_dirs.append(p.parent)
    assert len(test_env_dirs) == 5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    if DISCRIMINATOR:
        col_checker_path = os.path.join(CUR_DIR, "models/snake_8d/model_col_global.pt")
        selector_path = os.path.join(CUR_DIR, "models/snake_8d/model_sel_global.pt")
        selector = DiscriminativeSampler(robot_dim, [1, 100, 100], col_checker_path, selector_path)
    else:
        model_path = os.path.join(CUR_DIR, "../local_sampler_g/models/cvae_sel_global.pt")
        z_dim = 5
        linkpos_dim = 12
        state_dim = robot_dim + linkpos_dim
        goal_state_dim = robot_dim + linkpos_dim + 1
        context_dim = state_dim + goal_state_dim
        generator = VAEInference(occ_grid_dim, z_dim, context_dim, state_dim)
        generator.load_state_dict(torch.load(model_path))
        generator.eval()
        generator.to(device)

    iter_num = 0
    data_cnt = 0
    fk = utils.FkTorch(device)
    train_spl = 0
    test_spl = 0

    if args.data == "train" or args.data == "all":
        print("----------- Collecting from train env -------------")

        manager = mp.Manager()
        env_obj_dict = manager.dict()
        for env_idx in range(len(train_env_dirs)):
            env_obj_dict[env_idx] = 0

        for i in range(25):
            train_spl += collect_gt(train_env_dirs, i, env_obj_dict, 4)

        train_spl /= 100
        print(train_spl)

    if args.data == "test" or args.data == "all":
        print("----------- Collecting from test env -------------")
        manager = mp.Manager()
        env_obj_dict = manager.dict()
        for env_idx in range(len(test_env_dirs)):
            env_obj_dict[env_idx] = 0

        for i in range(5):
            test_spl += collect_gt(test_env_dirs, i, env_obj_dict, 20)

        test_spl /= 100
        print(test_spl)

    all_res = {
        "train_spl": train_spl,
        "test_spl": test_spl
    }

    print(all_res)
    with open("eval_res/eval_sampler_res_global.json", "w") as f:
        json.dump(all_res, f)