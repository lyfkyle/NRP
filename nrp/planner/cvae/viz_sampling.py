import os.path as osp
import sys
import os
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))

import torch
import networkx as nx
import random
import numpy as np
import math
from pathlib import Path
import torch.multiprocessing as mp
import json

import utils
from env.maze import Maze
from implicit.model import VAEInference

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

def run_sampler(generator, occ_grid, start_pos, goal_pos, num_of_sample=100):
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
        samples = generator(num_of_sample, occ_grid_t, context_t)[:, :robot_dim]
        print(samples.shape)

        samples = samples.cpu().numpy().tolist()

    return samples

def collect_gt(train_env_dirs, env_idx, env_obj_dict, num_samples_per_env=10):
    maze_dir = train_env_dirs[env_idx]
    maze = Maze(gui=False)

    # print("Evaluation on {}".format(maze_dir))

    occ_grid = np.loadtxt(osp.join(maze_dir, "occ_grid_small.txt")).astype(np.uint8)
    orig_G = nx.read_graphml(osp.join(maze_dir, "dense_g_small.graphml"))

    maze.clear_obstacles()
    maze.load_mesh(osp.join(maze_dir, "env_small.obj"))
    maze.load_occupancy_grid(occ_grid)

    idx = 0
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
        global_samples = run_sampler(generator, occ_grid, global_start_pos, global_goal_pos)
        v_pos = [utils.unnormalize_state(v) for v in global_samples]

        if idx % 10 == 0:
            if DISCRIMINATOR:
                utils.visualize_nodes_global(occ_grid, v_pos, start_pos, goal_pos, show=False, save=True, file_name=osp.join(CUR_DIR, "eval_res/viz_sampling/{}_{}.png".format(env_idx, idx)))
            else:
                utils.visualize_nodes_global(occ_grid, v_pos, start_pos, goal_pos, show=False, save=True, file_name=osp.join(CUR_DIR, "eval_res/viz_sampling/{}_{}.png".format(env_idx, idx)))

        idx += 1

    return

if __name__ == '__main__':
    # constants
    model_path = osp.join(CUR_DIR, "../implicit/models/implicit_sel_global.pt")

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

    z_dim = 5
    robot_dim = 8
    linkpos_dim = 12
    occ_grid_dim = 100
    state_dim = robot_dim + linkpos_dim
    goal_state_dim = robot_dim + linkpos_dim + 1
    context_dim = state_dim + goal_state_dim
    generator = VAEInference(z_dim, context_dim, state_dim)
    generator.load_state_dict(torch.load(model_path))
    generator.eval()
    generator.to(device)

    iter_num = 0
    data_cnt = 0
    fk = utils.FkTorch(device)
    train_spl = 0
    test_spl = 0

    print("----------- Collecting from train env -------------")
    # print("Collecting gt")

    manager = mp.Manager()
    env_obj_dict = manager.dict()
    for env_idx in range(len(train_env_dirs)):
        env_obj_dict[env_idx] = 0

    # test
    for i in range(25):
        collect_gt(train_env_dirs, i, env_obj_dict, 4)

    print("----------- Collecting from test env -------------")
    # manager = mp.Manager()
    # env_obj_dict = manager.dict()
    # for env_idx in range(len(test_env_dirs)):
    #     env_obj_dict[env_idx] = 0

    # # # test
    # for i in range(5):
    #     test_spl += collect_gt(test_env_dirs, i, env_obj_dict, 20)

    # test_spl /= 100
    # print(test_spl)

    all_res = {
        "train_spl": train_spl,
        "test_spl": test_spl
    }

    with open("eval_sampler_res_global.json", "w") as f:
        json.dump(all_res, f)