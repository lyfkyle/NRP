import os
import os.path as osp
import json
import numpy as np
import argparse
import networkx as nx
import torch.multiprocessing as mp
from nrp.env.fetch_11d.maze import Fetch11DEnv
from nrp.env.rls import utils
from nrp import ROOT_DIR
CUR_DIR = osp.dirname(osp.abspath(__file__))

ARM_LEN_WEIGHT = 1  # 0.125

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--name',  default='test_time')
parser.add_argument('--planner', default='decoupled_rrt_star')
args = parser.parse_args()

# res_dir = osp.join(CUR_DIR, "eval_res/expert/bit")
# viz_res_dir = osp.join(CUR_DIR, "eval_res/path_len/expert")
res_dir = osp.join(ROOT_DIR, "test/eval_res/{}/{}".format(args.name, args.planner))
viz_res_dir = osp.join(ROOT_DIR, "test/eval_res/path_len/{}/{}".format(args.name, args.planner))
if not os.path.exists(viz_res_dir):
    os.makedirs(viz_res_dir)

maze = Maze(gui=False)

def run(repeat, env_idx, path_len_prop, path_success_num):
    print("Analyzing env {}".format(env_idx))
    log_dir = osp.join(res_dir, "{}".format(env_idx))
    with open(osp.join(CUR_DIR, "test_path/test_path_{}.json".format(env_idx)), "r") as f:
        start_goal = json.load(f)

    # getting expert path len
    with open(os.path.join(CUR_DIR, "map/rls_occ_grid.npy"), 'rb') as f:
        occ_grid = np.load(f)
    mesh_path = osp.join(CUR_DIR, "map/rls_mesh.obj")
    G = utils.get_prm(osp.join(CUR_DIR, "map"))

    maze.clear_obstacles()
    maze.load_mesh(mesh_path)
    maze.load_occupancy_grid(occ_grid)

    for node in G.nodes():
        nodepos = utils.node_to_numpy(G, node)
        if not G.nodes[node]['col']:
            if np.allclose(nodepos, np.array(start_goal[0])):
                start_node = node
            if np.allclose(nodepos, np.array(start_goal[1])):
                goal_node = node

    # expert_path_len = nx.shortest_path_length(G, start_node, goal_node)
    expert_path = nx.shortest_path(G, start_node, goal_node)
    expert_path_pos = [utils.node_to_numpy(G, n) for n in expert_path]
    # expert_path_len = utils.calc_path_len(expert_path_pos, arm_len_weight=ARM_LEN_WEIGHT)
    if args.path_len == 'max':
        expert_path_len = utils.calc_path_len_max(expert_path_pos)
    else:
        expert_path_len = utils.calc_path_len_norm(expert_path_pos, arm_len_weight=ARM_LEN_WEIGHT)
    print("Expert path len:", expert_path_len)

    idx = 0
    file_path = os.path.join(log_dir, "planned_path_{}_{}.json".format(repeat, idx))
    while os.path.exists(file_path):
        with open(file_path, "r") as f:
            planned_path = json.load(f)

        # plot
        # if env_idx % 25 == 0 and len(planned_path) > 1:
        # if repeat == 0 and (idx + 1) % 10 == 0 and len(planned_path) > 1:
            # print(planned_path)``
            # interpolated_path = utils.interpolate(planned_path)
        # if idx == 19:
        #     utils.visualize_nodes_global(mesh_path, occ_grid, planned_path, start_goal[0], start_goal[1], show=False, save=True, file_name=osp.join(viz_res_dir, "planned_path_{}_{}.png".format(env_idx, idx)))


        if len(planned_path) > 1 and np.allclose(np.array(planned_path[-1]), np.array(start_goal[1])):
            # cur_path_len = utils.calc_path_len(planned_path, arm_len_weight=ARM_LEN_WEIGHT)
            if args.path_len == 'max':
                cur_path_len = utils.calc_path_len_max(planned_path)
            else:
                cur_path_len = utils.calc_path_len_norm(planned_path, arm_len_weight=ARM_LEN_WEIGHT)
            print("Curr path len:", cur_path_len)
            # return
            if cur_path_len < expert_path_len:
                print("WARNING! cur path len smaller than expert!")
            path_len_prop[idx] += min(expert_path_len / cur_path_len, 1)
            path_success_num[idx] += 1
        # else:
        #     # print("[DEBUG] Planned path:", planned_path)
        #     path_len_prop[idx] += 0
        #     path_success_num[idx] += 0

        idx += 1
        file_path = os.path.join(log_dir, "planned_path_{}_{}.json".format(repeat, idx))

env_num = 10
print(res_dir)

process_num = 50
manager = mp.Manager()
path_len_prop = manager.dict()
path_success_num = manager.dict()

# run(3, 6, path_len_prop, path_success_num)

for repeat in range(10):
    # Reset dict
    for i in range(50):
        path_len_prop[i] = 0
        path_success_num[i] = 0

    # test
    j = 0
    while j < env_num:
        processes = []
        print("Running on env {} to {}".format(j, min(env_num, j + process_num)))
        for i in range(j, min(env_num, j + process_num)):
            p = mp.Process(target=run, args=(repeat, i, path_len_prop, path_success_num), daemon=True)
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        j += process_num

    print(path_success_num)
    print(path_len_prop)

    for idx in path_len_prop.keys():
        # over success only
        # if path_success_num[idx] != 0:
        #     path_len_over_success[idx] = path_len_prop[idx] / path_success_num[idx]
        # else:
        #     path_len_over_success[idx] = 0

        path_len_prop[idx] /= env_num

    print(path_len_prop)
    with open(osp.join(viz_res_dir, "res_{}.json".format(repeat)), "w") as f:
        json.dump(path_len_prop.copy(), f)
