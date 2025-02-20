import os
import os.path as osp
import json
import numpy as np
import argparse
import networkx as nx
import torch.multiprocessing as mp

from nrp.env.fetch_11d.maze import Fetch11DEnv
from nrp.env.snake_8d.maze import Snake8DEnv
from nrp.env.rls.rls_env import RLSEnv
from nrp import ROOT_DIR
# import utils

CUR_DIR = osp.dirname(osp.abspath(__file__))

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("--env", default="snake_8d")
parser.add_argument("--name", default="test_time")
parser.add_argument("--planner", default="rrt_star")
parser.add_argument("--repeat", default=10)
args = parser.parse_args()

if args.env == "snake_8d":
    env = Snake8DEnv(gui=False)
    env_num = 250
    process_num = 50
elif args.env == "fetch_11d":
    env = Fetch11DEnv(gui=False)
    env_num = 250
    process_num = 50
elif args.env == "rls":
    env = RLSEnv(gui=False)
    env_num = 10
    process_num = 10

res_dir = osp.join(ROOT_DIR, "test/eval_res/{}/{}/{}".format(args.env, args.name, args.planner))
viz_res_dir = osp.join(ROOT_DIR, "test/eval_res/{}/path_len/{}/{}".format(args.env, args.name, args.planner))
if not os.path.exists(viz_res_dir):
    os.makedirs(viz_res_dir)

print(res_dir)


def run(repeat, env_idx, path_len_prop, path_success_num, expert_path_len_dict):
    print("Analyzing env {}".format(env_idx))
    log_dir = osp.join(res_dir, "{}".format(env_idx))
    env_dir = osp.join(ROOT_DIR, "env/{}/dataset/test_env/{}".format(args.env, env_idx))

    start_goal = env.utils.get_start_goal(env_dir)
    if repeat == 0:
        G = env.utils.get_prm(env_dir)

        for node in G.nodes():
            nodepos = env.utils.node_to_numpy(G, node)
            if np.allclose(nodepos, np.array(start_goal[0])):
                start_node = node
            if np.allclose(nodepos, np.array(start_goal[1])):
                goal_node = node

        # expert_path_len = nx.shortest_path_length(G, start_node, goal_node)
        expert_path = nx.shortest_path(G, start_node, goal_node, "weight")
        expert_path_pos = [env.utils.node_to_numpy(G, n) for n in expert_path]
        expert_path_len = env.utils.calc_path_len(expert_path_pos)
        expert_path_len_dict[env_idx] = expert_path_len
    else:
        expert_path_len = expert_path_len_dict[env_idx]

    print("Expert path len:", expert_path_len)
    assert expert_path_len != 0

    idx = 0
    file_path = os.path.join(log_dir, "planned_path_{}_{}.json".format(repeat, idx))
    while os.path.exists(file_path):
        with open(file_path, "r") as f:
            planned_path = json.load(f)

            # plot
            # if env_idx % 25 == 0 and len(planned_path) > 1:
            #     # print(planned_path)``
            #     interpolated_path = utils.interpolate(planned_path)
            #     utils.visualize_nodes_global(occ_grid, interpolated_path, start_goal[0], start_goal[1], show=False, save=True, file_name=osp.join(viz_res_dir, "planned_path_{}_{}.png".format(env_idx, idx)))

            if len(planned_path) > 1 and np.allclose(np.array(planned_path[-1]), np.array(start_goal[1])):
                cur_path_len = env.utils.calc_path_len(planned_path)
                if cur_path_len < expert_path_len:
                    # print(f"WARNING! planned_path_{env_idx}_{idx}: {expert_path_len / cur_path_len}")
                    path_len_prop[idx] += 1
                else:
                    path_len_prop[idx] += min(1, expert_path_len / cur_path_len)
                path_success_num[idx] += 1

            elif len(planned_path) > 1:
                print(f"WARNING: wrong path success! env_idx: {env_idx} repeat: {repeat} idx: {idx}")
                print(planned_path[-1], start_goal[1])
            # else:
            #     path_len_prop[idx] += 0

        idx += 1
        file_path = os.path.join(log_dir, "planned_path_{}_{}.json".format(repeat, idx))


manager = mp.Manager()
path_len_prop = manager.dict()
# path_len_over_success = manager.dict()
path_success_num = manager.dict()
expert_path_len_dict = manager.dict()

for i in range(env_num):
    expert_path_len_dict[i] = 0

for repeat in range(args.repeat):
    # Reset dict
    for i in range(20):
        path_len_prop[i] = 0
        # path_len_over_success[i] = 0
        path_success_num[i] = 0

    # test
    j = 0
    while j < env_num:
        processes = []
        print("Running on env {} to {}".format(j, min(env_num, j + process_num)))
        for i in range(j, min(env_num, j + process_num)):
            p = mp.Process(
                target=run,
                args=(repeat, i, path_len_prop, path_success_num, expert_path_len_dict),
                daemon=True,
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        j += process_num

    print(path_success_num)
    print(path_len_prop)

    for idx in path_len_prop.keys():
        # # over success only
        # if path_success_num[idx] != 0:
        #     path_len_over_success[idx] = path_len_prop[idx] / path_success_num[idx]
        # else:
        #     path_len_over_success[idx] = 0

        path_len_prop[idx] /= env_num

    print(path_len_prop)
    with open(osp.join(viz_res_dir, "res_{}.json".format(repeat)), "w") as f:
        json.dump(path_len_prop.copy(), f)
