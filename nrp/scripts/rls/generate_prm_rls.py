import os.path as osp
import networkx as nx
import numpy as np
import itertools
import random
import time
import math


from nrp.env.rls.rls_env import RLSEnv
from nrp.env.rls import utils
from nrp import ROOT_DIR

CUR_DIR = osp.dirname(osp.abspath(__file__))
TURNING_RADIUS = 0.1
PRM_CONNECT_RADIUS = 2.0


def state_to_numpy(state):
    strlist = state.split(",")
    val_list = [float(s) for s in strlist]
    return np.array(val_list)


def process_env():
    env = RLSEnv(gui=False)

    start_time = time.time()
    env_dir = osp.join(ROOT_DIR, "env/rls/map")
    print("Process {}: generating env:{}".format(id, env_dir))

    # env
    occ_grid = utils.get_occ_grid(env_dir)
    mesh_path = utils.get_mesh_path(env_dir)

    env.clear_obstacles()
    env.load_mesh(mesh_path)
    env.load_occupancy_grid(occ_grid, add_enclosing=True)

    # utils.visualize_nodes_global(osp.join(env_dir, "env_final.obj"), occ_grid, [], None, None, show=False, save=True, file_name=osp.join(env_dir, "env.png"))

    # states
    low = env.robot.get_joint_lower_bounds()
    high = env.robot.get_joint_higher_bounds()

    print(low, high)

    # random sampling
    col_status = []
    states = []
    for _ in range(dense_num):
        random_state = [0] * env.robot.num_dim
        for i in range(env.robot.num_dim):
            random_state[i] = random.uniform(low[i], high[i])
        col_status.append(env.pb_ompl_interface.is_state_valid(random_state))
        states.append(random_state)
    # collision_states = np.array(collision_states)
    dense_G = nx.Graph()
    dense_G.add_nodes_from(
        [
            (
                "n{}".format(i),
                {"coords": ",".join(map(str, state)), "col": not col_status[i]},
            )
            for i, state in enumerate(states)
        ]
    )

    free_node_poss = np.array(
        [
            state_to_numpy(dense_G.nodes[node]["coords"])
            for node in dense_G.nodes()
            if not dense_G.nodes[node]["col"]
        ]
    )
    num_free_state = len(free_node_poss)
    # utils.visualize_nodes_global(occ_grid, node_pos, None, None, show=False, save=True, file_name=osp.join(env_dir, "dense_free.png"))

    print("Process: connecting dense graph, num_free_state = {}".format(num_free_state))

    # nodes = dense_G.nodes()
    nodes = [node for node in dense_G.nodes() if not dense_G.nodes[node]["col"]]
    node_pairs = itertools.combinations(nodes, 2)
    # print(len(list(node_pairs)))
    # print(list(node_pairs))
    pairs_to_check = []
    for node_pair in node_pairs:
        if dense_G.nodes[node_pair[0]]["col"] or dense_G.nodes[node_pair[1]]["col"]:
            continue

        if not dense_G.has_edge(node_pair[0], node_pair[1]):
            s1 = state_to_numpy(dense_G.nodes[node_pair[0]]["coords"])
            s2 = state_to_numpy(dense_G.nodes[node_pair[1]]["coords"])

            # ignore edges far apart
            if (
                math.fabs(s2[0] - s1[0]) > PRM_CONNECT_RADIUS
                or math.fabs(s2[1] - s1[1]) > PRM_CONNECT_RADIUS
            ):
                continue

            pairs_to_check.append((s1, s2, node_pair))

    print("Process: connecting dense graph, num edges to check = {}".format(len(pairs_to_check)))
    for s1, s2, node_pair in pairs_to_check:
        if utils.is_edge_free(env, s1, s2):
            dense_G.add_edge(
                node_pair[0], node_pair[1], weight=utils.calc_edge_len(s1, s2)
            )

    print("Process: edge_num: {}".format(dense_G.number_of_edges()))

    # for u, v in dense_G.edges:
    #     s1 = state_to_numpy(dense_G.nodes[u]["coords"])
    #     s2 = state_to_numpy(dense_G.nodes[v]["coords"])
    #     dense_G[u][v]["weight"] = utils.calc_edge_len(s1, s2)
    #     assert not dense_G.nodes[u]["col"]
    #     assert not dense_G.nodes[v]["col"]

    nx.write_graphml(dense_G, osp.join(env_dir, f"dense_g.graphml"))

    end_time = time.time()
    time_taken = end_time - start_time

    utils.visualize_tree_simple(
        occ_grid,
        dense_G,
        None,
        None,
        show=False,
        save=True,
        file_name=osp.join(env_dir, f"tree.png"),
    )


if __name__ == "__main__":
    dense_num = 5000
    process_env()
