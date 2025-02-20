import math
import networkx as nx
import itertools
import random
import numpy as np
from tqdm import tqdm

from nrp.env.fetch_11d import utils

PRM_CONNECT_RADIUS = 2.0


def generate_new_prm(orig_G, env, start_node, goal_node=None, local_env_size=2.0, show_tqdm=False):
    new_G = nx.create_empty_copy(orig_G)  # remove all edges
    # print(dense_G.number_of_edges(), dense_G.number_of_nodes())

    start_pos = utils.node_to_numpy(orig_G, start_node)
    # print(start_pos, goal_pos)

    # Set goal_node to free
    if goal_node is not None:
        goal_pos = utils.node_to_numpy(new_G, goal_node)
        nx.set_node_attributes(new_G, {goal_node: {"col": False, "coords": ",".join(map(str, goal_pos))}})

    print("Checking node status")
    # Get all nodes
    enlarged_size = local_env_size + 1.25
    all_nodes = []
    for node in new_G.nodes():
        node_pos = utils.node_to_numpy(new_G, node)
        if utils.is_robot_within_local_env(start_pos, node_pos, enlarged_size):
            # nx.set_node_attributes(new_G, {node: {"col": False, "coords": ",".join(map(str, node_pos))}})
            all_nodes.append(node)

    # Check node collision status
    for node in tqdm(all_nodes, disable=not show_tqdm):
        node_pos = utils.node_to_numpy(new_G, node)
        # if orig_G says it is free, it is definitely free
        if not orig_G.nodes[node]["col"]:
            nx.set_node_attributes(new_G, {node: {"col": False, "coords": ",".join(map(str, node_pos))}})
            continue

        # if within local env, trust orig_G collision status
        if utils.is_robot_within_local_env(start_pos, node_pos, local_env_size):
            continue

        col_status = env.pb_ompl_interface.is_state_valid(node_pos)
        nx.set_node_attributes(new_G, {node: {"col": not col_status, "coords": ",".join(map(str, node_pos))}})

    free_nodes = [node for node in all_nodes if not new_G.nodes[node]["col"]]
    num_free_state = len(free_nodes)
    print(f"total number of free nodes: {num_free_state}")

    node_pairs = list(itertools.combinations(free_nodes, 2))
    print(f"Checking edges, total_num: {len(node_pairs)}")
    for pair in tqdm(node_pairs, disable=not show_tqdm):
        s1 = utils.node_to_numpy(new_G, pair[0])
        s2 = utils.node_to_numpy(new_G, pair[1])

        # Ignore edges far apart
        if math.fabs(s2[0] - s1[0]) > PRM_CONNECT_RADIUS or math.fabs(s2[1] - s1[1]) > PRM_CONNECT_RADIUS:
            continue

        # If original graph already has this connection, this is surely free.
        if orig_G.has_edge(pair[0], pair[1]):
            new_G.add_edge(pair[0], pair[1], weight=utils.calc_edge_len(s1, s2))
            continue

        # If both s1 and s2 are in local environment, orig_G should already contain the edge collision information.
        if utils.is_robot_within_local_env(start_pos, s1, local_env_size) and utils.is_robot_within_local_env(
            start_pos, s2, local_env_size
        ):
            continue

        # if edge does not pass through local env, it is definitely free
        if not utils.path_pass_through_local_env(start_pos, s1, s2, local_env_size):
            new_G.add_edge(pair[0], pair[1], weight=utils.calc_edge_len(s1, s2))
            continue

        if utils.is_edge_free(env, s1, s2):
            new_G.add_edge(pair[0], pair[1], weight=utils.calc_edge_len(s1, s2))

    # Finally connect all nodes to goal
    if goal_node is not None:
        print("Connecting outside nodes to goal!!")
        if not utils.is_robot_within_local_env(start_pos, goal_pos, enlarged_size):
            for node in tqdm(all_nodes, disable=not show_tqdm):
                node_pos = utils.node_to_numpy(new_G, node)
                if new_G.nodes[node]["col"]:
                    continue

                # For node outside local env, can check connection to goal by checking whether sl path pass through
                # local env
                if utils.is_robot_outside_local_env(start_pos, node_pos, local_env_size):
                    valid = not utils.path_pass_through_local_env(start_pos, node_pos, goal_pos, local_env_size)

                    # if does not pass through local env
                    if valid:
                        new_G.add_edge(node, goal_node, weight=utils.calc_edge_len(node_pos, goal_pos))
                        continue

                # for others, have to invoke collision checker
                if utils.is_edge_free(env, node_pos, goal_pos):
                    new_G.add_edge(node, goal_node, weight=utils.calc_edge_len(node_pos, goal_pos))

    new_G.remove_nodes_from(list(nx.isolates(new_G)))
    all_nodes = [node for node in all_nodes if new_G.has_node(node)]
    # print(dense_G.number_of_edges(), dense_G.number_of_nodes())

    return new_G, all_nodes

def add_goal_node_to_prm(G, env, start_node, goal_pos, local_env_size=2.0, show_tqdm=False):
    start_pos = utils.node_to_numpy(G, start_node)
    goal_node = "n{}".format(G.number_of_nodes() + 1)
    G.add_node(goal_node, coords=",".join(map(str, goal_pos)), col=False)

    print("Connecting outside nodes to goal!!")
    enlarged_size = local_env_size + 1
    free_nodes = [node for node in G.nodes() if not G.nodes[node]["col"]]
    if not utils.is_robot_within_local_env(start_pos, goal_pos, enlarged_size):
        for node in tqdm(free_nodes, disable=not show_tqdm):
            if node == goal_node:
                continue

            node_pos = utils.node_to_numpy(G, node)

            # For node outside local env, can check connection to goal by checking whether sl path pass through
            # local env
            if utils.is_robot_outside_local_env(start_pos, node_pos, local_env_size):
                valid = not utils.path_pass_through_local_env(start_pos, node_pos, goal_pos, local_env_size)

                # if does not pass through local env
                if valid:
                    G.add_edge(node, goal_node, weight=utils.calc_edge_len(node_pos, goal_pos))
                    continue

            # for others, have to invoke collision checker
            if utils.is_edge_free(env, node_pos, goal_pos):
                G.add_edge(node, goal_node, weight=utils.calc_edge_len(node_pos, goal_pos))

    return G, goal_node


# Legacy:
"""
    print("Connecting outside nodes to goal")
    # Outside nodes are those that are within 1 meter away from local env
    size = local_env_size + 1
    outside_nodes = []
    for node in new_G.nodes():
        node_pos = utils.node_to_numpy(new_G, node)
        if (
            math.fabs(node_pos[0] - start_pos[0]) > local_env_size
            or math.fabs(node_pos[1] - start_pos[1]) > local_env_size
        ) and (math.fabs(node_pos[0] - start_pos[0]) <= size and math.fabs(node_pos[1] - start_pos[1]) <= size):
            nx.set_node_attributes(new_G, {node: {"col": False, "coords": ",".join(map(str, node_pos))}})
            outside_nodes.append(node)

    for node in outside_nodes:
        node_pos = utils.node_to_numpy(new_G, node)
        assert not utils.is_robot_within_local_env_2(start_pos, node_pos)

    # print(len(outside_nodes))

    # check valid outside nodes. path from valid outside nodes to goal should not pass through the local environment
    valid_outside_nodes = []
    for node in outside_nodes:
        node_pos = utils.node_to_numpy(new_G, node)
        path_to_goal = utils.interpolate([node_pos, goal_pos])

        valid = True
        for p in path_to_goal:
            if math.fabs(p[0] - start_pos[0]) <= local_env_size and math.fabs(p[1] - start_pos[1]) <= local_env_size:
                valid = False
                break

        if valid:
            valid_outside_nodes.append(node)
            new_G.add_edge(node, goal_node, weight=utils.calc_edge_len(node_pos, goal_pos))

    print("Connecting inside nodes using the original graph")
    # Inside nodes are nodes within local environment
    inside_nodes = []
    for node in new_G.nodes():
        node_pos = utils.node_to_numpy(new_G, node)
        if (
            math.fabs(node_pos[0] - start_pos[0]) <= local_env_size
            and math.fabs(node_pos[1] - start_pos[1]) <= local_env_size
        ):
            if not new_G.nodes[node]["col"]:
                inside_nodes.append(node)

    print(f"total number of nodes: {len(inside_nodes) + len(outside_nodes)}")

    all_nodes = inside_nodes + outside_nodes
    node_pairs = list(itertools.combinations(all_nodes, 2))
    pairs_to_check = []
    for node_pair in tqdm(node_pairs):
        s1 = utils.node_to_numpy(new_G, node_pair[0])
        s2 = utils.node_to_numpy(new_G, node_pair[1])

        # Ignore edges far apart
        if math.fabs(s2[0] - s1[0]) > PRM_CONNECT_RADIUS or math.fabs(s2[1] - s1[1]) > PRM_CONNECT_RADIUS:
            continue

        # If original graph already has this connection, this is surely free.
        if orig_G.has_edge(node_pair[0], node_pair[1]):
            new_G.add_edge(node_pair[0], node_pair[1], weight=utils.calc_edge_len(s1, s2))
            continue

        # If the edge does not pass through local environment, then it is definitely collision-free.
        path = utils.interpolate([s1, s2])
        valid = True
        for p in path:
            if (
                math.fabs(p[0] - start_pos[0]) <= local_env_size
                and math.fabs(p[1] - start_pos[1]) <= local_env_size
            ):
                valid = False
                break

        if valid:
            new_G.add_edge(node_pair[0], node_pair[1], weight=utils.calc_edge_len(s1, s2))
            continue

        # If both s1 and s2 are in local environment, orig_G should already contain the edge collision information.
        if (
            math.fabs(s1[0] - start_pos[0]) > local_env_size
            or math.fabs(s1[1] - start_pos[1]) > local_env_size
            or math.fabs(s2[0] - start_pos[0]) > local_env_size
            or math.fabs(s2[1] - start_pos[1]) > local_env_size
        ):
            pairs_to_check.append([node_pair[0], node_pair[1]])

    print(f"Number of edges to check: {len(pairs_to_check)}")
    for pair in tqdm(pairs_to_check):
        s1 = utils.node_to_numpy(new_G, pair[0])
        s2 = utils.node_to_numpy(new_G, pair[1])
        if utils.is_edge_free(env, s1, s2):
            new_G.add_edge(pair[0], pair[1], weight=utils.calc_edge_len(s1, s2))

    # use the original graph to connect inside nodes
    # node_pairs = itertools.combinations(inside_nodes, 2)
    # for node_pair in node_pairs:
    #     if orig_G.has_edge(node_pair[0], node_pair[1]):
    #         s1 = utils.node_to_numpy(new_G, node_pair[0])
    #         s2 = utils.node_to_numpy(new_G, node_pair[1])
    #         new_G.add_edge(node_pair[0], node_pair[1], weight=orig_G[node_pair[0]][node_pair[1]]["weight"])

    # print("Connecting outside nodes using the original graph")
    # pairs_to_check = []
    # node_pairs = itertools.combinations(outside_nodes, 2)
    # for node_pair in node_pairs:
    #     s1 = utils.node_to_numpy(new_G, node_pair[0])
    #     s2 = utils.node_to_numpy(new_G, node_pair[1])
    #     if orig_G.has_edge(node_pair[0], node_pair[1]):
    #         new_G.add_edge(node_pair[0], node_pair[1], weight=orig_G[node_pair[0]][node_pair[1]]["weight"])
    #     else:
    #         if math.fabs(s2[0] - s1[0]) < PRM_CONNECT_RADIUS and math.fabs(s2[1] - s1[1]) < PRM_CONNECT_RADIUS:
    #             pairs_to_check.append((s1, s2, node_pair))

    # print("num edges to check = {}".format(len(pairs_to_check)))
    # random.shuffle(pairs_to_check)
    # if len(pairs_to_check) > 5000:
    #     pairs_to_check = pairs_to_check[:5000]

    # for s1, s2, node_pair in pairs_to_check:
    #     if new_G.has_edge(node_pair[0], node_pair[1]):
    #         continue
    #     else:
    #         path = utils.interpolate([s1, s2])

    #         # If the edge does not pass through local environment, then it is definitely collision-free.
    #         valid = True
    #         for p in path:
    #             if (
    #                 math.fabs(p[0] - start_pos[0]) <= local_env_size
    #                 and math.fabs(p[1] - start_pos[1]) <= local_env_size
    #             ):
    #                 valid = False
    #                 break

    #         if valid:
    #             new_G.add_edge(node_pair[0], node_pair[1], weight=utils.calc_edge_len(s1, s2))
    #         # else need to check collision using collision checker
    #         else:
    #             if utils.is_edge_free(env, s1, s2):
    #                 new_G.add_edge(node_pair[0], node_pair[1], weight=utils.calc_edge_len(s1, s2))

    # print("Connecting inside nodes to outside nodes")
    # node_pairs_to_check = []
    # for node in inside_nodes:
    #     for node2 in outside_nodes:
    #         node_pairs_to_check.append([node, node2])
    # # random.shuffle(node_pairs_to_check)

    # pairs_to_check = []
    # for node_pair in node_pairs_to_check:
    #     if new_G.nodes[node_pair[0]]["col"] or new_G.nodes[node_pair[1]]["col"]:
    #         continue

    #     s1 = utils.node_to_numpy(new_G, node_pair[0])
    #     s2 = utils.node_to_numpy(new_G, node_pair[1])

    #     if orig_G.has_edge(node_pair[0], node_pair[1]):
    #         new_G.add_edge(node_pair[0], node_pair[1], weight=utils.calc_edge_len(s1, s2))
    #         continue

    #     # ignore edges far apart
    #     if (
    #         np.allclose(s1, s2)
    #         or math.fabs(s2[0] - s1[0]) > PRM_CONNECT_RADIUS
    #         or math.fabs(s2[1] - s1[1]) > PRM_CONNECT_RADIUS
    #     ):
    #         continue

    #     pairs_to_check.append((s1, s2, node_pair))

    # # print("num edges to check = {}".format(len(pairs_to_check)))
    # random.shuffle(pairs_to_check)
    # if len(pairs_to_check) > 5000:
    #     pairs_to_check = pairs_to_check[:5000]

    # for s1, s2, node_pair in pairs_to_check:
    #     if new_G.has_edge(node_pair[0], node_pair[1]):
    #         continue
    #     else:
    #         if utils.is_edge_free(env, s1, s2):
    #             new_G.add_edge(node_pair[0], node_pair[1], weight=utils.calc_edge_len(s1, s2))

"""
