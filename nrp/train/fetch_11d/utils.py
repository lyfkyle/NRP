import networkx as nx

from nrp.env.fetch_11d import utils

def get_near_optimal_samples(local_prm, start_node, goal_node, diff_threshold = 0.25):
    expert_node_path = nx.shortest_path(local_prm, start_node, goal_node)
    expert_path = [utils.node_to_numpy(local_prm, n) for n in expert_node_path]

    start_pos = utils.node_to_numpy(local_prm, start_node)

    # Calculate path difference
    expert_path_len = utils.calc_path_len(expert_path)
    non_optimal_nodes = []
    near_optimal_nodes = []
    near_optimal_nodes.append(expert_node_path[1])
    for node in local_prm.nodes():
        node_pos = utils.node_to_numpy(local_prm, node)
        if not utils.is_robot_within_local_env(start_pos, node_pos, env_size=2.0):
            continue

        if node == start_node:
            continue

        if not local_prm.has_edge(start_node, node):
            continue

        node_2_goal_node_path = nx.shortest_path(local_prm, node, goal_node)
        node_2_goal_path = [utils.node_to_numpy(local_prm, n) for n in node_2_goal_node_path]
        node_2_goal_path_len = utils.calc_path_len(node_2_goal_path)
        start_2_goal_passing_node_path_len = local_prm[start_node][node]["weight"] + node_2_goal_path_len
        # print(start_2_goal_passing_node_path_len, expert_path_len)

        optimal = False
        if (
            start_node not in node_2_goal_node_path
            and ((start_2_goal_passing_node_path_len - expert_path_len) / (expert_path_len + 1e-6)) < diff_threshold
        ):
            optimal = True

        if optimal:
            near_optimal_nodes.append(node)
        else:
            non_optimal_nodes.append(node)