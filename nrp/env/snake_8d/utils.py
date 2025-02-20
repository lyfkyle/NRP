import os
import os.path as osp
import numpy as np
import math
import matplotlib.pyplot as plt
import imageio
import torch
import copy
import time
import matplotlib.patheffects as mpe
from matplotlib.patches import Ellipse
from matplotlib.colors import LinearSegmentedColormap
# for k-shortest path
from heapq import heappush, heappop
from itertools import count
import networkx as nx
import json

from nrp.planner.decomposed_planner import AStar

CUR_DIR = osp.dirname(osp.abspath(__file__))

LINK_LENGTH = 0.12
LINK_WIDTH = 0.05
BASE_SIZE = 0.15

"""
Env utils
"""


def get_occ_grid(env_dir):
    return np.loadtxt(os.path.join(env_dir, "occ_grid_small.txt")).astype(np.uint8)


def get_prm(env_dir):
    g = nx.read_graphml(os.path.join(env_dir, "dense_g_small.graphml.xml"))
    return g


def get_mesh_path(env_dir):
    return os.path.join(env_dir, "env_small.obj")

def get_start_goal(env_dir):
    with open(osp.join(env_dir, "start_goal.json")) as f:
        test_path = json.load(f)

    return test_path


"""
Path Utils
"""


def rrt_extend_path(env, path, step_size=0.1, intermediate=False):
    final_path = [path[0]]
    for i in range(1, len(path)):
        v1 = path[i - 1]
        v2 = path[i]
        if intermediate:
            res_path = rrt_extend_intermediate(env, v1, v2, step_size)
        else:
            res_path = rrt_extend(env, v1, v2)

        if len(res_path) > 1:
            final_path += res_path[1:]
        if not np.allclose(np.array(res_path[-1]), np.array(v2)):
            # print("collision detected!!!")
            break

    return final_path


def rrt_extend(maze, v, g, step_size=0.1):
    node1_pos = np.array(v)
    node2_pos = np.array(g)

    if np.allclose(node1_pos, node2_pos):
        return [node1_pos.tolist()]

    actual_step_size, num_points_per_edge = discretize_edge(v, g, step_size)

    res_path = [node1_pos.tolist()]
    for i in range(1, num_points_per_edge):
        nodepos = node1_pos + actual_step_size * i
        if not maze.pb_ompl_interface.is_state_valid(nodepos.tolist()):
            break

        if len(res_path) == 1:
            res_path.append(nodepos.tolist())
        else:
            res_path[1] = nodepos.tolist()

    return res_path


def rrt_extend_intermediate(maze, v, g, step_size=0.1):
    node1_pos = np.array(v)
    node2_pos = np.array(g)

    if np.allclose(node1_pos, node2_pos):
        return [node1_pos.tolist()]

    actual_step_size, num_points_per_edge = discretize_edge(v, g, step_size)

    res_path = [node1_pos.tolist()]
    nodepos = np.zeros(len(node1_pos))
    for i in range(1, num_points_per_edge):
        nodepos = node1_pos + actual_step_size * i
        if not maze.pb_ompl_interface.is_state_valid(nodepos.tolist()):
            break

        res_path.append(nodepos.tolist())

    # to avoid float calculation precision problem
    if np.allclose(np.array(res_path[-1]), node2_pos, atol=1e-4):
        res_path.append(g)
    # else:
        # to prevent res_path to be too near obstacle
        # res_path = res_path[:max(1, len(res_path) - 10)]

    return res_path


def is_edge_free(maze, node1_state, node2_state, step_size=0.1):
    node1_pos = np.array(node1_state)
    node2_pos = np.array(node2_state)

    if not maze.pb_ompl_interface.is_state_valid(node1_pos.tolist()):
        return False

    if not maze.pb_ompl_interface.is_state_valid(node2_pos.tolist()):
        return False

    if np.allclose(node1_pos, node2_pos, atol=step_size):
        return True

    actual_step_size, num_points_per_edge = discretize_edge(node1_pos, node2_pos, step_size)

    # assert(np.max(step) < step_size and np.min(step) > -step_size)
    for i in range(1, num_points_per_edge):
        nodepos = node1_pos + actual_step_size * i
        if not maze.pb_ompl_interface.is_state_valid(nodepos.tolist()):
            return False
    return True


def is_path_free(maze, path, step_size=0.1):
    if not maze.pb_ompl_interface.is_state_valid(path[0]):
        return False

    for i in range(1, len(path)):
        node1_pos = np.array(path[i - 1])
        node2_pos = np.array(path[i])
        actual_step_size, num_points_per_edge = discretize_edge(node1_pos, node2_pos, step_size)

        # assert(np.max(step) < step_size and np.min(step) > -step_size)
        for j in range(1, num_points_per_edge):
            nodepos = node1_pos + actual_step_size * j
            if not maze.pb_ompl_interface.is_state_valid(nodepos.tolist()):
                return False

    return True


def get_collision_vector(maze, path, step_size=0.1):
    collision_vector = []
    if not maze.pb_ompl_interface.is_state_valid(path[0]):
        collision_vector.append(1)
    else:
        collision_vector.append(0)

    for i in range(1, len(path)):
        node1_pos = np.array(path[i - 1])
        node2_pos = np.array(path[i])
        actual_step_size, num_points_per_edge = discretize_edge(node1_pos, node2_pos, step_size)

        collision = 0
        # assert(np.max(step) < step_size and np.min(step) > -step_size)
        for j in range(1, num_points_per_edge):
            nodepos = node1_pos + actual_step_size * j
            if not maze.pb_ompl_interface.is_state_valid(nodepos.tolist()):
                collision = 1
                break
        collision_vector.append(collision)
    # print(collision_vector)
    assert len(collision_vector) == len(path)
    return collision_vector


def calc_edge_len(s1, s2):
    config1 = np.array(s1).reshape(-1, 8)
    config2 = np.array(s2).reshape(-1, 8)

    base_len = np.linalg.norm((config2[:, :2] - config1[:, :2]), axis=-1)
    arm_len = np.linalg.norm((config2[:, 2:] - config1[:, 2:]), axis=-1)
    res = base_len + 0.167 * arm_len
    if res.shape[0] == 1:
        res = res.reshape(-1)
    return res


def calc_edge_len_base(s1, s2):
    config1 = np.array(s1).reshape(-1, 2)
    config2 = np.array(s2).reshape(-1, 2)

    res = np.linalg.norm((config2 - config1), axis=-1)
    if res.shape[0] == 1:
        res = res.reshape(-1)
    return res


def calc_path_len(path):
    length = 0
    for i, node in enumerate(path):
        if i == 0:
            continue
        length += calc_edge_len(path[i - 1], node).item()

    return length


def calc_path_len_base(path):
    length = 0
    for i, node in enumerate(path):
        if i == 0:
            continue
        length += calc_edge_len_base(path[i - 1][:2], node[:2]).item()

    return length


def interpolate(path, step_size=0.1, num_points_per_edge=None):
    new_path = [path[0]]
    for i in range(1, len(path)):
        node1_pos = np.array(path[i - 1])
        node2_pos = np.array(path[i])

        actual_step_size, num_points_per_edge = discretize_edge(node1_pos, node2_pos, step_size)
        for j in range(1, num_points_per_edge):
            nodepos = node1_pos + actual_step_size * j
            new_path.append(nodepos.tolist())

    new_path.append(path[-1])

    return new_path


def interpolate_base(path, step_size=0.1, num_points_per_edge=None):
    new_path = [path[0]]
    for i in range(1, len(path)):
        node1_pos = np.array(path[i - 1])
        node2_pos = np.array(path[i])

        base_step_size, num_points_per_edge = discretize_edge_base(node1_pos, node2_pos, step_size)
        arm_step_size = (node2_pos[2:] - node1_pos[2:]) / (num_points_per_edge - 1)
        for j in range(1, num_points_per_edge):
            nodepos = node1_pos.copy()
            nodepos[:2] = node1_pos[:2] + base_step_size * j
            nodepos[2:] = node1_pos[2:] + arm_step_size * j
            new_path.append(nodepos.tolist())

    new_path.append(path[-1])

    return new_path


def interpolate_to_fixed_horizon(path, horizon=20):
    if len(path) >= horizon:
        print("Warning: Original path length >= horizon, returning cropped original path")
        return path[:horizon]
    edge_diffs = []
    edge_lengths = []
    total_length = 0
    for i in range(1, len(path)):
        node1_pos = np.array(path[i - 1])
        node2_pos = np.array(path[i])
        diff = node2_pos - node1_pos
        edge_diffs.append(diff)
        edge_length = np.max(np.abs(diff))
        edge_lengths.append(edge_length)
        total_length += edge_length
    # print("total length:", total_length)

    # proportional representation
    total_num_segs = horizon - 1
    proportion_per_seg = 1 / total_num_segs

    proportions = np.array(edge_lengths) / total_length - proportion_per_seg
    num_segs_per_edge = np.ones_like(proportions, dtype=int)

    remainder = total_num_segs - sum(num_segs_per_edge)
    for i in range(remainder):
        max_id = proportions.argmax()
        num_segs_per_edge[max_id] += 1
        proportions[max_id] -= proportion_per_seg

    # print(num_segs_per_edge)

    assert sum(num_segs_per_edge) == total_num_segs

    # num_points_per_edge = [math.ceil(l * horizon / total_length) for l in edge_lengths]
    # for i in range(sum(num_points_per_edge) - horizon - len(path) + 2):
    #     num_points_per_edge[-i] -= 1
    # print(num_points_per_edge)

    # assert sum(num_points_per_edge) == horizon + len(path) - 2

    new_path = [path[0]]
    for i in range(len(path)-1):
        node1_pos = np.array(path[i])
        # node2_pos = np.array(path[i+1])
        num_segs = num_segs_per_edge[i]
        step_size = edge_diffs[i] / num_segs
        for j in range(1, num_segs):
            nodepos = node1_pos + step_size * j
            new_path.append(nodepos.tolist())

        new_path.append(path[i+1])

    assert len(new_path) == horizon

    return new_path


def interpolate_edge(v1, v2, step_size=0.1, num_points_per_edge=None):
    node1_pos = np.array(v1)
    node2_pos = np.array(v2)
    # if np.allclose(node1_pos, node2_pos):
    #     return [node1_pos.tolist()], 0

    new_path = [node1_pos.tolist()]
    actual_step_size, num_points_per_edge = discretize_edge(node1_pos, node2_pos, step_size)
    for j in range(1, num_points_per_edge):
        nodepos = node1_pos + actual_step_size * j
        new_path.append(nodepos.tolist())

    # new_path.append(v2)
    new_path[-1] = v2

    return new_path, np.linalg.norm(actual_step_size)


def discretize_edge(v, g, step_size=0.1):
    node1_pos = np.array(v)
    node2_pos = np.array(g)

    diff = node2_pos - node1_pos
    num_points_per_edge = math.ceil(np.max(np.abs(diff)) / step_size) + 1

    step_size = diff / (num_points_per_edge - 1)

    return step_size, num_points_per_edge


def discretize_edge_base(v, g, step_size=0.1):
    node1_base_pos = np.array(v)[:2]
    node2_base_pos = np.array(g)[:2]

    diff = node2_base_pos - node1_base_pos
    num_points_per_edge = math.ceil(np.max(np.abs(diff)) / step_size) + 1

    step_size = diff / (num_points_per_edge - 1)

    return step_size, num_points_per_edge


def clip_path(path, local_env_size=2.0):
    # clip the trajectory in local range
    found_first_pos_out = False
    new_path_len = len(path)
    new_path = []
    for i, pos in enumerate(path):
        if is_robot_within_local_env(pos, local_env_size):
            continue
        else:
            first_pos_out = pos
            last_pos_in = path[i-1]
            found_first_pos_out = True
            new_path_len = i
            break

    if found_first_pos_out:
        q_n = get_intersection(last_pos_in, first_pos_out, local_env_size)
        new_path = copy.deepcopy(path[:new_path_len])
        new_path.append(q_n)
    else:
        q_n = path[-1]
        new_path = copy.deepcopy(path)

    return new_path


def get_intersection(v1, v2, env_size):
    assert (is_robot_within_local_env(v1, env_size) and (not is_robot_within_local_env(v2, env_size)))

    node1_pos = np.array(v1)
    node2_pos = np.array(v2)
    step_size, num_points_per_edge = discretize_edge(node1_pos, node2_pos, step_size=0.05)
    for j in range(1, num_points_per_edge):
        nodepos = node2_pos - step_size * j
        if is_robot_within_local_env(nodepos, env_size):
            return nodepos.tolist()
    # if no intermediate point in local env, return first point
    return v1


"""
Helpers
"""


def string_to_numpy(state):
    strlist = state.split(",")
    val_list = [float(s) for s in strlist]
    return np.array(val_list)


def node_to_numpy(G, n):
    return string_to_numpy(G.nodes[n]["coords"])


def calculate_stats(true_pos, true_neg, false_pos, false_neg):
    accuracy = float(true_pos + true_neg) / (
        true_pos + true_neg + false_pos + false_neg
    )
    if true_pos + false_pos > 0:
        precision = float(true_pos) / (true_pos + false_pos)
    else:
        precision = 1
    if true_pos + false_neg > 0:
        recall = float(true_pos) / (true_pos + false_neg)
    else:
        recall = 1

    return accuracy, precision, recall


"""
Plotting APIs
"""

c_map = {
    "NRP-d": "tab:orange",
    "NRP-g": "tab:red",
    "NRP*-d": "tab:orange",
    "NRP*-g": "tab:red",
    "NRP-d-global": "tab:cyan",
    "NRP-g-global": "tab:pink",
    "CVAE-RRT": "tab:cyan",
    "CVAE-IRRT*": "tab:cyan",
    "IRRT*": "tab:blue",
    "BIT*": "tab:green",
    "NEXT": "tab:purple",
    "RRT": "tab:gray",
    "RRT-IS": "tab:olive",
    "Decoupled": "tab:brown"
}


def visualize_nodes_manual(
    occ_g, curr_node_posns, start_pos, goal_pos, show=True, save=False, file_name=None
):
    fig1 = plt.figure(figsize=(10, 10), dpi=100)
    ax1 = fig1.add_subplot(111, aspect="equal")
    occ_grid_resolution = 0.5

    occ_grid_size = occ_g.shape[0]
    tmp = occ_grid_size / 4.0 - 0.25

    inch_per_grid = 10.0 / occ_grid_size
    inch_per_meter = inch_per_grid / occ_grid_resolution
    points_per_inch = 54
    points_per_meter = points_per_inch * inch_per_meter

    # s = (10.0 / occ_grid_size * 100 / 2)
    # s = (10 / occ_g.shape[0] * 50)
    for i in range(occ_grid_size):
        for j in range(occ_grid_size):
            if occ_g[i, j] == 1:
                # ax1.add_patch(patches.Rectangle(
                #     (i/10.0 - 2.5, j/10.0 - 2.5),   # (x,y)
                #     0.1,          # width
                #     0.1,          # height
                #     alpha=0.6
                #     ))
                plt.scatter(
                    j / 2.0 - tmp,
                    tmp - i / 2.0,
                    color="black",
                    marker="s",
                    s=(points_per_meter * 0.5) ** 2,
                    alpha=1,
                )  # init

    if len(curr_node_posns) > 0:
        # plt.scatter(curr_node_posns[:,0], curr_node_posns[:,1], s = 50, color = 'green')
        for i, pos in enumerate(curr_node_posns):
            visualize_robot(pos, ax=ax1)
            plt.text(pos[0], pos[1], str(i), color="black", fontsize=12)

    if start_pos is not None:
        # plt.scatter(start_pos[0], start_pos[1], color="red", s=100, edgecolors='black', alpha=1, zorder=10) # init
        visualize_robot(start_pos, start=True, ax=ax1, s=int(points_per_meter * 0.1))
    if goal_pos is not None:
        # plt.scatter(start_pos[0], start_pos[1], color="red", s=100, edgecolors='black', alpha=1, zorder=10) # init
        visualize_robot(goal_pos, goal=True, ax=ax1, s=int(points_per_meter * 0.1))

    limit = occ_grid_size / 4.0
    plt.xlim(-limit, limit)
    plt.ylim(-limit, limit)
    if show:
        plt.show()
    if save:
        fig1.savefig(file_name, dpi=fig1.dpi)
        plt.close()


def visualize_nodes_global(
    mesh,
    occ_g,
    curr_node_posns,
    start_pos=None,
    goal_pos=None,
    sample_pos=None,
    show=True,
    save=False,
    file_name=None,
    viz_edge=False,
    edge_path=None,
):
    fig1 = plt.figure(figsize=(10, 10), dpi=250)
    img = plt.imread(os.path.join(CUR_DIR, "tile.jpg"))

    occ_grid_size = max(occ_g.shape[0], occ_g.shape[1])
    s = 10 / occ_grid_size * 60

    ax1 = fig1.add_subplot(111, aspect="equal")
    ax1.imshow(img, extent=[0, 10, 0, 10])

    for i in range(occ_g.shape[0]):
        for j in range(occ_g.shape[1]):
            if occ_g[i, j] == 1:
                plt.scatter(
                    (i + 0.5) * 0.1,
                    (j + 0.5) * 0.1,
                    color="black",
                    marker="s",
                    s=s**2,
                    alpha=1,
                )  # init

    # visualize edges
    if viz_edge:
        if edge_path is None:
            edge_path = curr_node_posns

        for i in range(1, len(edge_path)):
            plt.plot((edge_path[i - 1][0], edge_path[i][0]), (edge_path[i - 1][1], edge_path[i][1]), ls="--", lw=s / 2, c='#66B2FF', zorder=1)

    if len(curr_node_posns) > 0:
        # plt.scatter(curr_node_posns[:,0], curr_node_posns[:,1], s = 50, color = 'green')
        for i, pos in enumerate(curr_node_posns):
            if sample_pos is not None and np.allclose(np.array(pos), np.array(sample_pos)):
                continue
            if start_pos is not None and np.allclose(np.array(pos), np.array(start_pos)):
                continue
            if goal_pos is not None and np.allclose(np.array(pos), np.array(goal_pos)):
                continue

            visualize_robot(pos, ax=ax1, s=s, color='#cce5ff')

    if start_pos is not None:
        visualize_robot(start_pos, ax=ax1, s=s, color="#FFBF0F")
    if goal_pos is not None:
        visualize_robot(goal_pos, ax=ax1, s=s, color="#FF9999")
    if sample_pos is not None:
        # visualize_robot(sample_pos, ax=ax1, s=s, color="#66B2FF")
        visualize_robot(sample_pos, ax=ax1, s=s, color="#00FF00")

    plt.xlim(0, occ_g.shape[0] * 0.1)
    plt.ylim(0, occ_g.shape[1] * 0.1)

    # Remove paddings
    plt.axis('off')  # remove axis
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)

    if show:
        plt.show()
    if save:
        print(f"saving img to {file_name}...")
        fig1.savefig(file_name, dpi=fig1.dpi)
        plt.close()


def visualize_nodes_local(occ_g, cur_nodes_pos, start_pos, goal_pos, max_num=float('inf'), show=True, save=False, file_name=None, title=None):
    fig1 = plt.figure(figsize=(10, 10), dpi=250)

    occ_grid_size = occ_g.shape[0]
    tmp = occ_grid_size // 2
    # s = 225
    s = (10 / occ_g.shape[0] * 60) ** 2

    ax1 = fig1.add_subplot(111, aspect="equal")

    for i in range(occ_g.shape[0]):
        for j in range(occ_g.shape[1]):
            if occ_g[i, j] == 1:
                plt.scatter(
                    (i - tmp + 0.5) * 0.1,
                    (j - tmp + 0.5) * 0.1,
                    color="black",
                    marker="s",
                    s=s,
                    alpha=1,
                )  # init

    if len(cur_nodes_pos) > 0:
        if len(cur_nodes_pos) > max_num:
            cur_nodes_pos = cur_nodes_pos[:max_num]

        # plt.scatter(curr_node_posns[:,0], curr_node_posns[:,1], s = 50, color = 'green')
        for i, pos in enumerate(cur_nodes_pos):
            visualize_robot(pos, ax=ax1)
            # plt.text(pos[0], pos[1], str(i), color="black", fontsize=12)

    if start_pos is not None:
        # plt.scatter(start_pos[0], start_pos[1], color="red", s=100, edgecolors='black', alpha=1, zorder=10) # init
        visualize_robot(start_pos, color='y', ax=ax1)
    if goal_pos is not None:
        # plt.scatter(start_pos[0], start_pos[1], color="red", s=100, edgecolors='black', alpha=1, zorder=10) # init
        if math.fabs(goal_pos[0]) > 2.0 or math.fabs(goal_pos[1]) > 2.0:
            goal_dir = math.atan2(goal_pos[1], goal_pos[0])

            goal_viz = [0] * 2
            if math.fabs(goal_pos[0]) < math.fabs(goal_pos[1]):
                goal_viz[1] = 2.0 if goal_pos[1] > 0 else -2.0
                goal_viz[0] = goal_viz[1] / math.tan(goal_dir)
            else:
                goal_viz[0] = 2.0 if goal_pos[0] > 0 else -2.0
                goal_viz[1] = goal_viz[0] * math.tan(goal_dir)

            print(goal_viz)
            visualize_robot(goal_viz, color='r', ax=ax1)
        else:
            visualize_robot(goal_pos, color='r', ax=ax1)

    if title is not None:
        plt.title(title)

    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    if show:
        plt.show()
    if save:
        fig1.savefig(file_name, dpi=fig1.dpi)
        plt.close()


def visualize_nodes_local_with_guidance(
    occ_g, cur_nodes_pos, uncond_node_pos, cond_node_pos, start_pos, goal_pos, max_num=float('inf'), show=True, save=False, file_name=None, title=None
):
    fig1 = plt.figure(figsize=(10, 10), dpi=250)

    occ_grid_size = occ_g.shape[0]
    tmp = occ_grid_size // 2
    # s = 225
    s = (10 / occ_g.shape[0] * 60) ** 2

    ax1 = fig1.add_subplot(111, aspect="equal")

    for i in range(occ_g.shape[0]):
        for j in range(occ_g.shape[1]):
            if occ_g[i, j] == 1:
                plt.scatter(
                    (i - tmp + 0.5) * 0.1,
                    (j - tmp + 0.5) * 0.1,
                    color="black",
                    marker="s",
                    s=s,
                    alpha=1,
                )  # init

    if len(cur_nodes_pos) > 0:
        if len(cur_nodes_pos) > max_num:
            cur_nodes_pos = cur_nodes_pos[:max_num]

        # plt.scatter(curr_node_posns[:,0], curr_node_posns[:,1], s = 50, color = 'green')
        for i, pos in enumerate(cur_nodes_pos):
            visualize_robot(pos, ax=ax1)
            # plt.text(pos[0], pos[1], str(i), color="black", fontsize=12)

    if len(uncond_node_pos) > 0:
        if len(uncond_node_pos) > max_num:
            uncond_node_pos = uncond_node_pos[:max_num]
        for i, pos in enumerate(uncond_node_pos):
            visualize_robot(pos, ax=ax1, color="grey")

    if len(cond_node_pos) > 0:
        if len(cond_node_pos) > max_num:
            cond_node_pos = cond_node_pos[:max_num]
        for i, pos in enumerate(cond_node_pos):
            visualize_robot(pos, ax=ax1, color="blue")

    if start_pos is not None:
        # plt.scatter(start_pos[0], start_pos[1], color="red", s=100, edgecolors='black', alpha=1, zorder=10) # init
        visualize_robot(start_pos, color='y', ax=ax1)
    if goal_pos is not None:
        # plt.scatter(start_pos[0], start_pos[1], color="red", s=100, edgecolors='black', alpha=1, zorder=10) # init
        if math.fabs(goal_pos[0]) > 2.0 or math.fabs(goal_pos[1]) > 2.0:
            goal_dir = math.atan2(goal_pos[1], goal_pos[0])

            goal_viz = [0] * 2
            if math.fabs(goal_pos[0]) < math.fabs(goal_pos[1]):
                goal_viz[1] = 2.0 if goal_pos[1] > 0 else -2.0
                goal_viz[0] = goal_viz[1] / math.tan(goal_dir)
            else:
                goal_viz[0] = 2.0 if goal_pos[0] > 0 else -2.0
                goal_viz[1] = goal_viz[0] * math.tan(goal_dir)

            print(goal_viz)
            visualize_robot(goal_viz, color='r', ax=ax1)
        else:
            visualize_robot(goal_pos, color='r', ax=ax1)

    if title is not None:
        plt.title(title)

    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    if show:
        plt.show()
    if save:
        fig1.savefig(file_name, dpi=fig1.dpi)
        plt.close()

def visualize_distributions(occ_g, search_dist_mu, search_dist_sigma, q_min, q_max, start_pos=None, goal_pos=None, show=True, save=False, file_name=None):
    fig1 = plt.figure(figsize=(10, 10), dpi=250)
    # img = plt.imread("tile.jpg")

    occ_grid_size = max(occ_g.shape[0], occ_g.shape[1])
    s = 10 / occ_grid_size * 60

    ax1 = fig1.add_subplot(111, aspect="equal")
    # ax1.imshow(img, extent=[0, 10, 0, 10])

    for i in range(occ_g.shape[0]):
        for j in range(occ_g.shape[1]):
            if occ_g[i, j] == 1:
                plt.scatter(
                    (i + 0.5) * 0.1,
                    (j + 0.5) * 0.1,
                    color="black",
                    marker="s",
                    s=s**2,
                    alpha=1,
                )  # init

    # visualize distributions (2D)
    colors = [(0, 0, 1), (1, 1, 0), (1, 0, 0)]  # Blue to yellow to red
    cmap_name = 'gradient_cmap'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors)
    for i in range(len(search_dist_mu)):
        mu = (search_dist_mu * (q_max - q_min) + q_min)[i, :2]
        sigma = (search_dist_sigma * (q_max - q_min) + q_min)[i, :2, :2]
        eigenvalues, eigenvectors = np.linalg.eigh(sigma)
        angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
        width, height = 2 * np.sqrt(5.991 * eigenvalues)  # 95% confidence interval
        color = cm(i / len(search_dist_mu))
        ellipse = Ellipse(xy=mu, width=width, height=height, angle=angle, edgecolor=color, lw=2, fill=False)
        plt.gca().add_patch(ellipse)

    if start_pos is not None:
        visualize_robot(start_pos, ax=ax1, s=s, color="#FFBF0F")
    if goal_pos is not None:
        visualize_robot(goal_pos, ax=ax1, s=s, color="#FF9999")

    plt.xlim(0, occ_g.shape[0] * 0.1)
    plt.ylim(0, occ_g.shape[1] * 0.1)

    # Remove paddings
    plt.axis('off')  #  remove axis
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)

    if show:
        plt.show()
    if save:
        fig1.savefig(file_name, dpi=fig1.dpi)
        plt.close()

def visualize_diffusion(
    occ_g, diffusion_process, start_pos, goal_pos, max_num=float('inf'), show=True, save=True, file_name=None
):
    suffix = '.gif'
    file_name_base = file_name[:-len(suffix)]
    with imageio.get_writer(file_name, mode='I') as writer:
        for i in range(diffusion_process.shape[0]):
            frame_file_name = file_name_base + '_{}.png'.format(i)
            visualize_nodes_local(occ_g, diffusion_process[i, :, :], start_pos, goal_pos, show=False, save=True, file_name=frame_file_name, title='Diffusion step {}'.format(i))
            image = imageio.imread(frame_file_name)
            writer.append_data(image)
            os.remove(frame_file_name)


def visualize_diffusion_with_guidance(
    occ_g, diffusion_process, uncond_process, cond_process, start_pos, goal_pos, max_num=float('inf'), show=True, save=True, file_name=None
):
    suffix = '.gif'
    file_name_base = file_name[:-len(suffix)]
    with imageio.get_writer(file_name, mode='I') as writer:
        for i in range(diffusion_process.shape[0]):
            frame_file_name = file_name_base + '_{}.png'.format(i)
            visualize_nodes_local_with_guidance(occ_g, diffusion_process[i, :, :], uncond_process[i, :, :], cond_process[i, :, :], start_pos,
                                                goal_pos, show=False, save=True, file_name=frame_file_name, title='Diffusion step {}'.format(i))
            image = imageio.imread(frame_file_name)
            writer.append_data(image)
            os.remove(frame_file_name)


def visualize_tree(mesh, occ_g, G, start_pos=None, goal_pos=None, cur_pos=None, target_pos=None, show=True, save=False, file_name=None, string=False):
    fig1 = plt.figure(figsize=(10, 10), dpi=100)
    img = plt.imread("tile.jpg")

    s = 10 / occ_g.shape[0] * 60

    ax1 = fig1.add_subplot(111, aspect="equal")
    ax1.imshow(img, extent=[0, 10, 0, 10])
    for i in range(occ_g.shape[0]):
        for j in range(occ_g.shape[1]):
            if occ_g[i, j] == 1:
                plt.scatter(
                    (i + 0.5) * 0.1,
                    (j + 0.5) * 0.1,
                    color="black",
                    marker="s",
                    s=s**2,
                    alpha=1,
                )  # init

    # visualize node base positions
    nodes = list(G.nodes())
    # print(nodes)
    if string:
        node_pos = [string_to_numpy(node[1:-1]) for node in nodes]
    else:
        node_pos = [node_to_numpy(G, node) for node in nodes]

    # visualize edges
    for u, v in G.edges:
        if string:
            u_pos = string_to_numpy(u[1:-1])
            v_pos = string_to_numpy(v[1:-1])
        else:
            u_pos = node_to_numpy(G, u)
            v_pos = node_to_numpy(G, v)

        plt.plot((u_pos[0], v_pos[0]), (u_pos[1], v_pos[1]), "go-")

    # node_base_pos = [n[:2] for n in node_pos]
    # print(node_base_pos)
    if len(node_pos) > 0:
        # plt.scatter(curr_node_posns[:,0], curr_node_posns[:,1], s = 50, color = 'green')
        for i, pos in enumerate(node_pos):
            visualize_robot(pos, color="g", ax=ax1, s=s)
            # plt.text(pos[0], pos[1], str(i), color="black", fontsize=12)

    if start_pos is not None:
        visualize_robot(start_pos, ax=ax1, s=s, color="y")
    if goal_pos is not None:
        visualize_robot(goal_pos, ax=ax1, s=s, color="r")
    if cur_pos is not None:
        visualize_robot(cur_pos, ax=ax1, s=s, color="#FFBF0F")
    if target_pos is not None:
        visualize_robot(target_pos, ax=ax1, s=s, color="#FF9999")

    plt.xlim(0, occ_g.shape[0] * 0.1)
    plt.ylim(0, occ_g.shape[1] * 0.1)

    # Remove paddings
    plt.axis('off')  # remove axis
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)

    if show:
        plt.show()
    if save:
        fig1.savefig(file_name, dpi=fig1.dpi)
        plt.close()


def visualize_robot(robot_state, color="green", ax=None, s=15):
    link_length = LINK_LENGTH
    base_marker_size = (s * BASE_SIZE * 10) ** 2  # 30 * 30
    link_line_wdith = s * LINK_WIDTH * 10  # s correspond to 0.1m
    base_x, base_y = robot_state[:2]
    if len(robot_state) > 2:
        angle1 = robot_state[2]
        angle2 = robot_state[3]
        angle3 = robot_state[4]
        angle4 = robot_state[5]
        angle5 = robot_state[6]
        angle6 = robot_state[7]

        x1 = base_x + math.sin(angle1) * link_length
        y1 = base_y + math.cos(angle1) * link_length

        x2 = x1 + math.sin(angle2 + angle1) * link_length
        y2 = y1 + math.cos(angle2 + angle1) * link_length

        x3 = x2 + math.sin(angle2 + angle1 + angle3) * link_length
        y3 = y2 + math.cos(angle2 + angle1 + angle3) * link_length

        x4 = x3 + math.sin(angle2 + angle1 + angle3 + angle4) * link_length
        y4 = y3 + math.cos(angle2 + angle1 + angle3 + angle4) * link_length

        x5 = x4 + math.sin(angle2 + angle1 + angle3 + angle4 + angle5) * link_length
        y5 = y4 + math.cos(angle2 + angle1 + angle3 + angle4 + angle5) * link_length

        x6 = (
            x5
            + math.sin(angle2 + angle1 + angle3 + angle4 + angle5 + angle6) * link_length
        )
        y6 = (
            y5
            + math.cos(angle2 + angle1 + angle3 + angle4 + angle5 + angle6) * link_length
        )

    plt.scatter(base_x, base_y, color=color, marker="s", s=base_marker_size, alpha=1, edgecolors='black', zorder=10)  # base

    outline = mpe.withStroke(linewidth=link_line_wdith+2, foreground='black')
    if len(robot_state) > 2:
        ax.plot((base_x, x1), (base_y, y1), "-", color=color, linewidth=link_line_wdith, ms=link_line_wdith, path_effects=[outline], zorder=11)
        ax.plot((x1, x2), (y1, y2), "-", color=color, linewidth=link_line_wdith, ms=link_line_wdith, path_effects=[outline], zorder=11)
        ax.plot((x1, x2), (y1, y2), "-", color=color, linewidth=link_line_wdith, ms=link_line_wdith, path_effects=[outline], zorder=11)
        ax.plot((x2, x3), (y2, y3), "-", color=color, linewidth=link_line_wdith, ms=link_line_wdith, path_effects=[outline], zorder=11)
        ax.plot((x3, x4), (y3, y4), "-", color=color, linewidth=link_line_wdith, ms=link_line_wdith, path_effects=[outline], zorder=11)
        ax.plot((x4, x5), (y4, y5), "-", color=color, linewidth=link_line_wdith, ms=link_line_wdith, path_effects=[outline], zorder=11)
        ax.plot((x5, x6), (y5, y6), "-", color=color, linewidth=link_line_wdith, ms=link_line_wdith, path_effects=[outline], zorder=11)


"""
Transform APIs
"""


def normalize_sample(state, low, high):
    # new_state = np.zeros_like(state)
    # for i in range(len(state)):
    #     mean = (high[i] + low[i]) / 2.0
    #     print(state[i], high[i])
    #     new_state[i] = (state[i] - mean) / (high[i] - mean)

    mean = (high + low) / 2.0
    new_state = (state - mean) / (high - mean)
    return new_state


def unnormalize_sample(state, low, high):
    # new_state = np.zeros_like(state)
    # for i in range(len(state)):
    #     mean = (high[i] + low[i]) / 2.0
    #     new_state[i] = mean + state[i] * (high[i] - mean)

    mean = (high + low) / 2.0
    new_state = mean + state * (high - mean)
    return new_state


def global_to_local(state, r_state, ndigits=1):
    base_x = r_state[0]
    base_y = r_state[1]

    local_cx = round(base_x, ndigits)
    local_cy = round(base_y, ndigits)

    local_state = list(state).copy()
    local_state[0] = state[0] - local_cx
    local_state[1] = state[1] - local_cy

    return local_state


def local_to_global(state, r_state, ndigits=1):
    base_x = r_state[0]
    base_y = r_state[1]

    local_cx = round(base_x, ndigits)
    local_cy = round(base_y, ndigits)

    global_state = list(state).copy()
    global_state[0] = state[0] + local_cx
    global_state[1] = state[1] + local_cy

    return global_state


def global_to_local_np(state, r_state, g_size):
    base_x = r_state[0]
    base_y = r_state[1]

    local_cx = round(base_x)
    local_cy = round(base_y)

    local_state = np.copy(state)
    local_state[:, 0] = state[:, 0] - local_cx
    local_state[:, 1] = state[:, 1] - local_cy

    return local_state


def local_to_global_tensor(states, r_state, g_size):
    base_x = r_state[0]
    base_y = r_state[1]

    if g_size % 2 == 1:
        local_cx = round(base_x)
        local_cy = round(base_y)
    else:
        local_cx = math.floor(base_x) + 0.5
        local_cy = math.floor(base_y) + 0.5

    global_states = torch.zeros_like(states)
    global_states[:, 0] = states[:, 0] + local_cx
    global_states[:, 1] = states[:, 1] + local_cy

    return global_states


def global_to_local_tensor(states, r_state, g_size):
    base_x = r_state[0]
    base_y = r_state[1]

    if g_size % 2 == 1:
        local_cx = round(base_x)
        local_cy = round(base_y)
    else:
        local_cx = math.floor(base_x) + 0.5
        local_cy = math.floor(base_y) + 0.5

    local_state = torch.zeros_like(states)
    local_state[:, 0] = states[:, 0] - local_cx
    local_state[:, 1] = states[:, 1] - local_cy

    return local_state


def get_local_center(r_state, g_size):
    base_x = r_state[0]
    base_y = r_state[1]

    if g_size % 2 == 1:
        local_cx = round(base_x)
        local_cy = round(base_y)
    else:
        local_cx = math.floor(base_x) + 0.5
        local_cy = math.floor(base_y) + 0.5

    return (local_cx, local_cy)


def get_goal_viz_local(g, v, env_size):
    dy = g[1] - v[1]
    dx = g[0] - v[0]
    theta = math.atan2(dy, dx)
    if math.fabs(dy) >= math.fabs(dx):
        x = v[0] + env_size / math.tan(theta)
        y = env_size if dy > 0 else -env_size
    else:
        x = env_size if dx > 0 else -env_size
        y = v[1] + env_size * math.tan(theta)

    return [x, y]


def is_robot_within_local_env(robot_state, env_size):
    link_size = LINK_LENGTH
    base_size = BASE_SIZE
    base_x, base_y = robot_state[:2]

    if (
        math.fabs(base_x) > env_size - base_size
        or math.fabs(base_y) > env_size - base_size
    ):
        return False

    angle1 = robot_state[2]
    angle2 = robot_state[3]
    angle3 = robot_state[4]
    angle4 = robot_state[5]
    angle5 = robot_state[6]
    angle6 = robot_state[7]

    x1 = base_x + math.sin(angle1) * link_size
    y1 = base_y + math.cos(angle1) * link_size

    x2 = x1 + math.sin(angle2 + angle1) * link_size
    y2 = y1 + math.cos(angle2 + angle1) * link_size

    x3 = x2 + math.sin(angle2 + angle1 + angle3) * link_size
    y3 = y2 + math.cos(angle2 + angle1 + angle3) * link_size

    x4 = x3 + math.sin(angle2 + angle1 + angle3 + angle4) * link_size
    y4 = y3 + math.cos(angle2 + angle1 + angle3 + angle4) * link_size

    x5 = x4 + math.sin(angle2 + angle1 + angle3 + angle4 + angle5) * link_size
    y5 = y4 + math.cos(angle2 + angle1 + angle3 + angle4 + angle5) * link_size

    x6 = x5 + math.sin(angle2 + angle1 + angle3 + angle4 + angle5 + angle6) * link_size
    y6 = y5 + math.cos(angle2 + angle1 + angle3 + angle4 + angle5 + angle6) * link_size

    res = (
        math.fabs(x1) < env_size
        and math.fabs(y1) < env_size
        and math.fabs(x2) < env_size
        and math.fabs(y2) < env_size
        and math.fabs(x3) < env_size
        and math.fabs(y3) < env_size
        and math.fabs(x4) < env_size
        and math.fabs(y4) < env_size
        and math.fabs(x5) < env_size
        and math.fabs(y5) < env_size
        and math.fabs(x6) < env_size
        and math.fabs(y6) < env_size
    )

    return res


def get_link_positions(robot_state):
    link_size = LINK_LENGTH
    base_x, base_y = robot_state[:2]

    angle1 = robot_state[2]
    angle2 = robot_state[3]
    angle3 = robot_state[4]
    angle4 = robot_state[5]
    angle5 = robot_state[6]
    angle6 = robot_state[7]

    x1 = base_x + math.sin(angle1) * link_size
    y1 = base_y + math.cos(angle1) * link_size

    x2 = x1 + math.sin(angle2 + angle1) * link_size
    y2 = y1 + math.cos(angle2 + angle1) * link_size

    x3 = x2 + math.sin(angle2 + angle1 + angle3) * link_size
    y3 = y2 + math.cos(angle2 + angle1 + angle3) * link_size

    x4 = x3 + math.sin(angle2 + angle1 + angle3 + angle4) * link_size
    y4 = y3 + math.cos(angle2 + angle1 + angle3 + angle4) * link_size

    x5 = x4 + math.sin(angle2 + angle1 + angle3 + angle4 + angle5) * link_size
    y5 = y4 + math.cos(angle2 + angle1 + angle3 + angle4 + angle5) * link_size

    x6 = x5 + math.sin(angle2 + angle1 + angle3 + angle4 + angle5 + angle6) * link_size
    y6 = y5 + math.cos(angle2 + angle1 + angle3 + angle4 + angle5 + angle6) * link_size

    return [x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6]


def get_link_positions_torch(robot_state):
    link_size = LINK_LENGTH
    base_x = robot_state[:, 0].view(-1, 1)
    base_y = robot_state[:, 1].view(-1, 1)
    angle1 = robot_state[:, 2].view(-1, 1)
    angle2 = robot_state[:, 3].view(-1, 1)
    angle3 = robot_state[:, 4].view(-1, 1)
    angle4 = robot_state[:, 5].view(-1, 1)
    angle5 = robot_state[:, 6].view(-1, 1)
    angle6 = robot_state[:, 7].view(-1, 1)

    x1 = base_x + torch.sin(angle1) * link_size
    y1 = base_y + torch.cos(angle1) * link_size

    x2 = x1 + torch.sin(angle2 + angle1) * link_size
    y2 = y1 + torch.cos(angle2 + angle1) * link_size

    x3 = x2 + torch.sin(angle2 + angle1 + angle3) * link_size
    y3 = y2 + torch.cos(angle2 + angle1 + angle3) * link_size

    x4 = x3 + torch.sin(angle2 + angle1 + angle3 + angle4) * link_size
    y4 = y3 + torch.cos(angle2 + angle1 + angle3 + angle4) * link_size

    x5 = x4 + torch.sin(angle2 + angle1 + angle3 + angle4 + angle5) * link_size
    y5 = y4 + torch.cos(angle2 + angle1 + angle3 + angle4 + angle5) * link_size

    x6 = x5 + torch.sin(angle2 + angle1 + angle3 + angle4 + angle5 + angle6) * link_size
    y6 = y5 + torch.cos(angle2 + angle1 + angle3 + angle4 + angle5 + angle6) * link_size

    return torch.cat((x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6), dim=1)


"""
FkTorch
"""


class FkTorch():
    def __init__(self, device):
        self.device = device

    def set_device(self, device):
        self.device = device

    def get_link_positions(self, robot_state):
        link_size = LINK_LENGTH
        base_x = robot_state[:, 0].view(-1, 1)
        base_y = robot_state[:, 1].view(-1, 1)
        angle1 = robot_state[:, 2].view(-1, 1)
        angle2 = robot_state[:, 3].view(-1, 1)
        angle3 = robot_state[:, 4].view(-1, 1)
        angle4 = robot_state[:, 5].view(-1, 1)
        angle5 = robot_state[:, 6].view(-1, 1)
        angle6 = robot_state[:, 7].view(-1, 1)

        x1 = base_x + torch.sin(angle1) * link_size
        y1 = base_y + torch.cos(angle1) * link_size

        x2 = x1 + torch.sin(angle2 + angle1) * link_size
        y2 = y1 + torch.cos(angle2 + angle1) * link_size

        x3 = x2 + torch.sin(angle2 + angle1 + angle3) * link_size
        y3 = y2 + torch.cos(angle2 + angle1 + angle3) * link_size

        x4 = x3 + torch.sin(angle2 + angle1 + angle3 + angle4) * link_size
        y4 = y3 + torch.cos(angle2 + angle1 + angle3 + angle4) * link_size

        x5 = x4 + torch.sin(angle2 + angle1 + angle3 + angle4 + angle5) * link_size
        y5 = y4 + torch.cos(angle2 + angle1 + angle3 + angle4 + angle5) * link_size

        x6 = x5 + torch.sin(angle2 + angle1 + angle3 + angle4 + angle5 + angle6) * link_size
        y6 = y5 + torch.cos(angle2 + angle1 + angle3 + angle4 + angle5 + angle6) * link_size

        return torch.cat((x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6), dim=1)


"""
For global
"""


def normalize_state(state, env_size=10):
    new_state = state.copy()
    new_state[0] = state[0] - env_size // 2
    new_state[1] = state[1] - env_size // 2
    return new_state


def unnormalize_state(state, env_size=10):
    new_state = state.copy()
    new_state[0] = state[0] + env_size // 2
    new_state[1] = state[1] + env_size // 2
    return new_state


"""
K-shortest path
from: https://github.com/guilhermemm/k-shortest-path
"""


def k_shortest_paths(G, source, target, k=1, weight='weight', local_env_size=0.2):
    """Returns the k-shortest paths from source to target in a weighted graph G.

    Parameters
    ----------
    G : NetworkX graph

    source : node
       Starting node

    target : node
       Ending node

    k : integer, optional (default=1)
        The number of shortest paths to find

    weight: string, optional (default='weight')
       Edge data key corresponding to the edge weight

    Returns
    -------
    lengths, clipped_local_paths : lists
       Returns a tuple with two lists.
       The first list stores the length of each k-shortest path.
       The second list stores each non-repetitive clipped local path.

    Raises
    ------
    NetworkXNoPath
       If no path exists between source and target.

    Examples
    --------
    >>> G=nx.complete_graph(5)
    >>> print(k_shortest_paths(G, 0, 4, 4))
    ([1, 2, 2, 2], [[0, 4], [0, 1, 4], [0, 2, 4], [0, 3, 4]])

    Notes
    ------
    Edge weight attributes must be numerical and non-negative.
    Distances are calculated as sums of weighted edges traversed.

    """
    if source == target:
        return ([0], [[source]])

    length, path = nx.single_source_dijkstra(G, source, target, weight=weight)
    if target not in path:
        raise nx.NetworkXNoPath("node %s not reachable from %s" % (source, target))
    start_pos = node_to_numpy(G, source)

    lengths = [length]
    paths = [path]
    np_path = [node_to_numpy(G, node) for node in path]
    local_path = [global_to_local(n, start_pos) for n in np_path]
    clipped_path = clip_path(local_path, local_env_size=local_env_size)
    clipped_paths = [clipped_path]
    c = count()
    B = []
    G_original = G.copy()
    start_time = time.time()
    # for i in range(1, k):
    while len(clipped_paths) < k:
        for j in range(len(paths[-1]) - 1):
            spur_node = paths[-1][j]
            root_path = paths[-1][:j + 1]

            edges_removed = []
            for c_path in paths:
                if len(c_path) > j and root_path == c_path[:j + 1]:
                    u = c_path[j]
                    v = c_path[j + 1]
                    if G.has_edge(u, v):
                        edge_attr = G.edges[u, v]
                        G.remove_edge(u, v)
                        edges_removed.append((u, v, edge_attr))

            for n in range(len(root_path) - 1):
                node = root_path[n]
                # out-edges
                edges = list(G.edges(nbunch=node, data=True))
                for u, v, edge_attr in edges:
                    G.remove_edge(u, v)
                    edges_removed.append((u, v, edge_attr))

                if G.is_directed():
                    # in-edges
                    for u, v, edge_attr in G.in_edges_iter(node, data=True):
                        G.remove_edge(u, v)
                        edges_removed.append((u, v, edge_attr))

            try:
                spur_path_length, spur_path = nx.single_source_dijkstra(G, spur_node, target, weight=weight)
                if target in spur_path:
                    total_path = root_path[:-1] + spur_path
                    total_path_length = get_path_length(G_original, root_path, weight) + spur_path_length
                    heappush(B, (total_path_length, next(c), total_path))
            except:
                print("Warning: single_source_dijkstra failed")

            for e in edges_removed:
                u, v, edge_attr = e
                G.add_edge(u, v, **edge_attr)

        if B:
            (l, _, p) = heappop(B)
            paths.append(p)
            path = [node_to_numpy(G, node) for node in p]
            local_path = [global_to_local(n, start_pos) for n in path]
            clipped_path = clip_path(local_path, local_env_size=local_env_size)
            if not is_path_in_list(clipped_path, clipped_paths):
                lengths.append(l)
                clipped_paths.append(clipped_path)
        else:
            break

        # break if it takes too long (more than 60s)
        if time.time() - start_time > 120:
            print("k-shortest path algorithm taking more than 120s, breaking!")
            break

    return (lengths, clipped_paths)


def is_path_in_list(new_path, paths):
    for path in paths:
        if new_path == path:
            return True

    return False


def get_path_length(G, path, weight='weight'):
    length = 0
    if len(path) > 1:
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]

            length += G.edges[u, v].get(weight, 1)

    return length


def get_ebsa_path(occ_grid, start_pos, target_pos):
    occ_grid_resolution = 0.1
    astar_start = (start_pos[0] / occ_grid_resolution, start_pos[1] / occ_grid_resolution, start_pos[2])
    astar_goal = (target_pos[0] / occ_grid_resolution, target_pos[1] / occ_grid_resolution, target_pos[2])

    base_expert_planner = AStar(astar_start, astar_goal, occ_grid)
    if base_expert_planner.run():
        astar_path = base_expert_planner.reconstruct_path()
    else:
        astar_path = None

    if astar_path is not None:
        expert_base_path = []
        for i in range(len(astar_path)):
            expert_base_path.append([astar_path[i][0] * occ_grid_resolution, astar_path[i][1] * occ_grid_resolution])

        num_waypoints = len(expert_base_path)
        node1_pos_arm = np.array(start_pos)[2:]
        node2_pos_arm = np.array(target_pos)[2:]
        arm_diff = node2_pos_arm - node1_pos_arm
        ebsa_path = []
        for i in range(num_waypoints):
            arm_step_size = arm_diff / (num_waypoints - 1)
            pos_arm = node1_pos_arm + arm_step_size * i
            ebsa_path.append(expert_base_path[i] + pos_arm.tolist())
    else:
        ebsa_path = []

    return ebsa_path