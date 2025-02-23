import numpy as np
import math
import matplotlib.pyplot as plt
import torch
import pytorch_kinematics as pk
import scipy
import os.path as osp
from pytorch3d.transforms import Transform3d
import os
from PIL import Image
import reeds_shepp as rs
import networkx as nx
from matplotlib.patches import Ellipse
from matplotlib.colors import LinearSegmentedColormap
import json

from nrp.planner.decomposed_planner import HybridAStar

CUR_DIR = osp.dirname(osp.abspath(__file__))

TURNING_RADIUS = 0.1  # 0.75
STEP_SIZE = 0.2  # 0.2

"""
Constants
"""

"""
Env utils
"""


def get_occ_grid(env_dir):
    with open(os.path.join(CUR_DIR, "map/occ_grid.npy"), "rb") as f:
        occ_grid = np.load(f)

    return occ_grid


def get_prm(env_dir):
    g = nx.read_graphml(osp.join(CUR_DIR, "map/dense_g.graphml"))
    return g


def get_mesh_path(env_dir):
    return osp.join(CUR_DIR, "map/rls_mesh.obj")


def get_start_goal(env_dir):
    with open(osp.join(env_dir, "test_path.json"), "r") as f:
        start_goal = json.load(f)

    return start_goal


"""
Path Utils
"""


def rrt_extend_path(
    env, path, step_size=STEP_SIZE, turning_radius=TURNING_RADIUS, intermediate=False
):
    final_path = [path[0]]
    for i in range(1, len(path)):
        v1 = path[i - 1]
        v2 = path[i]
        if intermediate:
            res_path = rrt_extend_intermediate(env, v1, v2, step_size, turning_radius)
        else:
            res_path = rrt_extend(env, v1, v2, step_size, turning_radius)

        if len(res_path) > 1:
            final_path += res_path[1:]
        if not np.allclose(np.array(res_path[-1]), np.array(v2)):
            # print("collision detected!!!")
            break

    return final_path


def rrt_extend(maze, v, g, step_size=STEP_SIZE, turning_radius=TURNING_RADIUS):
    node1_pos = np.array(v)

    node1_pos_arm = node1_pos[3:]
    base_traj, arm_step_size, num_points_per_edge = discretize_edge(
        v, g, step_size, turning_radius
    )

    res_path = [node1_pos.tolist()]
    nodepos = np.zeros(len(node1_pos))
    for i in range(1, num_points_per_edge):
        nodepos[:3] = base_traj[i][:3]
        nodepos[3:] = node1_pos_arm + arm_step_size * i
        if not maze.pb_ompl_interface.is_state_valid(nodepos.tolist()):
            break

        if len(res_path) == 1:
            res_path.append(nodepos.tolist())
        else:
            res_path[1] = nodepos.tolist()

    # to avoid float calculation precision problem
    if np.allclose(np.array(res_path[-1]), np.array(g), atol=1e-4):
        res_path[-1] = g

    return res_path


def rrt_extend_intermediate(
    maze, v, g, step_size=STEP_SIZE, turning_radius=TURNING_RADIUS
):
    node1_pos = np.array(v)
    node2_pos = np.array(g)
    node1_pos_arm = node1_pos[3:]
    base_traj, arm_step_size, num_points_per_edge = discretize_edge(
        v, g, step_size, turning_radius
    )

    res_path = [node1_pos.tolist()]
    nodepos = np.zeros(len(node1_pos))
    for i in range(1, num_points_per_edge):
        nodepos[:3] = base_traj[i][:3]
        nodepos[3:] = node1_pos_arm + arm_step_size * i
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


def expand_until_local_env_edge(
    v, g, step_size=STEP_SIZE, turning_radius=TURNING_RADIUS, local_env_size=2.0
):
    path = interpolate([v, g], step_size=step_size)
    for i, p in enumerate(path):
        if not is_robot_within_local_env(p, v, local_env_size):
            return path[i - 1]

    return g


def is_edge_free(
    maze, node1_state, node2_state, step_size=STEP_SIZE, turning_radius=TURNING_RADIUS
):
    node1_pos = np.array(node1_state)
    node2_pos = np.array(node2_state)

    if not maze.pb_ompl_interface.is_state_valid(node1_pos.tolist()):
        return False

    if not maze.pb_ompl_interface.is_state_valid(node2_pos.tolist()):
        return False

    if np.allclose(node1_pos, node2_pos, atol=step_size):
        return True

    node1_pos_arm = node1_pos[3:]
    base_traj, arm_step_size, num_points_per_edge = discretize_edge(
        node1_pos, node2_pos, step_size, turning_radius
    )

    # assert(np.max(step) < step_size and np.min(step) > -step_size)
    nodepos = np.zeros(len(node1_pos))
    for i in range(1, num_points_per_edge):
        nodepos[:3] = base_traj[i][:3]
        nodepos[3:] = node1_pos_arm + arm_step_size * i
        if not maze.pb_ompl_interface.is_state_valid(nodepos.tolist()):
            return False
    return True


def calc_edge_len(s1, s2, turning_radius=TURNING_RADIUS, arm_len_weight=0.125):
    node1_pos = np.array(s1).reshape(-1, 11)
    node2_pos = np.array(s2).reshape(1, 11)

    node1_pos_base = node1_pos[:, :3]
    node2_pos_base = node2_pos[:, :3]
    node1_pos_arm = node1_pos[:, 3:]
    node2_pos_arm = node2_pos[:, 3:]

    # we can't calculate rs_length in parallel unfortunately
    base_len = np.zeros((node1_pos_base.shape[0]))
    for i in range(node1_pos_base.shape[0]):
        base_len[i] = rs.path_length(
            node1_pos_base[i], node2_pos_base[0], turning_radius
        )
    # base_len = math.sqrt(float(np.sum((node1_pos_base-node2_pos_base)**2)))
    arm_len = np.linalg.norm((node2_pos_arm - node1_pos_arm), axis=-1)
    res = base_len + arm_len_weight * arm_len
    if res.shape[0] == 1:
        res = res.item()

    return res


def calc_edge_len_norm(s1, s2, turning_radius=TURNING_RADIUS, arm_len_weight=0.125):
    node1_pos = np.array(s1).reshape(-1, 11)
    node2_pos = np.array(s2).reshape(1, 11)

    node1_pos_base = node1_pos[:, :3]
    node2_pos_base = node2_pos[:, :3]
    node1_pos_arm = node1_pos[:, 3:]
    node2_pos_arm = node2_pos[:, 3:]

    # we can't calculate rs_length in parallel unfortunately
    base_len = np.zeros((node1_pos_base.shape[0]))
    for i in range(node1_pos_base.shape[0]):
        base_len[i] = rs.path_length(
            node1_pos_base[i], node2_pos_base[0], turning_radius
        )
    # base_len = math.sqrt(float(np.sum((node1_pos_base-node2_pos_base)**2)))
    arm_len = np.linalg.norm((node2_pos_arm - node1_pos_arm), axis=-1)
    res = np.linalg.norm((base_len, arm_len_weight * arm_len), axis=0)
    # res = base_len
    if res.shape[0] == 1:
        res = res.item()

    return res


def calc_edge_len_max(s1, s2, turning_radius=TURNING_RADIUS):
    node1_pos = np.array(s1).reshape(-1, 11)
    node2_pos = np.array(s2).reshape(1, 11)

    node1_pos_base = node1_pos[:, :3]
    node2_pos_base = node2_pos[:, :3]
    node1_pos_arm = node1_pos[:, 3:]
    node2_pos_arm = node2_pos[:, 3:]

    # we can't calculate rs_length in parallel unfortunately
    base_len = np.zeros((node1_pos_base.shape[0]))
    for i in range(node1_pos_base.shape[0]):
        base_len[i] = rs.path_length(
            node1_pos_base[i], node2_pos_base[0], turning_radius
        )
    # base_len = math.sqrt(float(np.sum((node1_pos_base-node2_pos_base)**2)))
    arm_len = np.max(np.abs((node2_pos_arm - node1_pos_arm)), axis=-1)
    res = np.max((base_len, arm_len), axis=0)
    if res.shape[0] == 1:
        res = res.reshape(-1)

    return res


def calc_edge_len_base(s1, s2, turning_radius=TURNING_RADIUS):
    node1_pos = np.array(s1)
    node2_pos = np.array(s2)

    base_len = rs.path_length(node1_pos, node2_pos, turning_radius)
    return base_len


def calc_path_len(path, turning_radius=TURNING_RADIUS, arm_len_weight=0.125):
    length = 0
    for i, node in enumerate(path):
        if i == 0:
            continue
        length += calc_edge_len(path[i - 1], node, turning_radius, arm_len_weight)

    return length


def calc_path_len_norm(path, turning_radius=TURNING_RADIUS, arm_len_weight=0.125):
    length = 0
    for i, node in enumerate(path):
        if i == 0:
            continue
        length += calc_edge_len_norm(
            path[i - 1], node, turning_radius, arm_len_weight
        ).item()

    return length


def calc_path_len_max(path, turning_radius=TURNING_RADIUS):
    length = 0
    for i, node in enumerate(path):
        if i == 0:
            continue
        length += calc_edge_len_max(path[i - 1], node, turning_radius).item()

    return length


def calc_path_len_base(path, turning_radius=TURNING_RADIUS):
    length = 0
    for i, node in enumerate(path):
        if i == 0:
            continue
        length += rs.path_length(path[i - 1], node, turning_radius)

    return length


def interpolate(
    path, step_size=STEP_SIZE, num_points_per_edge=None, turning_radius=TURNING_RADIUS
):
    new_path = []
    # print(path)
    for i in range(1, len(path)):
        node1_pos = np.array(path[i - 1])
        node2_pos = np.array(path[i])

        node1_pos_arm = node1_pos[3:]
        base_traj, arm_step_size, num_points_per_edge = discretize_edge(
            node1_pos, node2_pos, step_size, turning_radius
        )

        nodepos = np.zeros(len(node1_pos))
        for j in range(num_points_per_edge):
            nodepos[:3] = base_traj[j][:3]
            nodepos[3:] = node1_pos_arm + arm_step_size * j
            new_path.append(nodepos.tolist())

    new_path.append(path[-1])

    return new_path


def interpolate_base(
    path, step_size=0.2, num_points_per_edge=None, turning_radius=TURNING_RADIUS
):
    new_path = []
    # print(path)
    for i in range(1, len(path)):
        node1_pos = np.array(path[i - 1])
        node2_pos = np.array(path[i])

        node1_pos_base = node1_pos[:3]
        node2_pos_base = node2_pos[:3]
        node1_pos_arm = node1_pos[3:]
        node2_pos_arm = node2_pos[3:]

        node1_pos_arm = node1_pos[3:]
        base_diff = rs.path_length(node1_pos_base, node2_pos_base, turning_radius)
        num_points_per_edge = math.ceil(np.max(np.abs(base_diff)) / step_size) + 1
        base_step_size = base_diff / (num_points_per_edge - 1)
        base_traj = rs_path_sample(
            node1_pos_base, node2_pos_base, turning_radius, base_step_size
        )
        if not np.allclose(base_traj[-1][:3], node2_pos_base):
            if len(base_traj) < num_points_per_edge:
                base_traj.append(node2_pos_base)
            else:
                base_traj[-1] = node2_pos_base

        arm_diff = node2_pos_arm - node1_pos_arm
        arm_step_size = arm_diff / (num_points_per_edge - 1)

        nodepos = np.zeros(len(node1_pos))
        for j in range(num_points_per_edge):
            nodepos[:3] = base_traj[j][:3]
            nodepos[3:] = node1_pos_arm + arm_step_size * j
            new_path.append(nodepos.tolist())

    new_path.append(path[-1])

    return new_path


def cal_col_prop(maze, path):
    # TODO
    return 1.0
    # interpolated_path = interpolate(path)

    # if len(interpolated_path) == 0:
    #     return 1

    # total_col_cnt = 0
    # for nodepos in interpolated_path:
    #     if not maze.pb_ompl_interface.is_state_valid(nodepos):
    #         total_col_cnt += 1

    # return total_col_cnt / len(interpolated_path)


def get_path_type(path, turning_radius=TURNING_RADIUS):
    res = []
    for i, node in enumerate(path):
        if i == 0:
            continue

        res += list(rs.path_type(path[i - 1][:3], node[:3], turning_radius))
    return res


def discretize_edge(v, g, step_size=STEP_SIZE, turning_radius=TURNING_RADIUS):
    node1_pos = np.array(v)
    node2_pos = np.array(g)

    node1_pos_base = node1_pos[:3]
    node2_pos_base = node2_pos[:3]
    node1_pos_arm = node1_pos[3:]
    node2_pos_arm = node2_pos[3:]

    if np.allclose(v, g):
        return [node1_pos_base], 0, 1

    arm_diff = node2_pos_arm - node1_pos_arm

    if not np.allclose(node1_pos_base, node2_pos_base):
        # extract reeds shepp curve on base
        base_diff = rs.path_length(node1_pos_base, node2_pos_base, turning_radius)
    else:
        base_diff = 0

    diff = np.concatenate((arm_diff, np.array([base_diff])), axis=-1)
    num_points_per_edge = math.ceil(np.max(np.abs(diff)) / step_size) + 1

    base_step_size = base_diff / (num_points_per_edge - 1)
    if base_diff == 0:
        base_traj = [node1_pos_base] * num_points_per_edge
    else:
        base_traj = rs_path_sample(
            node1_pos_base, node2_pos_base, turning_radius, base_step_size
        )
        if not np.allclose(base_traj[-1][:3], node2_pos_base):
            if len(base_traj) < num_points_per_edge:
                base_traj.append(node2_pos_base)
            else:
                base_traj[-1] = node2_pos_base

    if len(base_traj) != num_points_per_edge:
        print(len(base_traj), num_points_per_edge, node1_pos, node2_pos)
        raise Exception()

    arm_step_size = arm_diff / (num_points_per_edge - 1)

    return base_traj, arm_step_size, num_points_per_edge

    # nodepos = np.zeros(len(node1_pos))
    # res_path = []
    # for i in range(num_points_per_edge):
    #     nodepos[:3] = base_traj[i][:3]
    #     nodepos[3:] = node1_pos_arm + arm_step_size * i
    #     res_path.append(nodepos)

    # return res_path


def enforce_pi(theta):
    if theta > math.pi:
        theta = theta - 2 * math.pi
    elif theta < -math.pi:
        theta = theta + 2 * math.pi
    return theta


def rs_path_sample(node1_pos_base, node2_pos_base, turning_radius, base_step_size):
    # if node2_pos_base[2] < 0:
    #     node2_pos_base[2] += 2 * math.pi
    base_traj = rs.path_sample(
        node1_pos_base, node2_pos_base, turning_radius, base_step_size
    )
    # base_traj = [list(t)[:3] for t in base_traj]
    res_base_traj = []
    for t in base_traj:
        t = list(t)[:3]
        t[2] = enforce_pi(t[2])
        res_base_traj.append(t)

    return res_base_traj


"""
Helpers
"""


def string_to_numpy(state):
    strlist = state.split(",")
    val_list = [float(s) for s in strlist]
    return np.array(val_list)


def state_to_numpy(state):
    strlist = state.split(",")
    val_list = [float(s) for s in strlist]
    return np.array(val_list)


def node_to_numpy(G, n):
    return state_to_numpy(G.nodes[n]["coords"])


def get_free_nodes(G):
    return [n for n in G.nodes() if not G.nodes[n]["col"]]


def get_free_node_poss(G):
    return [
        state_to_numpy(G.nodes[node]["coords"])
        for node in G.nodes()
        if not G.nodes[node]["col"]
    ]


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


def get_rank(success_list):
    array = np.array(success_list)
    array = array.T

    ranks = np.zeros_like(array)
    for i in range(len(array)):
        ranks[i] = scipy.stats.rankdata(array[i])
    tmp = np.sum(ranks.T, axis=-1).tolist()
    return tmp


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
    "Decomposed": "tab:brown",
    "FIRE": "tab:pink",
    "FIRE*": "tab:pink",
    "VQMPT": "tab:brown",
    "VQMPT*": "tab:brown",
}


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
    gui=False,
):
    from nrp.env.fetch_11d.env import Fetch11DEnv

    env = Fetch11DEnv(gui=gui, add_robot=False)
    if mesh is not None:
        env.load_mesh(mesh)
    if occ_g is not None:
        if mesh is None:
            env.load_occupancy_grid(occ_g, add_box=True)
        else:
            env.load_occupancy_grid(occ_g)

    if len(curr_node_posns) > 0:
        for i, pos in enumerate(curr_node_posns):
            if sample_pos is not None and np.allclose(
                np.array(pos), np.array(sample_pos)
            ):
                continue
            if start_pos is not None and np.allclose(
                np.array(pos), np.array(start_pos)
            ):
                continue
            if goal_pos is not None and np.allclose(np.array(pos), np.array(goal_pos)):
                continue

            env.add_robot(pos, rgba=[102 / 255, 178 / 255, 255 / 255, 1])

    if viz_edge:
        if edge_path is None:
            edge_path = curr_node_posns

        for i in range(1, len(edge_path)):
            env.add_line(
                edge_path[i - 1][:2],
                edge_path[i][:2],
                colour=[102 / 255, 178 / 255, 255 / 255, 1],
            )

    if start_pos is not None:
        start_pos_tmp = start_pos.copy()
        env.add_robot(start_pos_tmp, rgba=[1, 0.749, 0.0588, 1])  # yellow
    if goal_pos is not None:
        goal_pos_tmp = goal_pos.copy()
        env.add_robot(goal_pos_tmp, rgba=[1, 0.6, 0.6, 1])  # red
    if sample_pos is not None:
        sample_pos_tmp = sample_pos.copy()
        env.add_robot(sample_pos_tmp, rgba=[0, 1, 0, 1])  # green

    img = env.get_global_img()
    print(img.shape)
    pil_img = Image.fromarray(img)
    rgb_img = pil_img.convert("RGB")
    print(rgb_img.mode)
    if show:
        rgb_img.show()
    if save:
        print(f"saving img to {file_name}...")
        rgb_img.save(file_name)

    if gui:
        input("Press anything to quit")


def visualize_nodes_global_2(
    mesh,
    occ_g,
    curr_node_posns,
    start_pos,
    goal_pos,
    show=True,
    save=False,
    file_name=None,
):
    from nrp.env.fetch_11d.env import Fetch11DEnv

    # Image from side.
    maze = Fetch11DEnv(gui=False, add_robot=False, load_floor=True)
    if mesh is not None:
        maze.load_mesh(mesh)
    if occ_g is not None:
        if mesh is None:
            maze.load_occupancy_grid(occ_g, add_box=True)
        else:
            maze.load_occupancy_grid(occ_g)

    if len(curr_node_posns) > 0:
        color_step = 1 / len(curr_node_posns)
        for i, pos in enumerate(curr_node_posns):
            tmp = i + 1
            # maze.add_robot(pos, rgba=[0 + tmp*color_step, 0 + tmp*color_step, 0 + tmp*color_step, 1])
            maze.add_robot(pos, rgba=[0.4, 0.698, 1, 1])

    if start_pos is not None:
        start_pos_tmp = start_pos.copy()
        maze.add_robot(start_pos_tmp, rgba=[1, 0.749, 0.0588, 1])  # yellow
    if goal_pos is not None:
        goal_pos_tmp = goal_pos.copy()
        maze.add_robot(goal_pos_tmp, rgba=[1, 0.6, 0.6, 1])  # red

    img = maze.get_global_img_2()
    print(img.shape)
    pil_img = Image.fromarray(img)
    rgb_img = pil_img.convert("RGB")
    print(rgb_img.mode)
    if show:
        rgb_img.show()
    if save:
        rgb_img.save(file_name)

    # input("Press anything to quit")


def visualize_nodes_local(
    occ_g,
    cur_node_pos,
    start_pos,
    goal_pos,
    max_num=50,
    color_coding=False,
    show=True,
    save=False,
    file_name=None,
):
    from nrp.env.fetch_11d.env import Fetch11DEnv

    maze = Fetch11DEnv(gui=False, add_robot=False)
    # maze.disable_visual()
    if occ_g is not None:
        maze.load_occupancy_grid(occ_g, add_box=True)

    if len(cur_node_pos) > 0:
        if len(cur_node_pos) > max_num:
            cur_node_pos = cur_node_pos[:max_num]

        if color_coding:
            color_step = 1 / len(cur_node_pos)
            for i, pos in enumerate(cur_node_pos):
                if not np.allclose(pos, start_pos):
                    pos_tmp = pos.copy()
                    pos_tmp[0] += 2
                    pos_tmp[1] += 2
                    maze.add_robot(
                        pos_tmp,
                        rgba=[
                            0 + i * color_step,
                            0 + i * color_step,
                            0 + i * color_step,
                            1,
                        ],
                    )
        else:
            for pos in cur_node_pos:
                if not np.allclose(pos, start_pos):
                    pos_tmp = pos.copy()
                    pos_tmp[0] += 2
                    pos_tmp[1] += 2
                    maze.add_robot(pos_tmp, rgba=[0, 1, 0, 1])

    if start_pos is not None:
        start_pos_tmp = start_pos.copy()
        start_pos_tmp[0] += 2
        start_pos_tmp[1] += 2
        maze.add_robot(start_pos_tmp, rgba=[1, 1, 0, 1])  # yellow

    if goal_pos is not None:
        goal_pos_tmp = goal_pos.copy()
        if math.fabs(goal_pos[0]) > 4.0 or math.fabs(goal_pos[1]) > 4.0:
            goal_dir = math.atan2(goal_pos[1], goal_pos[0])

            if math.fabs(goal_pos[0]) < math.fabs(goal_pos[1]):
                goal_pos_tmp[1] = 4 if goal_pos[1] > 0 else -4
                goal_pos_tmp[0] = goal_pos_tmp[1] / math.tan(goal_dir)
            else:
                goal_pos_tmp[0] = 4 if goal_pos[0] > 0 else -4
                goal_pos_tmp[1] = goal_pos_tmp[0] * math.tan(goal_dir)

            goal_color = [0.5, 0, 0, 0.2]
        else:
            goal_color = [1, 0, 0, 0.2]

        goal_pos_tmp[0] += 2
        goal_pos_tmp[1] += 2
        maze.add_robot(goal_pos_tmp, rgba=goal_color)  # red

    # maze.enable_visual()
    # input("Press anything to quit")

    img = maze.get_img(start_pos)
    pil_img = Image.fromarray(img)

    if show:
        pil_img.show()
    if save:
        pil_img.save(file_name)

    return img


def visualize_distributions(
    occ_g,
    search_dist_mu,
    search_dist_sigma,
    q_min,
    q_max,
    start_pos=None,
    goal_pos=None,
    show=True,
    save=False,
    file_name=None,
):
    fig1 = plt.figure(figsize=(10, 10), dpi=250)
    # img = plt.imread("tile.jpg")

    occ_grid_size = max(occ_g.shape[0], occ_g.shape[1])
    s = 10 / occ_grid_size * 60

    ax1 = fig1.add_subplot(111, aspect="equal")
    # ax1.imshow(img, extent=[0, 10, 0, 10])

    for i in range(occ_g.shape[0]):
        for j in range(occ_g.shape[1]):
            if occ_g[i, j].sum() >= 1:
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
    cmap_name = "gradient_cmap"
    cm = LinearSegmentedColormap.from_list(cmap_name, colors)
    for i in range(len(search_dist_mu)):
        mu = (search_dist_mu * (q_max - q_min) + q_min)[i, :2]
        sigma = (search_dist_sigma * (q_max - q_min) + q_min)[i, :2, :2]
        eigenvalues, eigenvectors = np.linalg.eigh(sigma)
        angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
        width, height = 2 * np.sqrt(5.991 * eigenvalues)  # 95% confidence interval
        color = cm(i / len(search_dist_mu))
        ellipse = Ellipse(
            xy=mu,
            width=width,
            height=height,
            angle=angle,
            edgecolor=color,
            lw=2,
            fill=False,
        )
        plt.gca().add_patch(ellipse)

    if start_pos is not None:
        # visualize_robot(start_pos, ax=ax1, s=s, color="#FFBF0F")
        visualize_robot(start_pos, start=True)
    if goal_pos is not None:
        # visualize_robot(goal_pos, ax=ax1, s=s, color="#FF9999")
        visualize_robot(goal_pos, goal=True)

    plt.xlim(0, occ_g.shape[0] * 0.1)
    plt.ylim(0, occ_g.shape[1] * 0.1)

    # Remove paddings
    plt.axis("off")  #  remove axis
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)

    if show:
        plt.show()
    if save:
        fig1.savefig(file_name, dpi=fig1.dpi)
        plt.close()


def visualize_tree(
    mesh,
    occ_g,
    G,
    start_pos,
    goal_pos=None,
    cur_pos=None,
    target_pos=None,
    show=True,
    save=False,
    file_name=None,
    string=False,
):
    from nrp.env.fetch_11d.env import Fetch11DEnv

    maze = Fetch11DEnv(gui=False, add_robot=False)
    if mesh is not None:
        maze.load_mesh(mesh)
    if occ_g is not None:
        if mesh is None:
            maze.load_occupancy_grid(occ_g, add_box=True)
        else:
            maze.load_occupancy_grid(occ_g)

    nodes = list(G.nodes())
    if string:
        node_pos = [string_to_numpy(node[1:-1]) for node in nodes]
    else:
        node_pos = [node_to_numpy(G, node) for node in nodes]
    for pos in node_pos:
        if np.allclose(pos, np.array(start_pos)):
            continue
        if goal_pos is not None and np.allclose(pos, np.array(goal_pos)):
            continue

        maze.add_robot(pos, rgba=[0, 1, 0, 1])

    # visualize edges
    for u, v in G.edges:
        if string:
            u_pos = string_to_numpy(u[1:-1])
            v_pos = string_to_numpy(v[1:-1])
        else:
            u_pos = node_to_numpy(G, u)
            v_pos = node_to_numpy(G, v)

        print(u_pos, v_pos)
        maze.add_line(u_pos.tolist()[:2], v_pos.tolist()[:2], colour=[0, 1, 0, 1])

    if start_pos is not None:
        start_pos_tmp = start_pos.copy()
        maze.add_robot(start_pos_tmp, rgba=[1, 1, 0, 1])  # yellow
    if goal_pos is not None:
        goal_pos_tmp = goal_pos.copy()
        maze.add_robot(goal_pos_tmp, rgba=[1, 0, 0, 1])  # red
    if cur_pos is not None:
        # visualize_robot(cur_pos, ax=ax1, s=s, color="#FFBF0F")
        maze.add_robot(cur_pos, rgba=[255 / 255, 191 / 255, 15 / 255, 1])  # red
    if target_pos is not None:
        # visualize_robot(target_pos, ax=ax1, s=s, color="#FF9999")
        maze.add_robot(target_pos, rgba=[255 / 255, 153 / 255, 153 / 255, 1])  # red

    img = maze.get_global_img()
    print(img.shape)
    pil_img = Image.fromarray(img)
    rgb_img = pil_img.convert("RGB")
    print(rgb_img.mode)
    if show:
        rgb_img.show()
    if save:
        rgb_img.save(file_name)


def visualize_tree_simple(
    occ_g,
    G,
    start_pos,
    goal_pos,
    draw_col_edges=False,
    show=True,
    save=False,
    file_name=None,
    string=False,
):
    occ_g = occ_g[:, :, 0]  # compress to 2d
    fig1 = plt.figure(figsize=(10, 10), dpi=100)

    s = 10 / occ_g.shape[0] * 60

    ax1 = fig1.add_subplot(111, aspect="equal")

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
    # nodes = list(G.nodes())
    # print(nodes)
    # node_pos = [string_to_numpy(node[1:-1]) for node in nodes]
    # node_base_pos = [n[:2] for n in node_pos]
    # print(node_base_pos)
    # if len(node_base_pos)>0:
    #     # plt.scatter(curr_node_posns[:,0], curr_node_posns[:,1], s = 50, color = 'green')
    #     for i, pos in enumerate(node_base_pos):
    #         visualize_robot(pos, ax = ax1, s=s)
    #         plt.text(pos[0], pos[1], str(i), color="black", fontsize=12)

    # visualize edges
    for u, v in G.edges:
        if string:
            plt.plot((u[0], v[0]), (u[1], v[1]), "go-")
        else:
            u_pos = node_to_numpy(G, u)
            v_pos = node_to_numpy(G, v)

            if not G.nodes[v]["col"] and not G.nodes[u]["col"]:
                plt.plot((u_pos[0], v_pos[0]), (u_pos[1], v_pos[1]), "go-")
            elif draw_col_edges:
                plt.plot((u_pos[0], v_pos[0]), (u_pos[1], v_pos[1]), "ro-")

    if start_pos is not None:
        visualize_robot(start_pos[:2], start=True, ax=ax1, s=s)
    if goal_pos is not None:
        visualize_robot(goal_pos[:2], goal=True, ax=ax1, s=s)

    plt.title("Visualization")
    plt.xlim(0, occ_g.shape[0] * 0.1)
    plt.ylim(0, occ_g.shape[1] * 0.1)
    if show:
        plt.show()
    if save:
        fig1.savefig(file_name, dpi=fig1.dpi)
        plt.close()


def visualize_robot(robot_state, start=False, goal=False, mu=False, ax=None, s=15):
    base_marker_size = (s * 0.8) ** 2  # 30 * 30
    base_x, base_y = robot_state[:2]

    if start:
        plt.scatter(
            base_x, base_y, color="yellow", marker="s", s=base_marker_size * 2, alpha=1
        )  # init
    elif goal:
        plt.scatter(
            base_x, base_y, color="red", marker="s", s=base_marker_size * 2, alpha=1
        )  # init
    else:
        plt.scatter(
            base_x, base_y, color="green", marker="s", s=base_marker_size, alpha=1
        )  # init


"""
Transform APIs
"""


def normalize_state(state, low, high):
    # new_state = np.zeros_like(state)
    # for i in range(len(state)):
    #     mean = (high[i] + low[i]) / 2.0
    #     print(state[i], high[i])
    #     new_state[i] = (state[i] - mean) / (high[i] - mean)

    mean = (high + low) / 2.0
    new_state = (state - mean) / (high - mean)
    return new_state


def unnormalize_state(state, low, high):
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


def is_robot_outside_local_env(cur_state, state, env_size):
    # Fetch robot radius ~= 0.25
    if (
        math.fabs(state[0] - cur_state[0]) < env_size + 0.25
        or math.fabs(state[1] - cur_state[1]) < env_size + 0.25
    ):
        return False

    # check linkpos
    state = torch.tensor(state).view(1, -1)
    linkpos = global_fk.get_link_positions(state).numpy().reshape(-1, 3)[:, :2]
    relative_linkpos = np.absolute(linkpos - cur_state[:2])
    return (relative_linkpos[:, 0] > env_size).all() and (
        relative_linkpos[:, 1] > env_size
    ).all()


def is_robot_within_local_env(cur_state, state, env_size):
    # Fetch robot radius ~= 0.25
    if (
        math.fabs(state[0] - cur_state[0]) > env_size - 0.25
        or math.fabs(state[1] - cur_state[1]) > env_size - 0.25
    ):
        return False

    # check linkpos
    state = torch.tensor(state).view(1, -1)
    linkpos = global_fk.get_link_positions(state).numpy().reshape(-1, 3)[:, :2]
    relative_linkpos = np.absolute(linkpos - cur_state[:2])
    return (relative_linkpos[:, 0] < env_size).all() and (
        relative_linkpos[:, 1] < env_size
    ).all()


def get_last_waypoint_within_local_env(cur_pos, target_pos, local_env_size=2.0):
    interpolated_path = interpolate([cur_pos, target_pos])
    for i, waypoint in enumerate(interpolated_path):
        if not is_robot_within_local_env(cur_pos, waypoint, local_env_size):
            return interpolated_path[i - 1]

    return None


def path_pass_through_local_env(start_pos, p1, p2, env_size):
    path = interpolate([p1, p2])
    for p in path:
        if not is_robot_outside_local_env(start_pos, p, env_size):
            return True

    return False


# def get_voxel_coords(bs, occ_grid_dim, occ_grid_dim_z, device):
#     # print("here", bs)
#     # voxel_coords = []
#     # for bs in range(bs):
#     #     for i in np.arange(occ_grid_dim):
#     #         for j in np.arange(occ_grid_dim):
#     #             for k in np.arange(occ_grid_dim_z):
#     #                 voxel_coords.append([bs, k, j, i])
#     # voxel_coords = torch.tensor(np.array(voxel_coords), device=device).view(-1, 4)

#     voxel_coords = np.indices((occ_grid_dim_z, occ_grid_dim, occ_grid_dim, bs)).T.reshape(-1, 4)
#     voxel_coords[:, [0, 1, 2, 3]] = voxel_coords[:, [3, 0, 1, 2]]
#     voxel_coords = torch.tensor(voxel_coords, device=device).view(-1, 4)
#     return voxel_coords


def get_voxel_and_feat(occ_grid):
    indices = (occ_grid > 0).nonzero().contiguous()  # N x 4

    voxel_feat = torch.clone(indices).to(torch.float)  # N x 4

    voxel_feat[:, 0] = 1
    voxel_feat[:, 1] = indices[:, 3] / 10  # z
    voxel_feat[:, 2] = indices[:, 2] / 10 - 2  # y
    voxel_feat[:, 3] = indices[:, 1] / 10 - 2  # x

    # print(indices)
    # print(voxel_feat)

    return voxel_feat, indices


def add_pos_channels(occ_grid):
    occ_grid_tmp = torch.zeros(
        (4, occ_grid.shape[0], occ_grid.shape[1], occ_grid.shape[2]),
        device=occ_grid.device,
    )

    voxel_coords = torch.from_numpy(
        np.indices((occ_grid.shape[2], occ_grid.shape[1], occ_grid.shape[0])).T
    )
    # print(voxel_coords)

    occ_grid_tmp[0, :, :, :] = occ_grid
    occ_grid_tmp[1, :, :, :] = voxel_coords[:, :, :, 2] / 10.0 - 1.95
    occ_grid_tmp[2, :, :, :] = voxel_coords[:, :, :, 1] / 10.0 - 1.95
    occ_grid_tmp[3, :, :, :] = voxel_coords[:, :, :, 0] / 10.0 + 0.05
    # print(occ_grid_tmp)

    return occ_grid_tmp


def add_pos_channels_np(occ_grid: np.ndarray) -> np.ndarray:
    occ_grid_tmp = np.zeros(
        (4, occ_grid.shape[0], occ_grid.shape[1], occ_grid.shape[2])
    )
    voxel_coords = np.indices(
        (occ_grid.shape[2], occ_grid.shape[1], occ_grid.shape[0])
    ).T
    # print(voxel_coords)

    occ_grid_tmp[0, :, :, :] = occ_grid
    occ_grid_tmp[1, :, :, :] = voxel_coords[:, :, :, 2] / 10.0 - 1.95
    occ_grid_tmp[2, :, :, :] = voxel_coords[:, :, :, 1] / 10.0 - 1.95
    occ_grid_tmp[3, :, :, :] = voxel_coords[:, :, :, 0] / 10.0 + 0.05
    # print(occ_grid_tmp)

    return occ_grid_tmp


def get_pc_from_voxel(occ_grid, desired_num_points=2048):
    indices = (occ_grid > 0).nonzero().contiguous()  # N x 4
    num_points = indices.shape[0]

    if num_points == 0:
        indices = (occ_grid > -1).nonzero().contiguous()  # N x 4
        num_points = indices.shape[0]
        choices = np.random.choice(num_points, desired_num_points, replace=True)
        indices = indices[choices]
    elif num_points < desired_num_points:
        choices = np.random.choice(
            num_points, desired_num_points - num_points, replace=True
        )
        indices = torch.cat((indices, indices[choices]))
    else:
        choices = np.random.choice(num_points, desired_num_points, replace=False)
        indices = indices[choices]

    # bs = occ_grid.shape[0]
    # num_points = occ_grid.shape[1] * occ_grid.shape[2] * occ_grid.shape[3]
    pc = torch.zeros((4, desired_num_points), device=occ_grid.device)
    # print(pc.shape)

    # voxel_coords = np.indices((occ_grid.shape[3], occ_grid.shape[2], occ_grid.shape[1])).T.reshape(-1, 3)
    # voxel_coords = torch.tensor(voxel_coords, device=occ_grid.device, dtype=torch.float).view(-1, 3)
    # voxel_coords[:, 0] = voxel_coords[:, 0] / 10 + 0.05
    # voxel_coords[:, 1] = voxel_coords[:, 1] / 10 - 1.95
    # voxel_coords[:, 2] = voxel_coords[:, 2] / 10 - 1.95

    # print(voxel_coords)
    # print(voxel_coords.shape)

    # pc[:, 3, :] = 1
    # pc[:, 0, :] = indices[:, 1] / 10 - 1.95 # x

    # print(pc)
    # print(voxel_feat)

    pc[0, :] = indices[:, 0] / 10 - 1.95
    pc[1, :] = indices[:, 1] / 10 - 1.95
    pc[2, :] = indices[:, 2] / 10 + 0.05
    if num_points != 0:
        pc[3, :] = 1.0
    else:
        pc[3, :] = 0.0

    return pc


"""
FkTorch
"""


class FkTorch:
    def __init__(self, device):
        self.d = device
        self.dtype = torch.float
        chain = pk.build_serial_chain_from_urdf(
            open(osp.join(CUR_DIR, "../../robot_model/fetch.urdf")).read(),
            "gripper_link",
        )
        self.chain = chain.to(dtype=self.dtype, device=self.d)

    def set_device(self, device):
        self.d = device
        self.chain.to(dtype=self.dtype, device=self.d)

    def get_link_positions(self, robot_state):
        # N = robot_state.shape[0]
        # th_batch = torch.rand(N, len(self.chain.get_joint_parameter_names()), dtype=self.dtype, device=self.d)
        robot_state = robot_state.to(self.d)

        # we want to forward the base offset but inverse the rotation. So we pass in the negative base offset
        bs = robot_state.shape[0]
        base_offset = torch.zeros((bs, 3), dtype=self.dtype, device=self.d)
        base_offset[:, :2] = robot_state[
            :, :2
        ]  # So we pass in the negative base offset
        transform3d = Transform3d(dtype=self.dtype, device=self.d)
        transform3d = transform3d.rotate_axis_angle(
            robot_state[:, 2], axis="Z", degrees=False
        ).translate(base_offset)
        # print(transform3d.get_matrix())

        # order of magnitudes faster when doing FK in parallel
        # elapsed 0.008678913116455078s for N=1000 when parallel
        # (N,4,4) transform matrix; only the one for the end effector is returned since end_only=True by default
        tg_batch = self.chain.forward_kinematics(robot_state[:, 3:], end_only=False)
        # print(tg_batch)

        torso_pos = tg_batch["torso_lift_link"].get_matrix()[:, :3, 3]
        shoulder_pan_pos = tg_batch["shoulder_pan_link"].get_matrix()[:, :3, 3]
        shoulder_lift_pos = tg_batch["shoulder_lift_link"].get_matrix()[:, :3, 3]
        upperarm_roll_pos = tg_batch["upperarm_roll_link"].get_matrix()[:, :3, 3]
        elbow_flex_pos = tg_batch["elbow_flex_link"].get_matrix()[:, :3, 3]
        forearm_roll_pos = tg_batch["forearm_roll_link"].get_matrix()[:, :3, 3]
        wrist_flex_pos = tg_batch["wrist_flex_link"].get_matrix()[:, :3, 3]
        wrist_roll_pos = tg_batch["wrist_roll_link"].get_matrix()[:, :3, 3]

        linkpos = torch.cat(
            (
                torso_pos,
                shoulder_pan_pos,
                shoulder_lift_pos,
                upperarm_roll_pos,
                elbow_flex_pos,
                forearm_roll_pos,
                wrist_flex_pos,
                wrist_roll_pos,
            ),
            dim=1,
        ).view(bs, -1, 3)
        # print(linkpos)

        linkpos = transform3d.transform_points(linkpos)
        linkpos = linkpos.view(bs, -1)
        # print(linkpos)
        return linkpos


global_fk = FkTorch("cpu")

"""
EBSA
"""


def get_ebsa_path(occ_grid, start_pos, target_pos):
    base_radius = 0.3
    map_resolution = 0.1
    base_expert_planner = HybridAStar(base_radius, TURNING_RADIUS, map_resolution)

    robot_height = 1.1
    occ_grid_resolution = 0.1
    occ_grid_2d = np.any(
        occ_grid[:, :, : int(robot_height / occ_grid_resolution)], axis=2
    ).astype(int)
    astar_start = (
        start_pos[0] / occ_grid_resolution,
        start_pos[1] / occ_grid_resolution,
        start_pos[2],
    )
    astar_goal = (
        target_pos[0] / occ_grid_resolution,
        target_pos[1] / occ_grid_resolution,
        target_pos[2],
    )
    astar_path, plan_time = base_expert_planner.plan(
        astar_start, astar_goal, occ_grid_2d, timeout=30
    )
    if astar_path is not None:
        expert_base_path = []
        for i in range(len(astar_path[0])):
            expert_base_path.append(
                [
                    astar_path[0][i] * occ_grid_resolution,
                    astar_path[1][i] * occ_grid_resolution,
                    astar_path[2][i],
                ]
            )

        num_waypoints = len(expert_base_path)
        node1_pos_arm = np.array(start_pos)[3:]
        node2_pos_arm = np.array(target_pos)[3:]
        arm_diff = node2_pos_arm - node1_pos_arm
        ebsa_path = []
        for i in range(num_waypoints):
            arm_step_size = arm_diff / (num_waypoints - 1)
            pos_arm = node1_pos_arm + arm_step_size * i
            ebsa_path.append(expert_base_path[i] + pos_arm.tolist())
    else:
        ebsa_path = []

    return ebsa_path
