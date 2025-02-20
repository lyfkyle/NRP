import os
import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../../'))

import numpy as np
import json
import matplotlib.pyplot as plt
import networkx as nx
import random
import math

from env.snake_8d.maze import Snake8DEnv


CUR_DIR = osp.dirname(osp.abspath(__file__))


def visualize_nodes_global(env, occ_g, curr_node_posns, start_pos=None, goal_pos=None, viz_edge=True, show=True, save=False, file_name=None):
    fig1 = plt.figure(figsize=(10, 10), dpi=250)
    img = plt.imread("tile.jpg")

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
        for i in range(1, len(curr_node_posns)):
            plt.plot((curr_node_posns[i - 1][0], curr_node_posns[i][0]), (curr_node_posns[i - 1][1], curr_node_posns[i][1]), ls="--", lw=s / 2, c='#66B2FF', zorder=1)

    if len(curr_node_posns) > 0:
        # plt.scatter(curr_node_posns[:,0], curr_node_posns[:,1], s = 50, color = 'green')
        for i, pos in enumerate(curr_node_posns):
            env.utils.visualize_robot(pos, ax=ax1, s=s, color='#66B2FF')
            # plt.text(pos[0], pos[1], str(i), color="black", fontsize=12)

    if start_pos is not None:
        env.utils.visualize_robot(start_pos, ax=ax1, s=s, color="#FFBF0F")
    if goal_pos is not None:
        env.utils.visualize_robot(goal_pos, ax=ax1, s=s, color="#FF9999")

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


def sample_problems(env, G, desired_start_pos=[2, 9], desired_goal_pos=[6, 0.5]):
    # path = dict(nx.all_pairs_shortest_path(G))
    free_nodes = [n for n in G.nodes() if not G.nodes[n]['col']]

    max_trial = 100
    i = 0
    while i < max_trial:
        while True:
            s_name = random.choice(free_nodes)
            start_pos = env.utils.node_to_numpy(G, s_name).tolist()

            if math.fabs(start_pos[0] - desired_start_pos[0]) < 0.5 and math.fabs(start_pos[1] - desired_start_pos[1]) < 0.5:
                break

        while True:
            g_name = random.choice(free_nodes)
            goal_pos = env.utils.node_to_numpy(G, g_name).tolist()

            if math.fabs(goal_pos[0] - desired_goal_pos[0]) < 0.5 and math.fabs(goal_pos[1] - desired_goal_pos[1]) < 0.5:
                break

        try:
            node_path = nx.shortest_path(G, source=s_name, target=g_name)
        except:
            continue

        path = [env.utils.node_to_numpy(G, n).tolist() for n in node_path]
        # for x in p:
        #     x[0] += 2
        #     x[1] += 2

        if len(path) > 4 or env.utils.calc_path_len_base(path) > 10:
            break

        i += 1

    return s_name, g_name, path


env = Snake8DEnv(gui=False)

data_dir = os.path.join(CUR_DIR, "../../env/snake_8d/dataset/gibson/mytest")
maze_dir = os.path.join(data_dir, "Ihlen")  # we choose this environment for visualization
output_dir = os.path.join(CUR_DIR, "viz_res")
print("generating test problem from {}".format(maze_dir))

occ_grid = env.utils.get_occ_grid(maze_dir)
G = env.utils.get_prm(maze_dir)
mesh_path = env.utils.get_mesh_path(maze_dir)

env.clear_obstacles()
env.load_mesh(osp.join(maze_dir, "env_small.obj"))
env.load_occupancy_grid(occ_grid)

# --------------- Stage 1
# s_node, g_node, expert_path = sample_problems(G, desired_start_pos=[3, 9], desired_goal_pos=[8.2, 8])
# start_pos = utils.node_to_numpy(G, s_node).tolist()
# goal_pos = utils.node_to_numpy(G, g_node).tolist()

# path_viz = utils.interpolate(expert_path, 1)
# utils.visualize_nodes_global(occ_grid, path_viz, np.array(start_pos), np.array(goal_pos), show=False, save=True, file_name=osp.join(output_dir, "viz.png"))

start_goal = [[5.2, 5.6, 0, 0.20754372628238282, 0.20754372628238282, -0.298117646607281, 1.5, 0.5], [8.2, 5, -3, 0.14471035176586566, -1.509814987502104, 0.3749476471727431, -1.354420777838151, -2.01864794643815]]
start_pos = start_goal[0]
goal_pos = start_goal[1]
visualize_nodes_global(env, occ_grid, [], np.array(start_pos), np.array(goal_pos), show=False, save=True, file_name=osp.join(output_dir, "viz.png"))

# with open(os.path.join(output_dir, "final_path.json"), "w") as f:
#     json.dump(expert_path, f)

###########
# with open(os.path.join(output_dir, "viz_path.json"), "r") as f:
#     expert_path = json.load(f)

# expert_path = utils.interpolate(expert_path, 1)

# start_pos = expert_path[0]
# goal_pos = expert_path[-1]

# print(start_pos, expert_path[9])

# start_pos[2] = -1.5
# start_pos[3] = 0.01
# start_pos[4] = 0.01
# start_pos[5] = -0.86
# start_pos[6] = 0.221
# start_pos[7] = -0.089
# goal_pos[5] = 0.02

# q2 = [3.23, 8.62, -1.47, 1.96, -1.64, -0.37, 0.077, -0.0323]

# print(expert_path[14])
# expert_path[14][0] = 5

# final_path = [start_pos, q2, expert_path[9], expert_path[10], expert_path[14], expert_path[15], expert_path[20], expert_path[23], expert_path[25], goal_pos]
# with open(os.path.join(output_dir, "final_path.json"), "w") as f:
#     json.dump(final_path, f)

# ---------------- Stage 2
# with open(os.path.join(output_dir, "final_path.json"), "r") as f:
#     final_path = json.load(f)

# start_pos = final_path[0]
# goal_pos = final_path[-1]

# final_path = utils.interpolate_base(final_path, 1)
# final_path = final_path[:4] + final_path[5:]

# visualize_nodes_global(occ_grid, final_path, np.array(start_pos), np.array(goal_pos), show=False, save=True, file_name=osp.join(output_dir, "viz2.png"))

# ---------------- Stage 3
# with open(os.path.join(output_dir, "final_path.json"), "r") as f:
#     final_path = json.load(f)

# start_pos = final_path[0]
# goal_pos = final_path[-1]

# final_path = env.utils.interpolate_base(final_path, 1)
# final_path = final_path[:4] + final_path[5:]
# viz_path = env.utils.interpolate_base(final_path[3:6], 0.5)

# # viz_path = [final_path[4], final_path[5], final_path[6], final_path[8], final_path[10], final_path[12]]
# visualize_nodes_global(occ_grid, viz_path, np.array(start_pos), np.array(goal_pos), viz_edge=True, show=False, save=True, file_name=osp.join(output_dir, "viz3.png"))

# idx = [4, 5, 6, 7, 8, 9]
# idx = [5]

# for i in idx:
#     viz_path = [final_path[i]]
#     if i == 5:
#         viz_path[0][1] -= 0.1
#     visualize_nodes_global(occ_grid, viz_path, np.array(start_pos), np.array(goal_pos), viz_edge=False, show=False, save=True, file_name=osp.join(output_dir, f"viz4_{i}.png"))

# goal_pos = np.array(goal_pos)
# goal_pos[:2] = 5
# goal_pos[2:] = 1
# goal_pos[5:7] = -1
# viz_path = [goal_pos]
# visualize_nodes_global(occ_grid, viz_path, show=False, save=True, file_name=osp.join(output_dir, "viz4.png"))
