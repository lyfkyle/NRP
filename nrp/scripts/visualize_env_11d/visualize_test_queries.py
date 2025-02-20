import os
import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../../'))

import reeds_shepp as rs
from scipy.spatial.transform import Rotation as R
from PIL import Image
import math
import random
import networkx as nx
import matplotlib.pyplot as plt
import json
import numpy as np


from env.fetch_11d.maze import Fetch11DEnv

CUR_DIR = osp.dirname(osp.abspath(__file__))


def get_global_img(maze):
    env_len, env_width = maze.size[0], maze.size[1]
    print(env_len, env_width)
    view_matrix = maze.p.computeViewMatrix([env_len / 2, 0, 5], [env_len / 2, env_width / 2, 0], [0, 0, 1])

    # fov = math.degrees(math.atan2(env_width / 2, 48) * 2)
    fov = 120
    proj_matrix = maze.p.computeProjectionMatrixFOV(fov, 0.8, 0.01, 52)
    # proj_matrix = maze.p.computeProjectionMatrix(0, maze.size[0], 0, maze.size[1], 0.01, 12)

    w, h, rgba, depth, mask = maze.p.getCameraImage(
        width=1280,
        height=1024,
        projectionMatrix=proj_matrix,
        viewMatrix=view_matrix,
        flags=maze.p.ER_NO_SEGMENTATION_MASK,
        renderer=maze.p.ER_BULLET_HARDWARE_OPENGL
    )
    return rgba


def visualize_nodes_global(
    env,
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
    if mesh is not None:
        col_id = env.p.createCollisionShape(env.p.GEOM_MESH, fileName=mesh, flags=env.p.GEOM_FORCE_CONCAVE_TRIMESH, meshScale=[1.2, 1.2, 1.2])
        visualBoxId = env.p.createVisualShape(env.p.GEOM_MESH, fileName=mesh, meshScale=[1.2, 1.2, 1])
        # env.p.createMultiBody(baseCollisionShapeIndex=col_id, baseVisualShapeIndex=visualBoxId, basePosition=[8.5, 3, 0.06])
        env.p.createMultiBody(baseCollisionShapeIndex=col_id, baseVisualShapeIndex=visualBoxId, basePosition=[3.5, 13, 0.06], baseOrientation=[0, 0, -0.7071068, 0.7071068])
    if occ_g is not None:
        if mesh is None:
            env.load_occupancy_grid(occ_g, add_box=True)
        else:
            env.load_occupancy_grid(occ_g)

    if len(curr_node_posns) > 0:
        for i, pos in enumerate(curr_node_posns):
            if sample_pos is not None and np.allclose(np.array(pos), np.array(sample_pos)):
                continue
            if start_pos is not None and np.allclose(np.array(pos), np.array(start_pos)):
                continue
            if goal_pos is not None and np.allclose(np.array(pos), np.array(goal_pos)):
                continue

            env.add_robot(pos, rgba=[1, 1, 1, 0.5])

    if viz_edge:
        if edge_path is None:
            edge_path = curr_node_posns

        for i in range(1, len(edge_path)):
            env.add_line(edge_path[i - 1][:2], edge_path[i][:2], colour=[255 / 255, 255 / 255, 255 / 255, 1])

    if start_pos is not None:
        start_pos_tmp = start_pos.copy()
        env.add_robot(start_pos_tmp, rgba=[1, 0.749, 0.0588, 1])  # yellow
    if goal_pos is not None:
        goal_pos_tmp = goal_pos.copy()
        env.add_robot(goal_pos_tmp, rgba=[1, 0.6, 0.6, 1])  # red
    if sample_pos is not None:
        sample_pos_tmp = sample_pos.copy()
        env.add_robot(sample_pos_tmp, rgba=[0, 1, 0, 1])  # green

    # if start_pos is not None:
    #     env.add_robot(start_pos) # yellow
    # if goal_pos is not None:
    #     env.add_robot(goal_pos) # red

    # img = maze.get_global_img()
    # print(img.shape)
    # pil_img = Image.fromarray(img)
    # rgb_img = pil_img.convert("RGB")
    # print(rgb_img.mode)
    # if show:
    #     rgb_img.show()
    # if save:
    #     rgb_img.save(file_name)
    input()


def visualize_nodes_global_2(mesh, occ_g, curr_node_posns, start_pos, goal_pos, selected_idx=[]):
    maze = Fetch11DEnv(gui=True, add_robot=False)
    if mesh is not None:
        col_id = maze.p.createCollisionShape(maze.p.GEOM_MESH, fileName=mesh, flags=maze.p.GEOM_FORCE_CONCAVE_TRIMESH)
        visualBoxId = maze.p.createVisualShape(maze.p.GEOM_MESH, fileName=mesh)
        maze.p.createMultiBody(baseCollisionShapeIndex=col_id, baseVisualShapeIndex=visualBoxId, basePosition=[8.5, 3, 0.06])
    if occ_g is not None:
        if mesh is None:
            maze.load_occupancy_grid(occ_g, add_box=True)
        else:
            maze.load_occupancy_grid(occ_g)

    if len(curr_node_posns) > 0:
        for i, state in enumerate(curr_node_posns):
            # if i not in selected_idx:
            #     # robot_id = maze.p.loadURDF(os.path.join(CUR_DIR, "../../robot_model/fetch_base.urdf"), (-1, -1, 0), flags=maze.p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
            #     # pos = [state[0], state[1], 0]
            #     # r = R.from_euler('z', state[2]) # unit is radians
            #     # quat = r.as_quat()
            #     # maze.p.resetBasePositionAndOrientation(robot_id, pos, quat)
            #     # num_links = maze.p.getNumJoints(robot_id)
            #     # for link in range(-1, num_links): # Changes the color of the robot
            #     #     maze.p.changeVisualShape(robot_id, link, rgbaColor=[1, 1, 1, 0.2])
            #     maze.add_robot(state, rgba=[1, 1, 1, 0.2])
            # else:
            #     maze.add_robot(state)
            maze.add_robot(state)
    # if start_pos is not None:
    #     start_pos_tmp = start_pos.copy()
    #     maze.add_robot(start_pos_tmp, rgba=[1, 0.749, 0.0588, 1]) # yellow
    # if goal_pos is not None:
    #     goal_pos_tmp = goal_pos.copy()
    #     maze.add_robot(goal_pos_tmp, rgba=[1, 0.6, 0.6, 1]) # red

    input()


def sample_problems(env, G, desired_start_pos=[2, 9], desired_goal_pos=[6, 0.5]):
    # path = dict(nx.all_pairs_shortest_path(G))
    free_nodes = [n for n in G.nodes() if not G.nodes[n]['col']]
    random.shuffle(free_nodes)

    found = False
    for s_name in free_nodes:
        start_pos = env.utils.node_to_numpy(G, s_name).tolist()
        if math.fabs(start_pos[0] - desired_start_pos[0]) < 0.5 and math.fabs(start_pos[1] - desired_start_pos[1]) < 0.5 and math.fabs(start_pos[2] - desired_start_pos[2]) < 1:
            found = True
            break

    if not found:
        return None

    found = False
    for g_name in free_nodes:
        goal_pos = env.utils.node_to_numpy(G, g_name).tolist()

        if math.fabs(goal_pos[0] - desired_goal_pos[0]) < 0.5 and math.fabs(goal_pos[1] - desired_goal_pos[1]) < 0.5 and math.fabs(goal_pos[2] - desired_goal_pos[2]) < 1:
            found = True
            break

    if not found:
        return None

    try:
        node_path = nx.shortest_path(G, source=s_name, target=g_name)
    except:
        pass

    path = [env.utils.node_to_numpy(G, n).tolist() for n in node_path]
    # for x in p:
    #     x[0] += 2
    #     x[1] += 2

    return s_name, g_name, path


def solve_problem(env, G, start_pos, goal_pos):
    # path = dict(nx.all_pairs_shortest_path(G))
    free_nodes = [n for n in G.nodes() if not G.nodes[n]['col']]

    for node in free_nodes:
        if np.allclose(env.utils.node_to_numpy(G, node), start_pos):
            s_node = node
            break

    for node in free_nodes:
        if np.allclose(env.utils.node_to_numpy(G, node), goal_pos):
            g_node = node
            break

    node_path = nx.shortest_path(G, source=s_node, target=g_node)
    path = [env.utils.node_to_numpy(G, n).tolist() for n in node_path]

    return path


def interpolate_base_rs(env, path, step_size=0.2, num_points_per_edge=None, turning_radius=0.75):
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
        base_traj = env.utils.rs_path_sample(node1_pos_base, node2_pos_base, turning_radius, base_step_size)

        arm_diff = node2_pos_arm - node1_pos_arm
        arm_step_size = arm_diff / (num_points_per_edge - 1)

        nodepos = np.zeros(len(node1_pos))
        for j in range(num_points_per_edge):
            nodepos[:3] = base_traj[j][:3]
            nodepos[3:] = node1_pos_arm + arm_step_size * j
            new_path.append(nodepos.tolist())

    new_path.append(path[-1])

    return new_path


env = Fetch11DEnv(gui=True)

data_dir = os.path.join(CUR_DIR, "../../env/fetch_11d/dataset/gibson/mytest")
# maze_dir = os.path.join(data_dir, "Connellsville")  # we choose this environment for visualization
maze_dir = os.path.join(data_dir, "Collierville")  # we choose this environment for visualization
# maze_dir = os.path.join(data_dir, "Ihlen")  # we choose this environment for visualization
output_dir = os.path.join(CUR_DIR, "viz_res")
print("generating test problem from {}".format(maze_dir))

occ_grid = env.utils.get_occ_grid(maze_dir)
G = env.utils.get_prm(maze_dir)
# mesh_path = env.utils.get_mesh_path(maze_dir)
mesh_path = os.path.join(maze_dir, "mesh_z_up.obj")

#  -------------- Stage 1
# maze.clear_obstacles()
# maze.load_mesh(mesh_path)
# input()

# maze.load_occupancy_grid(occ_grid)

visualize_nodes_global(env, mesh_path, occ_grid, [[7.5, 10, math.radians(90), 0, 0, 0, 0, 0, 0, 0, 0]], None, None, show=False, save=True, file_name=osp.join(output_dir, "viz.png"))
# input()

#  -------------- Stage 2
# start_goal = [[7.5, 6.5, 1.5708, 0.20754372628238282, 1.2, 0.298117646607281, 2.886076069386772, 0.5, 2.546488001092416, -0.2, -1.3258486452394675], [8.45832867057521, 10.280478674031647, 0.9711561164054712, 0.19064093560129133, -1.3789884255985303, 0.4500719015886152, -1.3792289778060858, -1.919265149979747, 2.5413427974725398, 1.8423831538214899, 0.5636244272867166]]
# start_goal = [[7.5, 10, 0, 0.1, 0.2, 0.298117646607281, 2.886076069386772, 0.5, 2.546488001092416, -0.2, -1.3258486452394675], [6.473157637271126, 13.524433485808451, 1.935917699316838, -1.5596249708924108, 0.14471035176586566, -1.509814987502104, 0.3749476471727431, -1.354420777838151, -2.01864794643815, 2.769560296629825, 1.599798291157819, 0.5251947053053616]]
# start_goal = [[6.473157637271126, 13.524433485808451, 1.935917699316838, 0.15652237240405248, 0.6373954314770698, 0.11033399912074215, -0.9432605390623823, 0.5621819885062247, -1.082413090768589, -0.5023517159358257, 0.27420777767522253], [7.766104417377146, 7.973047471072846, 1.6903024667081201, 0.2504105091094345, -0.6250377245447701, 0.12055823421034195, -2.1190553530929908, -1.2712066762140357, 2.9672829703782746, -0.08529559022930222, 1.5559487448598492]]
# start_pos = start_goal[0]
# goal_pos = start_goal[-1]
# visualize_nodes_global(env, mesh_path, occ_grid, [], np.array(start_pos), np.array(goal_pos), show=False, save=True, file_name=osp.join(output_dir, "viz.png"))

# expert_path = solve_problem(G, start_goal[0], start_goal[1])
# start_pos = start_goal[0]
# goal_pos = start_goal[1]
# s_node, g_node, expert_path = sample_problems(env, G, desired_start_pos=[6, 6.5, math.radians(0)], desired_goal_pos=[6.473157637271126, 13.524433485808451, 1.935917699316838])
# start_pos = env.utils.node_to_numpy(G, s_node).tolist()
# goal_pos = env.utils.node_to_numpy(G, g_node).tolist()

# path_viz = env.utils.interpolate(expert_path, 1)

# # # path_viz = path_viz[7: 12]
# # # path_viz = utils.interpolate(path_viz, 0.3)

# # print(start_pos, goal_pos)
# visualize_nodes_global(env, mesh_path, occ_grid, path_viz[:-3], np.array(start_pos), np.array(goal_pos), show=False, save=True, file_name=osp.join(output_dir, "viz.png"))

# with open(os.path.join(output_dir, "viz_path.json"), "w") as f:
#     json.dump(expert_path, f)

#  -------------- Stage 3

# # with open(os.path.join(output_dir, "viz_path.json"), "r") as f:
# #     expert_path = json.load(f)
# path_intended = [[7.5, 6.5, 1.5708, 0.20754372628238282, 1.2, 0.298117646607281, 2.886076069386772, 0.5, 2.546488001092416, -0.2, -1.3258486452394675], [7.939160108566284, 7.0399041175842285, -1.9484614133834839, 0.23101797699928284, 0.3615856170654297, -0.6643131375312805, 2.4782745838165283, 1.3047494888305664, 0.5106432437896729, -1.5895323753356934, -0.0917356014251709], [8.45832867057521, 10.280478674031647, 0.9711561164054712, 0.19064093560129133, -1.3789884255985303, 0.4500719015886152, -1.3792289778060858, -1.919265149979747, 2.5413427974725398, 1.8423831538214899, 0.5636244272867166]]

# path_intended_interpolated = env.utils.interpolate_base(path_intended, step_size=1.0)

# path_actual = [[7.5, 6.5, 1.5708, 0.20754372628238282, 1.2, 0.298117646607281, 2.886076069386772, 0.5, 2.546488001092416, -0.2, -1.3258486452394675], [7.177587270736694, 5.9458537101745605, 1.346405029296875, 0.19292482733726501, 0.41383224725723267, -0.03408195450901985, 0.1921413093805313, 1.1110771894454956, 2.203540325164795, -1.5210882425308228, -0.8479540944099426], [8.632794269756985, 10.412608698374317, 0.900408473975865, 0.06097080517106766, -1.154877208313908, 0.535154276188283, -1.1554103233833146, -2.091730900875924, 2.7688194920911235, 1.804283662834279, 0.3039934617922786]]

# path_actual_viz = env.utils.interpolate(path_actual[:2]) + env.utils.interpolate_base(path_actual[1:], step_size=1.0)[:4]
# path_actual_viz = env.utils.interpolate(path_actual[:2], step_size=0.5)
# start_pos = path_actual[0]
# goal_pos = path_actual[-1]

# expert_path = interpolate_base(expert_path, 1.5)
# expert_path = interpolate_base_rs(expert_path[1:6], 0.3)

# tmp = utils.interpolate(expert_path[4:6], 0.1)

# # expert_path[-2][1] += 0.2

# viz_path = expert_path[2:4] + tmp + [expert_path[6], expert_path[9], expert_path[13], expert_path[15]]

# viz_mesh = os.path.join(maze_dir, "mesh_cut_2.obj")
# visualize_nodes_global(env, mesh_path, occ_grid, path_actual_viz, np.array(start_pos), np.array(goal_pos), sample_pos=path_actual[1])


# sl_path_intended = env.utils.interpolate([start_pos, goal_pos], step_size=0.2)[:4]
# sl_path_actual = env.utils.rrt_extend_intermediate(env, start_pos, goal_pos, step_size=0.2)
# sl_path_actual.append(sl_path_intended[len(sl_path_actual)])
# visualize_nodes_global(env, mesh_path, occ_grid, sl_path_intended, start_pos, goal_pos)


# --------------- Stage 4

# with open(os.path.join(output_dir, "viz_path.json"), "r") as f:
#     expert_path = json.load(f)

# expert_path = utils.interpolate(expert_path)
# start_pos = expert_path[0]
# goal_pos = expert_path[-1]

# # new_path = expert_path[0:1] + expert_path[3:14] + expert_path[15:]
# new_path = [expert_path[0], expert_path[24], expert_path[31], expert_path[35]]

# viz_mesh = os.path.join(maze_dir, "mesh_z_up.obj")
# print(len(expert_path))
# # selected_idx = [3, 8, 30]
# # selected_idx = [3, 8, 11, 12, 13, 15, 27]
# selected_idx = [9, 10, 11, 12]
# visualize_nodes_global_2(viz_mesh, occ_grid, new_path, np.array(start_pos), np.array(goal_pos), selected_idx)

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

# with open(os.path.join(output_dir, "final_path.json"), "r") as f:
#     final_path = json.load(f)

# start_pos = final_path[0]
# goal_pos = final_path[-1]

path = [[5, 5, 0, 0.2, 1.0, 0.2, 0.2, 1.0, 1.0, -1.6, 0]]


visualize_nodes_global(None, None, path, None, None, show=False, save=True, file_name=osp.join(output_dir, "viz3.png"))
