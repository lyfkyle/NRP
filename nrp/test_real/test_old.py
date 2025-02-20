import os
import sys
import os.path as osp
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))

import pybullet as p
import pybullet_data
import math
import random
import json
import sys
import os.path as osp
import pybullet_utils.bullet_client as bc
import time
import datetime
import numpy as np
import copy
import rospy
import reeds_shepp as rs
# import moveit_commander
# import moveit_msgs.msg
# import geometry_msgs.msg


# import time
from rls.fetch_robot import FetchRobot
from rls.maze import Maze
import utils

CUR_DIR = osp.dirname(osp.abspath(__file__))

OCC_GRID_RESOLUTION = 0.1
LOAD_PATH = False
REAL_ROBOT = True

CTRL_FREQ = 50.0
BASE_SPEED = 0.25
TURNING_RADIUS = 0.5

def calc_path_len_arm_rs(path):
    v1 = np.array(path[0][3:])
    v2 = np.array(path[1][3:])
    return np.linalg.norm(v2 - v1)

def interpolate_base(cur_q, target_q, base_step_size):
    # res_base_waypoints = [path[0]]
    v1 = cur_q[:3]
    v2 = target_q[:3]

    base_traj = rs.path_sample(v1, v2, 1.0, base_step_size)
    # base_traj = [list(t)[:3] for t in base_traj]
    res_base_waypoints = []
    res_base_cmds = []
    for t in base_traj:
        t = list(t)
        t[2] = utils.enforce_pi(t[2])
        res_base_waypoints.append(t[:3])
        res_base_cmds.append(get_desired_base_vel(t[3], t[4]))

    # if len(base_waypoints) > 1:
    #     res_base_waypoints += base_waypoints[1:]

    if not np.allclose(np.array(res_base_waypoints[-1]), np.array(v2)):
        res_base_waypoints.append(v2)
        res_base_cmds.append([0, 0])

    return res_base_waypoints, res_base_cmds


def interpolate_arm(cur_q, target_q, num_steps):
    res_arm_waypoints = [path[0]]

    v1 = np.array(cur_q[3:])
    v2 = np.array(target_q[3:])

    arm_diff = v2 - v1
    actual_step_size = arm_diff / num_steps

    for i in range(num_steps):
        nodepos = v1 + actual_step_size * (i + 1)
        res_arm_waypoints.append(nodepos.tolist())

    return res_arm_waypoints


def get_desired_base_vel(rs_type, rs_len):
    if math.fabs(rs_len) < 1e-4:
        return [0, 0]

    forward = 1 if rs_len > 0 else -1
    linear_speed = BASE_SPEED * forward
    angular_speed = BASE_SPEED / TURNING_RADIUS
    rs_type = int(rs_type)
    if rs_type == 1:
        base_cmd = [linear_speed, angular_speed * forward]
    elif rs_type == 0:
        base_cmd = [linear_speed, 0]
    elif rs_type == -1:
        base_cmd = [linear_speed, angular_speed * forward * -1]

    return base_cmd


if REAL_ROBOT:
    rospy.init_node("test_wbmp")
    fetch = FetchRobot()
maze = Maze()

with open(os.path.join(CUR_DIR, "map/occ_grid_v2.npy"), 'rb') as f:
    occ_grid = np.load(f)

base_x_bounds = [0, occ_grid.shape[0] * OCC_GRID_RESOLUTION]
base_y_bounds = [0, occ_grid.shape[1] * OCC_GRID_RESOLUTION]
print(base_x_bounds, base_y_bounds)
maze.robot.set_base_bounds(base_x_bounds, base_y_bounds)
maze.robot_base.set_base_bounds(base_x_bounds, base_y_bounds)

maze.load_mesh(osp.join(CUR_DIR, "map/rls_v2.obj"))
maze.load_occupancy_grid(occ_grid)

low = maze.robot.get_joint_lower_bounds()
high = maze.robot.get_joint_higher_bounds()

# Generate env
# for i in range(10):
#     path = []
#     while len(path) <= 2:
#         maze.sample_start_goal()
#         res, path = maze.plan(allowed_time=20, interpolate=False)

#     with open("test_path_{}.json".format(i), "w") as f:
#         json.dump(path, f)

# # Test env
# avg_time = 0
# success_cnt = 0
# path_len = []
# expert_path_len = []
# for i in range(10):
#     with open("test_env/test_path_{}.json".format(i), "r") as f:
#         path = json.load(f)
#     maze.start = path[0]
#     maze.goal = path[-1]
#     start_time = time.time()
#     res, path = maze.plan(allowed_time=10, interpolate=False)
#     end_time = time.time()
#     avg_time += end_time - start_time
#     if res:
#         success_cnt += 1
# avg_time /= 10
# print(avg_time, success_cnt)

# 4.752179431915283 8
# 2.4317355394363402 9

if not LOAD_PATH:
    # cur_joint_state = fetch.get_current_joint_state()
    # This the joint after tucking fetch
    # cur_joint_state = [0.2, 1.3205522228271485, 1.399532370159912, -0.19974325208511354, 1.719844644293213, 0.0004958728740930562, 1.4, 0]
    # maze.sample_start_goal(joint_state=cur_joint_state)

    # T1:
    # goal = [5.2, 1.05, -1.7043219922841046, 0.184, -0.8968170951812744, 0.38978915650634766, -1.3763065746292114, -2.1162578413635256, -3.0271986299713136, -1.2057725833190918, -2.035894721689453]

    # T2:
    # goal = [8.4, 3.6, 1.0, 0.38, 0.202, -0.681, -0.676, 0.928, 0.543, 1.503, 0.0]

    # T3:
    # goal = [8.5, 6.15, 1.628, 0.366, -1.5460745166748047, -0.7265653089782715, -1.097505491064453, -1.3891512701660156, 1.3726417249481202, 0.39647036330566404, 0.8065717518899536]

    # T4:
    # goal = [11.3, 2.1, -3.0, 0.31, 0.3974791992694855, -0.320060677935791, -0.3124906530841827, 0.3534510781616211, 0.31266097486419675, 1.5672810627685547, 2.361261040029297]

    # T5:
    goal = [3.7, 2.2, 1.0239268783749709, 0.12, -0.4822588335483551, -0.44277853529663086, -2.2594961336120605, -1.161355192437744, -2.381776409741211, 0, 1.7729792893502807]

    cur_robot_state = fetch.get_current_state()
    print(cur_robot_state)
    maze.robot.set_state(cur_robot_state)
    maze.robot.set_state(goal)

    assert maze.pb_ompl_interface.is_state_valid(goal)

    maze.sample_start_goal(goal_state=goal)
    maze.start = cur_robot_state
    res, path = maze.plan(allowed_time=10, interpolate=False)
else:
    with open("test_path.json", "r") as f:
        path = json.load(f)
    res = True
    # start = path[0]
    # start_torso_state = start[3]
    # start_arm_state = start[4:]
    # fetch.arm_move_to_joint_state(start_arm_state)
    # fetch.set_height(start_torso_state)

if res:
    with open("test_path.json", "w") as f:
        json.dump(path, f)
    print(path)

    interpolated_path = utils.interpolate(path)
    with open("test_path_interpolated.json", "w") as f:
        json.dump(interpolated_path, f)
    # new_path = []
    # for i in range(1, len(path)):
    #     tmp_path = utils.rrt_extend_intermediate(maze, path[i - 1], path[i])
    #     new_path += tmp_path

    print("sanity check..")
    # new_path = utils.interpolate(path)
    for point in path[1:]:
        assert maze.pb_ompl_interface.is_state_valid(point)
    for point in interpolated_path[1:]:
        assert maze.pb_ompl_interface.is_state_valid(point)
    # if REAL_ROBOT:
    #     assert fetch.check_traj_col_free(point)

    input("press anything to visualize in simulator")

    maze.execute(interpolated_path)
    maze.robot.set_state(path[0])

    # key = input("press y to execute on real robot")
    # if key != "y":
    #     exit()

    # total_path_len = utils.get_path_len(path)
    for i in range(1, len(path)):
        cur_q = path[i - 1]
        target_q = path[i]

        cur_path = [cur_q, target_q]
        total_base_path_len = utils.calc_path_len_base(cur_path)
        total_arm_path_len = calc_path_len_arm_rs(cur_path)
        total_time = total_base_path_len / BASE_SPEED
        step_time = 1 / CTRL_FREQ
        num_steps = round(total_time / step_time)
        base_step_size = total_base_path_len / num_steps
        arm_step_size = total_arm_path_len / num_steps

        base_waypoints, desired_base_vel = interpolate_base(cur_q, target_q, base_step_size)
        arm_waypoints = interpolate_arm(cur_q, target_q, num_steps)
        print(len(base_waypoints), len(desired_base_vel), len(arm_waypoints), num_steps)
        print(step_time, num_steps, total_time)

        fetch.move_wb_ctrl(arm_waypoints, step_time, base_waypoints, desired_base_vel)


