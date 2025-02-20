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
from geometry_msgs.msg import Twist


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
    res_arm_waypoints = [cur_q]

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

with open(os.path.join(CUR_DIR, "map/occ_grid.npy"), 'rb') as f:
    occ_grid = np.load(f)

base_x_bounds = [0, occ_grid.shape[0] * OCC_GRID_RESOLUTION]
base_y_bounds = [0, occ_grid.shape[1] * OCC_GRID_RESOLUTION]
print(base_x_bounds, base_y_bounds)
maze.robot.set_base_bounds(base_x_bounds, base_y_bounds)
maze.robot_base.set_base_bounds(base_x_bounds, base_y_bounds)

maze.load_mesh(osp.join(CUR_DIR, "map/rls_fixed.obj"))
maze.load_occupancy_grid(occ_grid)

low = maze.robot.get_joint_lower_bounds()
high = maze.robot.get_joint_higher_bounds()

rate = rospy.Rate(100)
start_time = time.time()
end_time = time.time()
cur_states = []
twist_msg = Twist()
twist_msg.linear.x = -0.25
twist_msg.angular.z = -0.5
time_passed = end_time - start_time
while time_passed < 8:
    if time_passed < 0.5:
        twist_msg.linear.x = 0.5 * time_passed
        twist_msg.angular.z = time_passed
    if time_passed < 2:
        twist_msg.linear.x = 0.25
        twist_msg.angular.z = 0.5
    elif time_passed < 4:
        twist_msg.linear.x = 0.25
        twist_msg.angular.z = 0
    elif time_passed < 6:
        twist_msg.linear.x = 0.25
        twist_msg.angular.z = -0.5
    elif time_passed < 8:
        twist_msg.linear.x = -0.25
        twist_msg.angular.z = 0.5

    fetch._vel_pub.publish(twist_msg)
    cur_states.append(fetch.get_current_base_state())

    time_passed = time.time() - start_time
    rate.sleep()

with open("res2.json", "w") as f:
    json.dump(cur_states, f)