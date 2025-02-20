import os
import sys
import os.path as osp
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))

# import matplotlib.pyplot as plt
import time
import json
import sys
import os.path as osp
import numpy as np
import rospy
from geometry_msgs.msg import Pose
from scipy.spatial.transform import Rotation as R
from nav_msgs.msg import Path

# import time
from rls.fetch_robot import FetchRobot

CUR_DIR = osp.dirname(osp.abspath(__file__))

OCC_GRID_RESOLUTION = 0.1
LOAD_PATH = False
REAL_ROBOT = True

CTRL_FREQ = 50.0
BASE_SPEED = 0.25
TURNING_RADIUS = 0.75
LOCAL_OCC_GRID_SIZE = 4
HOME_JOINTS = [0.2, 1.3205522228271485, 1.399532370159912, -0.19974325208511354, 1.719844644293213, 0.0004958728740930562, 1.4, 0]

if REAL_ROBOT:
    rospy.init_node("test_wbmp")
    fetch = FetchRobot(use_moveit=True)
# maze = Maze()

with open(os.path.join(CUR_DIR, "map/occ_grid_v4.npy"), 'rb') as f:
    occ_grid = np.load(f)

def get_local_occ_grid(occ_grid, state):
    base_x = state[0]
    base_y = state[1]

    # base_x = round(base_x, 1)
    # base_y = round(base_y, 1)

    small_occ_grid_resolution = 0.1
    small_occ_grid_size = int(LOCAL_OCC_GRID_SIZE / small_occ_grid_resolution) // 2
    idx_x = int(round(base_x / small_occ_grid_resolution))
    idx_y = int(round(base_y / small_occ_grid_resolution))

    min_y = max(0, idx_y - small_occ_grid_size)
    max_y = min(occ_grid.shape[1], idx_y + small_occ_grid_size)
    min_x = max(0, idx_x - small_occ_grid_size)
    max_x = min(occ_grid.shape[0], idx_x + small_occ_grid_size)

    min_y_1 = 0 if min_y != 0 else small_occ_grid_size - idx_y
    max_y_1 = 2 * small_occ_grid_size if max_y != occ_grid.shape[1] else occ_grid.shape[1] - idx_y + small_occ_grid_size
    min_x_1 = 0 if min_x != 0 else small_occ_grid_size - idx_x
    max_x_1 = 2 * small_occ_grid_size if max_x != occ_grid.shape[0] else occ_grid.shape[0] - idx_x + small_occ_grid_size

    # print(state, idx_x, min_x, max_x, min_x_1, max_x_1)
    # print(state, idx_y, min_y, max_y, min_y_1, max_y_1)

    local_occ_grid = np.ones((2*small_occ_grid_size, 2*small_occ_grid_size, occ_grid.shape[2]), dtype=np.uint8)
    local_occ_grid[min_x_1:max_x_1, min_y_1:max_y_1] = occ_grid[min_x:max_x, min_y:max_y]

    return local_occ_grid

base_x_bounds = [0, occ_grid.shape[0] * OCC_GRID_RESOLUTION]
base_y_bounds = [0, occ_grid.shape[1] * OCC_GRID_RESOLUTION]
print(base_x_bounds, base_y_bounds)

if not LOAD_PATH:
    # cur_joint_state = fetch.get_current_joint_state()
    # This the joint after tucking fetch
    # cur_joint_state = [0.2, 1.3205522228271485, 1.399532370159912, -0.19974325208511354, 1.719844644293213, 0.0004958728740930562, 1.4, 0]
    # maze.sample_start_goal(joint_state=cur_joint_state)

    # T0:
    # goal = [5.4, 1.05, -1.7043219922841046, 0.184, -0.8968170951812744, 0.38978915650634766, -1.3763065746292114, -2.1162578413635256, -3.0271986299713136, -1.2057725833190918, -2.035894721689453]

    # T1:
    # goal = [8.4, 3.6, 1.0, 0.38, 0.202, -0.681, -0.676, 0.928, 0.543, 1.503, 0.0]

    # T2:
    # goal = [8.5, 6.15, 1.628, 0.366, -1.5460745166748047, -0.7265653089782715, -1.097505491064453, -1.3891512701660156, 1.3726417249481202, 0.39647036330566404, 0.8065717518899536]

    # T3:
    # goal = [11.3, 2.1, -3.0, 0.31, 0.3974791992694855, -0.320060677935791, -0.3124906530841827, 0.3534510781616211, 0.31266097486419675, 1.5672810627685547, 2.361261040029297]

    # T4
    # goal = [3.7, 2.2, 1.0239268783749709, 0.12, -0.4822588335483551, -0.44277853529663086, -2.2594961336120605, -1.161355192437744, -2.381776409741211, 0, 1.7729792893502807]

    # New T2
    # goal = [7.5, 3, 1.8, 0.1, -0.2, 0.8, 0.0, 0.07503349088134766, 0.1, 0.4, 0.12164933240152359]
    goal = [9.6, 4.05, 1.4, 0.36, -0.1, -0.86, -1.7275880983337402, 0.85, 1.0301804965774537, 0.3, 0]

    cur_robot_state = fetch.get_current_state()
    print(cur_robot_state)

    # maze.robot.set_state(goal)
    # assert maze.pb_ompl_interface.is_state_valid(goal)

    # maze.sample_start_goal(goal_state=goal)
    # maze.start = cur_robot_state
    # res, path = maze.plan(allowed_time=10, interpolate=False)

    # cur_robot_state[2] = 0
    # goal_robot_state = goal = [cur_robot_state[0] + 2, cur_robot_state[1], cur_robot_state[2], 0.184, -0.8968170951812744, 0.38978915650634766, -1.3763065746292114, -2.1162578413635256, -3.0271986299713136, -1.2057725833190918, -2.035894721689453]
    # path = [cur_robot_state, goal_robot_state]
    # res = True
else:
    with open("test_path.json", "r") as f:
        path = json.load(f)
    res = True
    # start = path[0]
    # start_torso_state = start[3]
    # start_arm_state = start[4:]
    # fetch.arm_move_to_joint_state(start_arm_state)
    # fetch.set_height(start_torso_state)

res = []

local_occ_grid = get_local_occ_grid(occ_grid, cur_robot_state)
fetch.add_local_obstacle(local_occ_grid, cur_robot_state[2])
start_time = time.time()
plan_res = fetch.arm_move_to_joint(HOME_JOINTS, allowed_planning_time=10)
print(plan_res)
end_time = time.time()
print("arm1 takes {}".format(end_time - start_time))
if not plan_res:
    exit()
res.append(end_time - start_time)

goal_pose = Pose()
goal_pose.position.x = goal[0]
goal_pose.position.y = goal[1]
r = R.from_euler('xyz', [0, 0, goal[2]])
quat = r.as_quat()
goal_pose.orientation.x = quat[0]
goal_pose.orientation.y = quat[1]
goal_pose.orientation.z = quat[2]
goal_pose.orientation.w = quat[3]
fetch.move_base(goal_pose)

end_time = time.time()
print("base takes {}".format(end_time - start_time))
res.append(end_time - start_time)
cur_robot_state = fetch.get_current_state()
local_occ_grid = get_local_occ_grid(occ_grid, cur_robot_state)
fetch.add_local_obstacle(local_occ_grid, cur_robot_state[2])
fetch.arm_move_to_joint(goal[3:])
end_time = time.time()
print("arm2 takes {}".format(end_time - start_time))
res.append(end_time - start_time)

with open("res1.json", "w") as f:
    path = json.dump(res, f)


