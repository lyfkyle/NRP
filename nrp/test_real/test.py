import os
import sys
import os.path as osp
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../../'))

import matplotlib.pyplot as plt
import math
import json
import sys
import os.path as osp
import pybullet_utils.bullet_client as bc
import numpy as np
import copy
import rospy
import reeds_shepp as rs
# import moveit_commander
# import moveit_msgs.msg
# import geometry_msgs.msg


# import time
from fetch_robot import FetchRobot
from nrp.env.rls.rls_env import RLSEnv
from nrp.env.rls import utils

CUR_DIR = osp.dirname(osp.abspath(__file__))

OCC_GRID_RESOLUTION = 0.1
LOAD_PATH = False
REAL_ROBOT = True
REVERSE = True

CTRL_FREQ = 50.0
BASE_SPEED = 0.25
TURNING_RADIUS = 0.75

step_time = 1 / CTRL_FREQ
acceleration_time = 0.5
num_acceleration_steps = int(acceleration_time / step_time)
BASE_STEP_SIZE = BASE_SPEED / (acceleration_time / (step_time ** 2))

def get_path_type(x):
    if x > 0:
        return 1
    elif x == 0:
        return 0
    else:
        return -1

def calc_path_len_arm_rs(path):
    v1 = np.array(path[0][3:])
    v2 = np.array(path[1][3:])
    return np.linalg.norm(v2 - v1)

def parameterize_base_path(base_path):
    res_base_waypoints = []
    res_base_cmds = []
    res_idx = []
    n_waypoints = len(base_path)
    idx = 0

    # path_increment = 1

    # Calculate accelerate idx. The index from where acceleration stops. At acceleration index, the speed is maximum.
    total_acceleration_path_decrements = int((1 + num_acceleration_steps) * num_acceleration_steps / 2)
    accelerate_idx = total_acceleration_path_decrements
    # decelerate_idx = max(decelerate_idx, int(n_waypoints / 2))

    # Calculate decelerate_idx. The index from where deceleration happens. At deceleration index, the speed is maximum.
    decelerate_idx = n_waypoints - total_acceleration_path_decrements - 1
    # decelerate_idx = max(decelerate_idx, int(n_waypoints / 2))

    if decelerate_idx < accelerate_idx:
        decelerate_idx = int(n_waypoints / 2)
        accelerate_idx = int(n_waypoints / 2)

    cur_path_increment = 1
    while idx < n_waypoints:
        t = list(base_path[idx])
        t[2] = utils.enforce_pi(t[2])
        res_base_waypoints.append(t[:3])
        res_idx.append(idx)

        scale = min(cur_path_increment / num_acceleration_steps, 1.0)
        res_base_cmds.append(get_desired_base_vel(t[3], t[4], scale))

        if idx >= decelerate_idx:
            next_path_increment = cur_path_increment - 1 # decrease NEXT path_increment
        elif idx < accelerate_idx:
            next_path_increment = cur_path_increment + 1  # increase NEXT path_increment
            next_path_increment = min(next_path_increment, num_acceleration_steps)

        idx += cur_path_increment
        cur_path_increment = next_path_increment

    return res_base_waypoints, res_base_cmds, res_idx

def get_vel_signs(t):
    v, w = get_desired_base_vel(t[3], t[4], scale=1)
    v_sign = -1 if v < 0 else 1
    if w > 0:
        w_sign = 1
    elif w < 0:
        w_sign = -1
    else:
        w_sign = 0
    return v_sign, w_sign

def get_acc_dec_for_each_segment(base_traj, base_rs_indices):
    velocity_signs = []
    for i in base_rs_indices:
        t = list(base_traj[i])
        v, w = get_desired_base_vel(t[3], t[4], scale=1)
        v_sign = -1 if v < 0 else 1
        if w > 0:
            w_sign = 1
        elif w < 0:
            w_sign = -1
        else:
            w_sign = 0
        velocity_signs.append((v_sign, w_sign))

    should_accelerate = [True]
    should_decelerate = []
    for i in range(len(velocity_signs) - 1):
        v0, w0 = velocity_signs[i]
        v1, w1 = velocity_signs[i + 1]
        if v1 * v0 < 0 or w0 * w1 < 0:
            should_decelerate.append(True)
            should_accelerate.append(True)
        else:
            should_decelerate.append(False)
            should_accelerate.append(False)
    should_decelerate.append(True)
    return should_accelerate, should_decelerate


def interpolate_arm(cur_q, target_q, num_steps):
    res_arm_waypoints = [cur_q]

    v1 = np.array(cur_q[3:])
    v2 = np.array(target_q[3:])

    arm_diff = v2 - v1
    actual_step_size = arm_diff / (num_steps - 1)

    for i in range(1, num_steps):
        nodepos = v1 + actual_step_size * i
        res_arm_waypoints.append(nodepos.tolist())

    return res_arm_waypoints


def get_desired_base_vel(rs_type, rs_len, scale):
    print(rs_type)
    if math.fabs(rs_len) < 1e-5:
        return [0, 0]

    forward = 1 if rs_len > 0 else -1
    linear_speed = scale * BASE_SPEED * forward
    angular_speed = scale * BASE_SPEED / TURNING_RADIUS
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
RLSEnv = RLSEnv()

with open(os.path.join(CUR_DIR, "map/occ_grid_v2.npy"), 'rb') as f:
    occ_grid = np.load(f)

base_x_bounds = [0, occ_grid.shape[0] * OCC_GRID_RESOLUTION]
base_y_bounds = [0, occ_grid.shape[1] * OCC_GRID_RESOLUTION]
print(base_x_bounds, base_y_bounds)
RLSEnv.robot.set_base_bounds(base_x_bounds, base_y_bounds)
RLSEnv.robot_base.set_base_bounds(base_x_bounds, base_y_bounds)

RLSEnv.load_mesh(osp.join(CUR_DIR, "map/rls_v4.obj"))
RLSEnv.load_occupancy_grid(occ_grid)

low = RLSEnv.robot.get_joint_lower_bounds()
high = RLSEnv.robot.get_joint_higher_bounds()

# Generate env
# for i in range(10):
#     path = []
#     while len(path) <= 2:
#         RLSEnv.sample_start_goal()
#         res, path = RLSEnv.plan(allowed_time=20, interpolate=False)

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
#     RLSEnv.start = path[0]
#     RLSEnv.goal = path[-1]
#     start_time = time.time()
#     res, path = RLSEnv.plan(allowed_time=10, interpolate=False)
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
    # RLSEnv.sample_start_goal(joint_state=cur_joint_state)

    # T0:
    # goal = [5.2, 1.05, -1.7043219922841046, 0.184, -0.8968170951812744, 0.38978915650634766, -1.3763065746292114, -2.1162578413635256, -3.0271986299713136, -1.2057725833190918, -2.035894721689453]

    # T1:
    # goal = [8.4, 3.6, 1.0, 0.38, 0.202, -0.681, -0.676, 0.928, 0.543, 1.503, 0.0]

    # T2:
    # goal = [8.5, 6.15, 1.628, 0.366, -1.5460745166748047, -0.7265653089782715, -1.097505491064453, -1.3891512701660156, 1.3726417249481202, 0.39647036330566404, 0.8065717518899536]

    # T3:
    # goal = [11.3, 2.1, -3.0, 0.31, 0.3974791992694855, -0.320060677935791, -0.3124906530841827, 0.3534510781616211, 0.31266097486419675, 1.5672810627685547, 2.361261040029297]

    # T4
    # goal = [3.7, 2.2, 1.0239268783749709, 0.12, -0.4822588335483551, -0.44277853529663086, -2.2594961336120605, -1.161355192437744, -2.381776409741211, 0, 1.7729792893502807]

    # New T1
    # goal = [9.9, 3.3, -3.13, 0.15, -1.3685162137954712, 1.1782551332214355, -0.1, -0.6045200178771972, -1.2987856441696166, 0.539130504296875, -1.478676170053711]
    # goal = [8.5, 6.15, 1.628, 0.366, -1.5460745166748047, -0.7265653089782715, -1.097505491064453, -1.3891512701660156, 1.3726417249481202, 0.39647036330566404, 0.8065717518899536]

    # New T2
    # goal = [7.5, 3, 1.8, 0.1, -0.2, 0.8, 0.0, 0.07503349088134766, 0.1, -0.5, 0.12164933240152359]
    # goal = [9.6, 3.85, 1.4, 0.35, -0.1, -0.2, -1.7275880983337402, 0.85, 1.0301804965774537, -0.08481598360671996, 0.43266394740081787]

    # New T3
    goal = [7.5, 3, 1.8, 0.1, -0.2, 0.8, 0.0, 0.07503349088134766, 0.1, 0.4, 0.12164933240152359]
    # goal = [9.6, 4.05, 1.4, 0.36, -0.1, -0.86, -1.7275880983337402, 0.85, 1.0301804965774537, 0.3, 0]
    # goal = [7.5, 3, 1.8, 0.36, -0.1, -0.86, -1.7275880983337402, 0.85, 1.0301804965774537, 0.3, 0]

    cur_robot_state = fetch.get_current_state()
    # print(cur_robot_state)
    # cur_robot_state = [1.090546test_path9025325267, 4, -1.024216337017743, 0.2, 1.3194016147644043, 1.4002991243103027, -0.20051024465408326, 1.519276445135498, -0.0002711178322315211, 1.6581695629821778, 0.00046484643704652797]
    RLSEnv.robot.set_state(cur_robot_state)

    RLSEnv.robot.set_state(goal)
    assert RLSEnv.pb_ompl_interface.is_state_valid(goal)

    RLSEnv.robot.set_state(cur_robot_state)
    assert RLSEnv.pb_ompl_interface.is_state_valid(cur_robot_state)


    RLSEnv.sample_start_goal(goal_state=goal)
    RLSEnv.start = cur_robot_state
    if REVERSE:
        start, goal = RLSEnv.start, RLSEnv.goal
        RLSEnv.start = goal
        RLSEnv.goal = start

    res, path = RLSEnv.plan(allowed_time=20, interpolate=False)
    if REVERSE:
        path.reverse()

    # cur_robot_state[2] = 0
    # goal_robot_state = goal = [cur_robot_state[0] + 2, cur_robot_state[1], cur_robot_state[2], 0.184, -0.8968170951812744, 0.38978915650634766, -1.3763065746292114, -2.1162578413635256, -3.0271986299713136, -1.2057725833190918, -2.035894721689453]
    # path = [cur_robot_state, goal_robot_state]
    # res = True
else:
    with open("test_path_bk.json", "r") as f:
        path = json.load(f)
    # path = [[9.865162590927758, 3.5829416789439286, 3.086932780895325, 0.15282541513442993, -1.3673657249420166, 1.1774883790710449, -0.09965081124153137, -0.6041366408020019, -1.2972517782409667, 0.536829765008545, -1.4790596663381959], [9.922628226445743, 3.4767797252808483, -2.392566247682002, 0.21500962369768845, -0.9544758389380307, 0.9978220288350885, 0.29168935751002417, 0.11236719252427818, -1.087945636745846, 0.7230299959793658, -1.1559670763499215], [8.75777515240012, 3.4370916900992485, 2.408557396120326, 0.17166095673902565, -0.30664046222076324, 0.8340678170421018, -0.7773893077425539, 0.03935294577606538, 0.05334478084296301, -0.1592737565754524, -1.2484547197111138], [7.989904237840142, 4.014346290030626, 2.764743205431914, 0.21194056709780162, -0.057748985996715296, 0.18465853095455176, 0.13487687112416558, -0.4501631934624408, -0.39044599833585525, 0.5918580040565522, -0.31681307192074304], [7.639189769216285, 5.286399403594076, 1.20741801297028, 0.3118659742524399, -0.9688226734107048, -0.6999892145125767, 0.07510411616389229, -0.7649534537971179, 0.9161797344163818, 0.6832545434716475, 0.7424377246792097], [8.5, 6.15, 1.628, 0.366, -1.5460745166748047, -0.7265653089782715, -1.097505491064453, -1.3891512701660156, 1.3726417249481202, 0.39647036330566404, 0.8065717518899536]]
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
    #     tmp_path = utils.rrt_extend_intermediate(RLSEnv, path[i - 1], path[i])
    #     new_path += tmp_path

    print("sanity check..")
    # new_path = utils.interpolate(path)
    # for point in path[1:]:
    #     assert RLSEnv.pb_ompl_interface.is_state_valid(point)
    # for point in interpolated_path[1:]:
    #     assert RLSEnv.pb_ompl_interface.is_state_valid(point)
    # if REAL_ROBOT:
    #     assert fetch.check_traj_col_free(point)

    input("press anything to visualize in simulator")

    RLSEnv.execute(interpolated_path)
    RLSEnv.robot.set_state(path[0])

    key = input("press y to execute on real robot")
    if key != "y":
        exit()

    # Find out which part has changing directions
    interpolated_base_path = []  # include start and finish
    interpolated_arm_path = []  # include start and finish
    traj_indices = []  # indices where changing directions happens
    prev_v_sign, prev_w_sign = None, None
    for i in range(1, len(path)):
        q0 = path[i - 1]
        q1 = path[i]
        base_traj = rs.path_sample(q0[:3], q1[:3], TURNING_RADIUS, BASE_STEP_SIZE)

        base_num_steps = 0
        for idx in range(len(base_traj)):
            t = list(base_traj[idx])
            if prev_v_sign is None:
                prev_v_sign, prev_w_sign = get_vel_signs(t)
            t[2] = utils.enforce_pi(t[2])
            t[3] = get_path_type(t[3])

            if math.fabs(t[4]) < 1e-5:
                continue

            interpolated_base_path.append(t)
            base_num_steps += 1
            v_sign, w_sign = get_vel_signs(t)

            # if idx >= 1:
            #     prev_t = list(base_traj[idx - 1])
            #     prev_t[2] = utils.enforce_pi(prev_t[2])
            #     prev_t[3] = get_path_type(prev_t[3])
            #     if t[3] != prev_t[3] or t[4] != prev_t[4]:
            #         print(prev_t, t)
            #         print(prev_v_sign, v_sign)
            #         print(prev_w_sign, w_sign)
            #         print("here")

            if v_sign * prev_v_sign < 0 or w_sign * prev_w_sign < 0:
                traj_indices.append(len(interpolated_base_path) - 1)
            prev_v_sign = v_sign
            prev_w_sign = w_sign

        arm_path = interpolate_arm(q0, q1, num_steps=base_num_steps)
        interpolated_arm_path.extend(arm_path)

    # Sanity check
    for idx in traj_indices:
        assert interpolated_base_path[idx - 1][3] != interpolated_base_path[idx][3] or interpolated_base_path[idx - 1][4] != interpolated_base_path[idx][4]
    assert len(interpolated_arm_path) == len(interpolated_base_path)

    traj_indices.insert(0, 0)
    traj_indices.append(len(interpolated_base_path) - 1)
    print(traj_indices)

    # should_accelerate, should_decelerate = get_acc_dec_for_each_segment(all_base_traj, traj_indices)

    # total_path_len = utils.get_path_len(path)
    target_base_states = []
    actual_base_states = []
    total_desired_vel = []
    for i in range(1, len(traj_indices)):
        idx1 = traj_indices[i - 1]
        idx2 = traj_indices[i]
        cur_base_q = interpolated_base_path[idx1]
        target_base_q = interpolated_base_path[idx2]

        cur_base_path = [cur_base_q, target_base_q]
        total_base_path_len = utils.calc_path_len_base(cur_base_path)

        # total_arm_path_len = calc_path_len_arm_rs(cur_base_path)
        # total_time = total_base_path_len / BASE_SPEED

        # num_steps = round(total_time / step_time)
        # base_step_size = total_base_path_len / num_steps
        # arm_step_size = total_arm_path_len / num_steps

        # base_waypoints, desired_base_vel = interpolate_base(
        #     cur_q, target_q, should_accelerate[i - 1], should_decelerate[i - 1]
        # )

        # After parameterization, traj_idx looks like 0, 1, 4, 10... because the robot is accelerating.
        # We use that to select the actual waypoint form both interpolated base path and interpolated arm path.
        base_waypoints, desired_base_vel, traj_idx = parameterize_base_path(interpolated_base_path[idx1:idx2])
        arm_path = interpolated_arm_path[idx1:idx2]
        arm_waypoints = [arm_path[idx] for idx in traj_idx]
        assert len(desired_base_vel) == len(base_waypoints), "Waypoint numbers mismatch for base and arm"
        assert len(base_waypoints) == len(arm_waypoints)
        print(len(desired_base_vel))
        # print(interpolated_base_path[idx1:idx2])

        # for idx in range(idx1+1, idx2):
        #     if interpolated_base_path[idx][3] != interpolated_base_path[idx - 1][3] or interpolated_base_path[idx][4] != interpolated_base_path[idx - 1][4]:
        #         print(interpolated_base_path[idx - 1], interpolated_base_path[idx])

        # print(desired_base_vel)
        total_desired_vel += desired_base_vel

        _, actual_base_state = fetch.move_wb_ctrl(arm_waypoints, step_time, base_waypoints, desired_base_vel)

        target_base_states += base_waypoints
        actual_base_states += actual_base_state

    print(total_desired_vel)
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    axes[0].plot(np.array(target_base_states)[:, 0])
    axes[0].plot(np.array(actual_base_states)[:, 0])
    axes[0].legend(["target", "actual"])
    axes[1].plot(np.array(target_base_states)[:, 1])
    axes[1].plot(np.array(actual_base_states)[:, 1])
    axes[1].legend(["target", "actual"])
    axes[2].plot(np.array(target_base_states)[:, 2])
    axes[2].plot(np.array(actual_base_states)[:, 2])
    axes[2].legend(["target", "actual"])
    plt.show()
