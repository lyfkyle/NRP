#!/usr/bin/env python

import actionlib
import control_msgs.msg
import trajectory_msgs.msg
import math
import moveit_commander
import rospy
import tf
import math
import time
import os
import trimesh
import rospy
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus
from moveit_msgs.msg import MoveItErrorCodes, MoveGroupAction, RobotState
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest, GetStateValidity, GetStateValidityRequest, GetStateValidityResponse
from tf.listener import TransformListener

from geometry_msgs.msg import Twist, Pose, PoseStamped
from sensor_msgs.msg import JointState
from math import sqrt, pow, sin, cos, atan2


CUR_DIR = os.path.dirname(os.path.abspath(__file__))

# ARM_GROUP_NAME = 'arm'
JOINT_ACTION_SERVER = '/arm_with_torso_controller/follow_joint_trajectory'
MOVE_GROUP_ACTION_SERVER = 'move_group'
# TIME_FROM_START = 5
DEGS_TO_RADS = math.pi / 180

K1 = 1.0
K2 = 1.0
K3 = 3.0


class ArmJoints(object):
    """ArmJoints holds the positions of the Fetch's arm joints.
    When setting values, it also enforces joint limits.
    js = ArmJoints()
    js.set_shoulder_pan(1.5)
    js.set_shoulder_lift(-0.6)
    js.set_upperarm_roll(3.0)
    js.set_elbow_flex(1.0)
    js.set_forearm_roll(3.0)
    js.set_wrist_flex(1.0)
    js.set_wrist_roll(3.0)
    """
    JOINT_LIMITS = {
        "torso_lift": (0.0, 0.4),
        'shoulder_pan': (-92 * DEGS_TO_RADS, 92 * DEGS_TO_RADS),
        'shoulder_lift': (-70 * DEGS_TO_RADS, 87 * DEGS_TO_RADS),
        'elbow_flex': (-129 * DEGS_TO_RADS, 129 * DEGS_TO_RADS),
        'wrist_flex': (-125 * DEGS_TO_RADS, 125 * DEGS_TO_RADS)
    }

    def __init__(self):
        self.shoulder_pan = 0
        self.shoulder_lift = 0
        self.upperarm_roll = 0
        self.elbow_flex = 0
        self.forearm_roll = 0
        self.wrist_flex = 0
        self.wrist_roll = 0

    @staticmethod
    def from_list(vals):
        if len(vals) != 8:
            rospy.logerr('Need 7 values to create ArmJoints (got {})'.format(
                len(vals)))
            return None
        j = ArmJoints()
        j.set_torso_lift(vals[0])
        j.set_shoulder_pan(vals[1])
        j.set_shoulder_lift(vals[2])
        j.set_upperarm_roll(vals[3])
        j.set_elbow_flex(vals[4])
        j.set_forearm_roll(vals[5])
        j.set_wrist_flex(vals[6])
        j.set_wrist_roll(vals[7])
        return j

    @staticmethod
    def names():
        return [
            'torso_lift_joint', 'shoulder_pan_joint', 'shoulder_lift_joint', 'upperarm_roll_joint',
            'elbow_flex_joint', 'forearm_roll_joint', 'wrist_flex_joint', 'wrist_roll_joint'
        ]

    def values(self):
        return [
            self.shoulder_pan, self.shoulder_lift, self.upperarm_roll,
            self.elbow_flex, self.forearm_roll, self.wrist_flex,
            self.wrist_roll
        ]

    def _clamp_val(self, val, joint_name):
        if joint_name in ArmJoints.JOINT_LIMITS:
            limits = ArmJoints.JOINT_LIMITS[joint_name]
            min_val, max_val = limits[0], limits[1]
            final_val = min(max(val, min_val), max_val)
            if val != final_val:
                rospy.logwarn('{} not in [{}, {}] for {} joint.'.format(
                    val, min_val, max_val, joint_name))
            return final_val
        else:
            return val

    def set_torso_lift(self, val):
        val = self._clamp_val(val, "torso_lift")
        self.torso_lift = val

    def set_shoulder_pan(self, val):
        val = self._clamp_val(val, 'shoulder_pan')
        self.shoulder_pan = val

    def set_shoulder_lift(self, val):
        val = self._clamp_val(val, 'shoulder_lift')
        self.shoulder_lift = val

    def set_upperarm_roll(self, val):
        val = self._clamp_val(val, 'upperarm_roll')
        self.upperarm_roll = val

    def set_elbow_flex(self, val):
        val = self._clamp_val(val, 'elbow_flex')
        self.elbow_flex = val

    def set_forearm_roll(self, val):
        val = self._clamp_val(val, 'forearm_roll')
        self.forearm_roll = val

    def set_wrist_flex(self, val):
        val = self._clamp_val(val, 'wrist_flex')
        self.wrist_flex = val

    def set_wrist_roll(self, val):
        val = self._clamp_val(val, 'wrist_roll')
        self.wrist_roll = val

class FetchRobot(object):
    """Arm controls the robot's arm.
    Joint space control:
        joints = ArmJoints()
        # Fill out joint states
        arm = fetch_api.Arm()
        arm.move_to_joints(joints)
    """

    def __init__(self, use_moveit=False):
        self._joint_client = actionlib.SimpleActionClient(JOINT_ACTION_SERVER, control_msgs.msg.FollowJointTrajectoryAction)
        print("Fetch: waiting for action server!!")
        self._joint_client.wait_for_server()

        if use_moveit:
            self._move_group_client = actionlib.SimpleActionClient(MOVE_GROUP_ACTION_SERVER, MoveGroupAction)
            self._move_group_client.wait_for_server(rospy.Duration(10))
            moveit_commander.roscpp_initialize(sys.argv)
            self.robot = moveit_commander.RobotCommander()
            self.scene = moveit_commander.PlanningSceneInterface()
            self.move_group = moveit_commander.MoveGroupCommander("arm_with_torso")
            self.move_group.set_max_velocity_scaling_factor(0.4) # to reduce fetch arm movement speed
            self.move_group.set_planner_id("RRTstarkConfigDefault") # to produce optimal trajectory
            # self.move_group.set_planner_id("BITstarkConfigDefault") # to produce optimal trajectory

        self._vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self._twist_msg = Twist()

        self._col_checker = rospy.ServiceProxy("/check_state_validity", GetStateValidity)
        self._tf_listener = tf.TransformListener()

    def move_wb_open_loop(self, joint_states, step_time, base_vel, total_time):
        self.move_to_joints(joint_states, duration=step_time)
        self.move_to_base_open_loop(base_vel, total_time)
        res = self._joint_client.wait_for_result()
        print(res)
        return res

    def move_wb_ctrl(self, joint_states, step_time, base_states, base_vel):
        rate = rospy.Rate(int(1 / step_time))

        self.move_to_joints(joint_states, duration=step_time)
        actual_base_states = []
        for i in range(1, len(base_states)):  # the first state is the current state, so we ignore
            cur_base_state = self.base_control(base_states[i], base_vel[i])
            actual_base_states.append(cur_base_state)
            rate.sleep()

        res = self._joint_client.wait_for_result()
        return res, actual_base_states

    def move_to_joints(self, joint_states, duration=0.5):
        # TODO
        # clamp joint limits

        goal = control_msgs.msg.FollowJointTrajectoryGoal()
        goal.trajectory.joint_names.extend(ArmJoints.names())
        for i, joint_state in enumerate(joint_states):
            if i > 0:
                point = trajectory_msgs.msg.JointTrajectoryPoint()
                point.positions = joint_state
                point.time_from_start = rospy.Duration(duration * i)
                goal.trajectory.points.append(point)
        self._joint_client.send_goal(goal)
        # self._joint_client.wait_for_result(rospy.Duration(10))

    def move_to_base_open_loop(self, cmd, applied_time):
        self._twist_msg.linear.x = cmd[0]
        self._twist_msg.angular.z = cmd[1]
        cur_time = time.time()
        end_time = cur_time + applied_time
        while cur_time < end_time:
            self._vel_pub.publish(self._twist_msg)
            cur_time = time.time()

    def base_control(self, base_pose_target, base_vel_target, k1=1.5, k2=1.5, k3=3):
        desired_v = base_vel_target[0]
        desired_w = base_vel_target[1]
        desired_yaw = base_pose_target[2]
        cur_base_state = self.get_current_base_state()

        desired_base_state = np.array([base_pose_target[2], base_pose_target[0], base_pose_target[1]])
        cur_base_pose = np.array([cur_base_state[2], cur_base_state[0], cur_base_state[1]])  # swap into theta, x, y for control calculation
        rot = np.array([
            [1, 0, 0.],
            [0, math.cos(desired_yaw), math.sin(desired_yaw)],
            [0, -math.sin(desired_yaw), math.cos(desired_yaw)]
        ])
        pose_error = np.matmul(rot, (cur_base_pose - desired_base_state))
        yaw_error, x_error, y_error = pose_error[0], pose_error[1], pose_error[2]
        yaw_error = self.wrap_angle(yaw_error)

        v = (desired_v - k1 * math.fabs(desired_v) * (x_error + y_error * math.tan(yaw_error))) / math.cos(yaw_error)
        w = desired_w - (k2 * desired_v * y_error + k3 * math.fabs(desired_v) * math.tan(yaw_error)) * (math.cos(yaw_error) ** 2)
        # v = desired_v
        # w = desired_w

        self._twist_msg.linear.x = v
        self._twist_msg.angular.z = w
        # print(base_pose_target, cur_base_state, v, w)
        self._vel_pub.publish(self._twist_msg)

        return cur_base_state

    def get_current_state(self):
        base_state = self.get_current_base_state()
        arm_state = self.get_current_joint_state()
        return base_state + arm_state

    def get_current_base_state(self):
        while not rospy.is_shutdown():
            try:
                (trans, rot) = self._tf_listener.lookupTransform('/map', '/base_link', rospy.Time(0))
                break
            except Exception as e:
                print(e)
                continue

        r = R.from_quat(rot)
        rpy = r.as_euler('xyz')
        trans[2] = rpy[2]
        return trans


    def get_current_joint_state(self):
        while True:
            msg = rospy.wait_for_message("/joint_states", JointState, timeout=5)
            all_joint_states = list(msg.position)
            if len(all_joint_states) > 2:
                break
        joint_states = [all_joint_states[2]] + all_joint_states[6:]
        return joint_states

    def check_traj_col_free(self, traj):
        for pos in traj:
            req = GetStateValidityRequest()
            robot_state = RobotState()
            robot_state.is_diff = False
            robot_state.joint_state.name = ArmJoints.names()
            robot_state.joint_state.position = pos
            req.robot_state = robot_state
            req.group_name = "arm_with_torso"
            resp = self._col_checker.call(req)
            if not resp.valid:
                print("{} is not valid".format(pos))
            return False

        return True

    def wrap_angle(self, angle):
        if angle > math.pi:
            angle = angle - 2 * math.pi
        elif angle < -math.pi:
            angle = angle + 2 * math.pi
        return angle

    def arm_move_to_joint(self,
                      joints,
                      allowed_planning_time=5.0,
                      execution_timeout=float("inf"),
                      num_planning_attempts=200,
                      plan_only=False,
                      replan=False,
                      replan_attempts=1,
                      tolerance=0.004,
                      return_plan=False):

        self.move_group.set_num_planning_attempts(num_planning_attempts)
        self.move_group.set_planning_time(allowed_planning_time)
        self.move_group.set_goal_joint_tolerance(tolerance)

        res = self.move_group.go(joints, wait=True)

        # Calling ``stop()`` ensures that there is no residual movement
        self.move_group.stop()
        return res

    def move_base(self, pose):
        """
        Navigate to a position in map
        Args:
            position: the geometry_msgs/Pose
        """
        client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        # Waits until the action server has started up and started
        # listening for goals.
        client.wait_for_server()

        # Creates a goal to send to the action server.
        goal = MoveBaseGoal()

        # Sends the goal to the action server.
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.pose = pose
        client.send_goal(goal)

        # Waits for the server to finish performing the action.
        client.wait_for_result()

        # Prints out the result of executing the action
        # res = client.get_state()

        # return res == GoalStatus.SUCCEEDED
        return

    def add_local_obstacle(self, local_occ_grid, cur_yaw):
        self.scene.clear()

        print(local_occ_grid.shape, local_occ_grid.max())
        file_name = os.path.join(CUR_DIR, "local_mesh.obj")
        local_mesh = trimesh.voxel.ops.matrix_to_marching_cubes(local_occ_grid, pitch=0.1)
        with open(file_name, 'w') as f:
            local_mesh.export(f, file_type='obj')

        pose = PoseStamped()
        pose.header.frame_id="base_link"
        pose.pose.position.x = -2 * cos(-cur_yaw) - (-2) * sin(-cur_yaw)
        pose.pose.position.y = -2 * sin(-cur_yaw) + (-2) * cos(-cur_yaw)
        r = R.from_euler('xyz', [0, 0, -cur_yaw])
        quat = r.as_quat()
        pose.pose.orientation.x = quat[0]
        pose.pose.orientation.y = quat[1]
        pose.pose.orientation.z = quat[2]
        pose.pose.orientation.w = quat[3]
        self.scene.add_mesh("local_obstacle", pose, file_name)

        os.remove(file_name)  # must delete the object file
        rospy.sleep(1.0)

    # def set_height(self, height, duration=5):
    #     """Sets the torso height.
    #     This will always take ~5 seconds to execute.
    #     Args:
    #         height: The height, in meters, to set the torso to. Values range
    #             from Torso.MIN_HEIGHT (0.0) to Torso.MAX_HEIGHT(0.4).
    #     """
    #     goal = control_msgs.msg.FollowJointTrajectoryGoal()
    #     goal.trajectory.joint_names.append("torso_lift_joint")
    #     point = trajectory_msgs.msg.JointTrajectoryPoint()
    #     point.positions.append(height)
    #     point.time_from_start = rospy.Duration(duration)
    #     goal.trajectory.points.append(point)
    #     self._joint_client.send_goal(goal)
    #     self._joint_client.wait_for_result(rospy.Duration(10))

if __name__ == "__main__":
    rospy.init_node("test")
    fetch = FetchRobot()
    cur_robot_state = fetch.get_current_state()
    print(cur_robot_state)
    # fetch.arm_move_to_joint([0.2, 1.3205522228271485, 1.399532370159912, -0.19974325208511354, 1.719844644293213, 0.0004958728740930562, 1.4, 0])