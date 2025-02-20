#!/usr/bin/env python

import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from math import pi
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
from moveit_msgs.srv import GetStateValidity, GetStateValidityRequest
import random

# 
DEBUG = True
def dbg_print(txt):
    if DEBUG:
        print(txt)

# --------- Settings ------------
BASE_GOAL = [0.2, -1.7, 0]
JOINT1_GOAL = 0
JOINT2_GOAL = 0
JOINT3_GOAL = 0
JOINT4_GOAL = 0

class MoveGroupPythonInteface(object):
    """MoveGroupPythonIntefaceTutorial"""

    def __init__(self):
        super(MoveGroupPythonInteface, self).__init__()

        # First initialize `moveit_commander`_ and a `rospy`_ node:
        moveit_commander.roscpp_initialize(sys.argv)

        self._tuck_joints = ["torso_lift_joint", "shoulder_pan_joint", "shoulder_lift_joint", "upperarm_roll_joint",
                             "elbow_flex_joint", "forearm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"]
        self._tuck_joint_vals = [0.05, 1.32, 1.40, -0.2, 1.72, 0.0, 1.66, 0.0]

        # self.tuck()

    def tuck(self):
        group_name = "arm_with_torso"
        move_group = moveit_commander.MoveGroupCommander(group_name)
        move_group.set_planner_id("RRTConnect")
        move_group.set_planning_time(10)

        # The go command can be called with joint values, poses, or without any
        # parameters if you have already set the pose or joint target for the group
        move_group.go(self._tuck_joint_vals, wait=True)

        # Calling ``stop()`` ensures that there is no residual movement
        move_group.stop()
        rospy.loginfo("tuck finished") 


if __name__ == '__main__':
    rospy.init_node('move_group_python_interface_tutorial', anonymous=True)
    move_group_interface = MoveGroupPythonInteface()
    move_group_interface.tuck()
