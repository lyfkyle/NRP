import numpy as np
from scipy.spatial.transform import Rotation as R
import math

from nrp.env.fetch_11d.pb_ompl import PbOMPLRobot
import nrp.env.fetch_11d.pb_utils as pb_utils

class Fetch(PbOMPLRobot):
    """
    Fetch Robot
    Reference: https://fetchrobotics.com/robotics-platforms/fetch-mobile-manipulator/
    Uses joint velocity control
    """

    def __init__(self, id, p, base_xy_bounds=5.0):
        self.id = id
        self.num_dim = 11
        self.p = p
        self.disabled_collision_pair = set()

        disable_collision_names = [
            ["torso_lift_joint", "torso_fixed_joint"],
            ["torso_lift_joint", "shoulder_lift_joint"],
            ["caster_wheel_joint", "estop_joint"],
            ["caster_wheel_joint", "laser_joint"],
            ["caster_wheel_joint", "torso_fixed_joint"],
            ["caster_wheel_joint", "l_wheel_joint"],
            ["caster_wheel_joint", "r_wheel_joint"],
        ]
        for names in disable_collision_names:
            link_a, link_b = pb_utils.joints_from_names(self.p, id, names)
            self.disabled_collision_pair.add((link_a, link_b))
            self.disabled_collision_pair.add((link_b, link_a))
            self.p.setCollisionFilterPair(id, id, link_a, link_b, 0)

        # disable_collision_link_names = [
        #     ["torso_lift_link", "torso_fixed_link"],
        # ]
        # for names in disable_collision_link_names:
        #     link_a, link_b = pb_utils.links_from_names(self.p, id, names)
        #     self.p.setCollisionFilterPair(id, id, link_a, link_b, 0)

        arm_joints = pb_utils.joints_from_names(self.p,
            id,
            [
                "torso_lift_joint",
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "upperarm_roll_joint",
                "elbow_flex_joint",
                "forearm_roll_joint",
                "wrist_flex_joint",
                "wrist_roll_joint",
            ],
        )
        print("Arm joints:", arm_joints)
        self.joint_idx = arm_joints
        self.eef_joint_idx = arm_joints[-1]

        # all_joint_indices = pb_utils.get_joints(self.p, id)
        # for joint_idx in all_joint_indices:
        #     joint_info = pb_utils.get_joint_info(self.p, id, joint_idx)
        #     if joint_info.jointType != self.p.JOINT_FIXED:
        #         print(joint_info.jointName)

        self.rest_position = (0.02, np.pi / 2.0 - 0.4, np.pi / 2.0 - 0.1, -0.4, np.pi / 2.0 + 0.1, 0.0, np.pi / 2.0, 0.0)
        self.state = [0, 0, 0] + list(self.rest_position)
        self._set_joint_positions(arm_joints, self.rest_position)

        # bounds
        self.joint_bounds = []
        self.joint_bounds.append([-base_xy_bounds, base_xy_bounds]) # x
        self.joint_bounds.append([-base_xy_bounds, base_xy_bounds]) # y
        self.joint_bounds.append([math.radians(-180), math.radians(180)]) # theta
        for j in self.joint_idx:
            joint_info = self.p.getJointInfo(self.id, j)
            if joint_info[8] > joint_info[9]:
                self.joint_bounds.append([math.radians(-180), math.radians(180)])
            else:
                self.joint_bounds.append([joint_info[8], joint_info[9]])

        # print(self.joint_bounds)

    def set_base_bounds(self, base_x_bounds, base_y_bounds):
        self.joint_bounds[0] = base_x_bounds
        self.joint_bounds[1] = base_y_bounds

    def get_joint_bounds(self):
        return self.joint_bounds

    def get_joint_lower_bounds(self):
        robot_bounds_low = [bound[0] for bound in self.joint_bounds]
        return robot_bounds_low

    def get_joint_higher_bounds(self):
        robot_bounds_high = [bound[1] for bound in self.joint_bounds]
        return robot_bounds_high

    def get_link_state(self):
        link_pos = []
        for joint_idx in self.joint_idx:
            res = self.p.getLinkState(self.id, joint_idx, computeForwardKinematics=True)
            link_pos.append(res[4])

        return link_pos

    def get_cur_state(self):
        return self.state

    def set_state(self, state):
        pos = [state[0], state[1], 0]
        r = R.from_euler('z', state[2]) # unit is radians
        quat = r.as_quat()
        self.p.resetBasePositionAndOrientation(self.id, pos, quat)
        self._set_joint_positions(self.joint_idx, state[3:])

        self.state = state

    def reset(self):
        self.p.resetBasePositionAndOrientation(self.id, [0,0,0], [0,0,0,1])
        self._set_joint_positions(self.joint_idx, self.reset_position)
        self.state = [0] * self.num_dim

    def _set_joint_positions(self, joints, positions):
        for joint, value in zip(joints, positions):
            self.p.resetJointState(self.id, joint, value, targetVelocity=0)

class FetchBase(Fetch):
    """
    Fetch Robot
    Reference: https://fetchrobotics.com/robotics-platforms/fetch-mobile-manipulator/
    Uses joint velocity control
    """

    def __init__(self, id, p, base_xy_bounds=5.0):
        super().__init__(id, p, base_xy_bounds)
        self.num_dim = 2

        # bounds
        self.joint_bounds = []
        self.joint_bounds.append([-base_xy_bounds, base_xy_bounds]) # x
        self.joint_bounds.append([-base_xy_bounds, base_xy_bounds]) # y

        print(self.joint_bounds)

    def set_state(self, state):
        pos = [state[0], state[1], 0]
        r = R.from_euler('z', state[2])
        quat = r.as_quat()
        self.p.resetBasePositionAndOrientation(self.id, pos, quat)

        self.state = state

    def reset(self):
        self.p.resetBasePositionAndOrientation(self.id, [0,0,0], [0,0,0,1])

