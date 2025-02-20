from scipy.spatial.transform import Rotation as R
import math

from .pb_ompl import PbOMPLRobot

class MyPlanarRobot(PbOMPLRobot):
    def __init__(self, id, p, base_xy_bounds=5.0) -> None:
        self.id = id
        self.num_dim = 8
        self.joint_idx=[0,1,2,3,4,5]
        self.p = p

        self.joint_bounds = []
        self.joint_bounds.append([-base_xy_bounds, base_xy_bounds]) # x
        self.joint_bounds.append([-base_xy_bounds, base_xy_bounds]) # y
        # self.joint_bounds.append([math.radians(-180), math.radians(180)]) # theta
        self.joint_bounds.append([math.radians(-180), math.radians(180)]) # joint_0
        self.joint_bounds.append([math.radians(-180), math.radians(180)]) # joint_1
        self.joint_bounds.append([math.radians(-180), math.radians(180)]) # joint_2
        self.joint_bounds.append([math.radians(-180), math.radians(180)]) # joint_3
        self.joint_bounds.append([math.radians(-180), math.radians(180)]) # joint_4
        self.joint_bounds.append([math.radians(-180), math.radians(180)]) # joint_5

        # self.reset()

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

    def get_cur_state(self):
        return self.state

    def set_state(self, state):
        pos = [state[0], state[1], 0]
        r = R.from_euler('z', 0)
        quat = r.as_quat()
        self.p.resetBasePositionAndOrientation(self.id, pos, quat)
        self._set_joint_positions(self.joint_idx, state[2:])

        self.state = state

    def reset(self):
        self.p.resetBasePositionAndOrientation(self.id, [0,0,0], [0,0,0,1])
        self._set_joint_positions(self.joint_idx, [0,0])
        self.state = [0] * self.num_dim

    def _set_joint_positions(self, joints, positions):
        for joint, value in zip(joints, positions):
            self.p.resetJointState(self.id, joint, value, targetVelocity=0)

class MyPlanarRobotBase(MyPlanarRobot):
    def __init__(self, id, p, base_xy_bounds=5.0) -> None:
        MyPlanarRobot.__init__(self, id, p, base_xy_bounds)
        self.num_dim = 2

        self.joint_bounds = []
        self.joint_bounds.append([-base_xy_bounds, base_xy_bounds]) # x
        self.joint_bounds.append([-base_xy_bounds, base_xy_bounds]) # y

        self.fixed_joint_pos = [0, math.radians(125), math.radians(45), math.radians(45), math.radians(90), math.radians(45)]

    def set_state(self, state):
        pos = [state[0], state[1], 0]
        r = R.from_euler('z', 0)
        quat = r.as_quat()
        self.p.resetBasePositionAndOrientation(self.id, pos, quat)
        self._set_joint_positions()
        self.state = state

    def reset(self):
        self.p.resetBasePositionAndOrientation(self.id, [0,0,0], [0,0,0,1])
        self._set_joint_positions()
        self.state = [0] * self.num_dim

    def _set_joint_positions(self):
        for joint, value in enumerate(self.fixed_joint_pos):
            self.p.resetJointState(self.id, joint, value, targetVelocity=0)
