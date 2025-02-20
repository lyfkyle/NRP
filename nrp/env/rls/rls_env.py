import os
import numpy as np
import pybullet as p
import pybullet_data
import math
import random
import json
import sys
import os.path as osp
import pybullet_utils.bullet_client as bc
import torch
from scipy.spatial.transform import Rotation as R
import time
import datetime

try:
    from ompl import util as ou
    from ompl import base as ob
    from ompl import geometric as og
except ImportError:
    # if the ompl module is not in the PYTHONPATH assume it is installed in a
    # subdirectory of the parent directory called "py-bindings."
    from os.path import abspath, dirname, join
    import sys

    sys.path.insert(0, join(dirname(abspath(__file__)), "../../third_party/ompl/py-bindings"))
    print(sys.path)
    from ompl import util as ou
    from ompl import base as ob
    from ompl import geometric as og

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), "../../"))

# import time
from nrp.env.fetch_11d.fetch_robot import Fetch, FetchBase
from nrp.env.rls import pb_ompl_real
from nrp.env.rls import utils
from nrp import ROOT_DIR

# from config import ROOT_DI

# sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
CUR_DIR = osp.dirname(osp.abspath(__file__))

# -------------- Settings ----------------
OCC_GRID_RESOLUTION = 0.1

class RLSEnv:
    EMPTY = 0
    GAP_ONLY = 1
    BOX_ONLY = 2
    GAP_AND_BOX = 3

    def __init__(self, gui=True, add_robot=True, load_floor=True):
        self.name = "fetch_11d"
        self.utils = utils  # this is weird, but works

        self.obstacles = []

        if gui:
            self.p = bc.BulletClient(connection_mode=p.GUI)
        else:
            self.p = bc.BulletClient(connection_mode=p.DIRECT)
        self.p.setGravity(0, 0, -9.8)
        self.p.setTimeStep(1.0 / 240.0)
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # load floor
        if load_floor:
            floor = os.path.join(pybullet_data.getDataPath(), "mjcf/ground_plane.xml")
            self.p.loadMJCF(floor)
        # self.p.loadURDF("plane_transparent.urdf")

        if add_robot:
            # load robot
            robot_model_path = osp.join(ROOT_DIR, "robot_model/fetch.urdf")
            print(robot_model_path)
            self.robot_id = self.p.loadURDF(robot_model_path, (-1, -1, 0), globalScaling=1)
            robot = Fetch(self.robot_id, self.p)
            self.robot = robot
            # self.robot_base = FetchBase(self.robot_id, self.p)

            # set up pb_ompl
            # self.pb_ompl_interface_base = pb_ompl.PbOMPLPRM(self.robot_base, self.obstacles, self.p)
            self.pb_ompl_interface = pb_ompl_real.PbOMPL(self.robot, self.obstacles, self.p, maze=self)
        else:
            self.robot = None

        # internal attributes
        self.goal_robot_id = None
        self.path = None
        self.approx_path = None
        self.sg_pairs = None

        # tmp mesh dir
        now = datetime.datetime.now()
        date_time = now.strftime("%m-%d-%Y-%H-%M-%S")
        self._tmp_mesh_dir = osp.join(CUR_DIR, "{}".format(date_time))

        self.obstacle_dict = {}
        self.fk = utils.FkTorch("cpu")

    def clear_obstacles(self):
        for obstacle in self.obstacles:
            self.p.removeBody(obstacle)
        self.occ_grid = None
        self.obstacles = []
        self.obstacle_dict = {}
        self.pb_ompl_interface.set_obstacles(self.obstacles)
        # self.pb_ompl_interface_base.set_obstacles(self.obstacles)

    def add_box(self, box_pos, half_box_size):
        colBoxId = self.p.createCollisionShape(p.GEOM_BOX, halfExtents=half_box_size)
        visualBoxId = self.p.createVisualShape(p.GEOM_BOX, halfExtents=half_box_size, rgbaColor=[0.5, 0.5, 0.5, 0.2])
        box_id = self.p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=colBoxId, baseVisualShapeIndex=visualBoxId, basePosition=box_pos
        )

        self.obstacles.append(box_id)

        return box_id

    def add_robot(self, robot_pos, rgba=None, scaling=1.0):
        robot_model_path = osp.join(ROOT_DIR, "robot_model/fetch.urdf")
        print(robot_model_path)
        self.robot_id = self.p.loadURDF(
            robot_model_path, (-1, -1, 0), flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES, globalScaling=scaling
        )
        robot = Fetch(self.robot_id, self.p)
        robot.set_state(robot_pos)

        if rgba is not None:
            num_links = self.p.getNumJoints(self.robot_id)
            for link in range(-1, num_links):  # Changes the color of the robot Ant
                self.p.changeVisualShape(self.robot_id, link, rgbaColor=rgba)

    def load_mesh(self, mesh_file):
        self.mesh = mesh_file
        collision_id = self.p.createCollisionShape(p.GEOM_MESH, fileName=mesh_file, flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
        # visualBoxId = self.p.createVisualShape(p.GEOM_MESH, fileName=mesh_file, rgbaColor=[0.2, 0.2, 0.2, 0.5])
        visualBoxId = self.p.createVisualShape(p.GEOM_MESH, fileName=mesh_file)
        mesh_body_id = self.p.createMultiBody(
            baseCollisionShapeIndex=collision_id, baseVisualShapeIndex=visualBoxId, basePosition=[0, 0, 0.06]
        )  # Raise mesh to prevent robot from spawning inside the hollow mesh
        self.obstacles.append(mesh_body_id)

        if self.robot is not None:
            # self.pb_ompl_interface_base.set_obstacles(self.obstacles)
            self.pb_ompl_interface.set_obstacles(self.obstacles)

    def load_occupancy_grid(self, occ_grid, add_box=False, add_enclosing=False):
        if add_box:
            self.p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            for i in range(occ_grid.shape[0]):
                for j in range(occ_grid.shape[1]):
                    for k in range(occ_grid.shape[2]):
                        if occ_grid[i, j, k] > 0:
                            self.add_box([i * 0.1 + 0.05, j * 0.1 + 0.05, k * 0.1 + 0.05], [0.05, 0.05, 0.05])
            self.p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        self.occ_grid = occ_grid
        self.size = [
            occ_grid.shape[0] * OCC_GRID_RESOLUTION,
            occ_grid.shape[1] * OCC_GRID_RESOLUTION,
            occ_grid.shape[2] * OCC_GRID_RESOLUTION,
        ]
        print(self.size)

        # add enclosing obstacles:
        if add_enclosing:
            self.add_box([-0.05, self.size[1] / 2, 1], [0.05, self.size[1] / 2 + 0.1, 1])
            self.add_box([self.size[0] + 0.05, self.size[1] / 2, 1.0], [0.05, self.size[1] / 2 + 0.1, 1])
            self.add_box([self.size[0] / 2, -0.05, 1], [self.size[0] / 2 + 0.1, 0.05, 1])
            self.add_box([self.size[0] / 2, self.size[0] + 0.05, 1], [self.size[0] / 2 + 0.1, 0.05, 1])

        if self.robot is None:
            return

        # robot
        base_x_bounds = [0, self.size[0]]
        base_y_bounds = [0, self.size[1]]
        self.robot.set_base_bounds(base_x_bounds, base_y_bounds)

        if add_box:
            # self.pb_ompl_interface_base.set_obstacles(self.obstacles)
            self.pb_ompl_interface.set_obstacles(self.obstacles)

        # ompl
        bounds = ob.RealVectorBounds(self.robot.num_dim)
        joint_bounds = self.robot.get_joint_bounds()
        print(joint_bounds)
        for i, bound in enumerate(joint_bounds):
            bounds.setLow(i, bound[0])
            bounds.setHigh(i, bound[1])
        self.pb_ompl_interface.space.setBounds(bounds)

        # bounds = ob.RealVectorBounds(2)
        # joint_bounds = self.robot.get_joint_bounds()
        # for i, bound in enumerate(joint_bounds[:2]):
        #     bounds.setLow(i, bound[0])
        #     bounds.setHigh(i, bound[1])
        # self.pb_ompl_interface.space.set_se2_bounds(bounds)

    def get_occupancy_grid(self):
        return self.occ_grid

    def get_obstacle_dict(self):
        return self.obstacle_dict

    def sample_start_goal(self, load=False):
        if load:
            print("Maze2D: loading start_goal from sg_paris.json!!!")
            with open(osp.join(ROOT_DIR, "sg_pairs.json"), "r") as f:
                self.sg_pairs = json.load(f)

            sg = random.choice(self.sg_pairs)
            self.start = sg[0]
            self.goal = sg[1]
            # self.start = [-4,-3,0,0,0,0,0]
            # self.goal = [1,0,math.radians(-90),0,0,0,0]
            # self.goal = [-2,0,0,0,0,0,0]

            self.start = [0, 0, math.radians(180), 0]
            # self.goal = [1,0,math.radians(-90),0,0,0,0]
            self.goal = [2, 2, math.radians(-90), 0]
        else:
            while True:
                start = [0] * self.robot.num_dim
                goal = [0] * self.robot.num_dim
                low_bounds = self.robot.get_joint_lower_bounds()
                low_bounds[0] = 0
                low_bounds[1] = 0
                high_bounds = self.robot.get_joint_higher_bounds()
                high_bounds[0] = self.size[0]
                high_bounds[1] = self.size[1]
                for i in range(self.robot.num_dim):
                    start[i] = random.uniform(low_bounds[i], high_bounds[i])
                    goal[i] = random.uniform(low_bounds[i], high_bounds[i])

                if self.pb_ompl_interface.is_state_valid(start) and self.pb_ompl_interface.is_state_valid(goal):
                    self.start = start
                    self.goal = goal
                    break

        print("Maze2d: start: {}".format(self.start))
        print("Maze2d: goal: {}".format(self.goal))

    def get_local_occ_grid(self, state, local_env_size=2.0):
        base_x = state[0]
        base_y = state[1]

        # base_x = round(base_x, 1)
        # base_y = round(base_y, 1)

        small_occ_grid_resolution = 0.1
        small_occ_grid_size = int(local_env_size / small_occ_grid_resolution)
        idx_x = round(base_x / small_occ_grid_resolution)
        idx_y = round(base_y / small_occ_grid_resolution)

        min_y = max(0, idx_y - small_occ_grid_size)
        max_y = min(self.occ_grid.shape[1], idx_y + small_occ_grid_size)
        min_x = max(0, idx_x - small_occ_grid_size)
        max_x = min(self.occ_grid.shape[0], idx_x + small_occ_grid_size)

        min_y_1 = 0 if min_y != 0 else small_occ_grid_size - idx_y
        max_y_1 = (
            2 * small_occ_grid_size
            if max_y != self.occ_grid.shape[1]
            else self.occ_grid.shape[1] - idx_y + small_occ_grid_size
        )
        min_x_1 = 0 if min_x != 0 else small_occ_grid_size - idx_x
        max_x_1 = (
            2 * small_occ_grid_size
            if max_x != self.occ_grid.shape[0]
            else self.occ_grid.shape[0] - idx_x + small_occ_grid_size
        )

        # print(state, idx_x, min_x, max_x, min_x_1, max_x_1)
        # print(state, idx_y, min_y, max_y, min_y_1, max_y_1)
        local_occ_grid = np.ones(
            (2 * small_occ_grid_size, 2 * small_occ_grid_size, self.occ_grid.shape[2]), dtype=np.uint8
        )
        local_occ_grid[min_x_1:max_x_1, min_y_1:max_y_1] = self.occ_grid[min_x:max_x, min_y:max_y]

        return local_occ_grid

    def clear_obstacles_outside_local_occ_grid(self, state, local_env_size=2.0, tmp_mesh_file_name="tmp.obj"):
        base_x = state[0]
        base_y = state[1]

        # base_x = round(base_x, 1)
        # base_y = round(base_y, 1)

        small_occ_grid_resolution = 0.1
        small_occ_grid_size = int(local_env_size / small_occ_grid_resolution)
        idx_x = round(base_x / small_occ_grid_resolution)
        idx_y = round(base_y / small_occ_grid_resolution)

        min_y = max(0, idx_y - small_occ_grid_size)
        max_y = min(self.occ_grid.shape[1], idx_y + small_occ_grid_size)
        min_x = max(0, idx_x - small_occ_grid_size)
        max_x = min(self.occ_grid.shape[0], idx_x + small_occ_grid_size)

        # print(state, idx_x, min_x, max_x, min_x_1, max_x_1)
        # print(state, idx_y, min_y, max_y, min_y_1, max_y_1)

        new_occ_grid = np.zeros_like(self.occ_grid)
        new_occ_grid[min_x:max_x, min_y:max_y] = self.occ_grid[min_x:max_x, min_y:max_y]
        new_occ_grid[0, :, :] = 1
        new_occ_grid[-1, :, :] = 1
        new_occ_grid[:, 0, :] = 1
        new_occ_grid[:, -1, :] = 1

        import trimesh

        new_mesh = trimesh.voxel.ops.matrix_to_marching_cubes(new_occ_grid, pitch=0.1)
        # new_mesh.show()

        os.makedirs(self._tmp_mesh_dir, exist_ok=True)
        new_mesh_path = os.path.join(self._tmp_mesh_dir, tmp_mesh_file_name)

        if os.path.exists(new_mesh_path):
            os.remove(new_mesh_path)

        with open(new_mesh_path, "w") as f:
            new_mesh.export(f, file_type="obj")

        self.clear_obstacles()
        self.load_mesh(new_mesh_path)
        self.load_occupancy_grid(new_occ_grid)

        return new_occ_grid, new_mesh_path

    # def plan_locally(self, start, goal, allowed_time=10.0, interpolate=False):
    #     self.pb_ompl_interface.clear()
    #     self.path = None
    #     self.approx_path = None

    #     local_cx, local_cy = utils.get_local_center(start, self.size)
    #     bounds = ob.RealVectorBounds(self.robot.num_dim)
    #     joint_bounds = self.robot.get_joint_bounds()
    #     for i, bound in enumerate(joint_bounds):
    #         if i == 0:
    #             bounds.setLow(i, local_cx - 2.5)
    #             bounds.setHigh(i, local_cx + 2.5)
    #         elif i == 1:
    #             bounds.setLow(i, local_cy - 2.5)
    #             bounds.setHigh(i, local_cy + 2.5)
    #         else:
    #             bounds.setLow(i, bound[0])
    #             bounds.setHigh(i, bound[1])
    #     self.pb_ompl_interface.space.setBounds(bounds)

    #     self.robot.set_state(start)
    #     res, path = self.pb_ompl_interface.plan(goal, allowed_time, interpolate)
    #     if res:
    #         self.path = path
    #     elif path is not None:
    #         self.approx_path = path
    #     return res, path

    # def construct_prm_base(self, allowed_time=5.0, clear=True):
    #     if clear:
    #         self.pb_ompl_interface_base.clear()
    #     self.pb_ompl_interface_base.construct_prm(allowed_time)

    # def plan_base(self, allowed_time=5.0, interpolate=False):
    #     # self.pb_ompl_interface_2.clear()
    #     self.path = None
    #     self.approx_path = None

    #     self.robot_base.set_state(self.start[:2])
    #     res, path = self.pb_ompl_interface_base.plan(self.goal[:2], allowed_time, interpolate)
    #     if res:
    #         self.path = path
    #     elif path is not None:
    #         self.approx_path = path
    #     return res, path

    def construct_prm(self, allowed_time=5.0, clear=True):
        if clear:
            self.pb_ompl_interface.clear()
        self.pb_ompl_interface.construct_prm(allowed_time)

    def plan(self, allowed_time=10.0, interpolate=False):
        # self.pb_ompl_interface_2.clear()
        self.path = None
        self.approx_path = None

        self.robot.set_state(self.start)
        res, path = self.pb_ompl_interface.plan(self.goal, allowed_time, interpolate)
        if res:
            self.path = path
        elif path is not None:
            self.approx_path = path
        return res, path

    def execute(self, path):
        time.sleep(1)
        for q in path:
            self.robot.set_state(q)
            self.p.stepSimulation()
            time.sleep(0.5)

    def disable_visual(self):
        self.p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

    def enable_visual(self):
        self.p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    def add_line(self, start, finish, colour):
        start.append(1.0)
        finish.append(1.0)
        length = math.sqrt((finish[1] - start[1]) ** 2 + (finish[0] - start[0]) ** 2)
        theta = math.atan2(finish[1] - start[1], finish[0] - start[0])
        r = R.from_euler("z", theta)
        q = r.as_quat()
        colBoxId = self.p.createCollisionShape(p.GEOM_BOX, halfExtents=[length / 2, 0.1, 0.1])
        visualBoxId = self.p.createVisualShape(p.GEOM_BOX, halfExtents=[length / 2, 0.1, 0.1], rgbaColor=colour)
        self.p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=colBoxId,
            baseVisualShapeIndex=visualBoxId,
            basePosition=[(finish[0] + start[0]) / 2, (finish[1] + start[1]) / 2, 0.1],
            baseOrientation=q,
        )

    def get_img(self, pos):
        view_matrix = self.p.computeViewMatrix([pos[0], pos[1], 5], [pos[0], pos[1], 0], [0, 1, 0])
        proj_matrix = self.p.computeProjectionMatrixFOV(120, 1, 0.01, 6)
        w, h, rgba, depth, mask = self.p.getCameraImage(
            width=640,
            height=640,
            projectionMatrix=proj_matrix,
            viewMatrix=view_matrix,
            flags=p.ER_NO_SEGMENTATION_MASK,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )
        return rgba

    def get_global_img(self):
        env_len, env_width = self.size[0], self.size[1]
        print(env_len, env_width)
        view_matrix = self.p.computeViewMatrix(
            [env_len / 2, env_width / 2, 50], [env_len / 2, env_width / 2, 0], [0, 1, 0]
        )

        fov = math.degrees(math.atan2(env_width / 2, 48) * 2)
        proj_matrix = self.p.computeProjectionMatrixFOV(fov, env_len / env_width, 0.01, 52)
        # proj_matrix = self.p.computeProjectionMatrix(0, self.size[0], 0, self.size[1], 0.01, 12)

        w, h, rgba, depth, mask = self.p.getCameraImage(
            width=int(env_len * 100),
            height=int(env_width * 100),
            projectionMatrix=proj_matrix,
            viewMatrix=view_matrix,
            flags=self.p.ER_NO_SEGMENTATION_MASK,
            renderer=self.p.ER_BULLET_HARDWARE_OPENGL,
        )
        return rgba

    def get_global_img_2(self):
        env_len, env_width = self.size[0], self.size[1]
        print(env_len, env_width)
        view_matrix = self.p.computeViewMatrix([0, 0, 10], [env_len / 2, env_width / 2, 0], [0, 0, 1])

        # fov = math.degrees(90)
        fov = 90
        proj_matrix = self.p.computeProjectionMatrixFOV(fov, env_len / env_width, 0.01, 52)
        # proj_matrix = self.p.computeProjectionMatrix(0, self.size[0], 0, self.size[1], 0.01, 12)

        w, h, rgba, depth, mask = self.p.getCameraImage(
            width=int(env_len * 100),
            height=int(env_width * 100),
            projectionMatrix=proj_matrix,
            viewMatrix=view_matrix,
            flags=p.ER_NO_SEGMENTATION_MASK,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )
        return rgba


if __name__ == "__main__":
    maze = RLSEnv()

    # maze.load_mesh(osp.join(CUR_DIR, "../dataset/local_env/1/env.obj"))
    with open(os.path.join(CUR_DIR, "map/occ_grid.npy"), "rb") as f:
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

    input()

    # while True:
    #     key = input()
    #     res = False
    #     while not res:
    #         random_state = [0] * maze.robot.num_dim
    #         for i in range(maze.robot.num_dim):
    #             random_state[i] = random.uniform(low[i], high[i])

    #         res = maze.pb_ompl_interface.is_state_valid(random_state)

    #     print(random_state, res)

    # maze.sample_start_goal()
    # res, path = maze.plan(interpolate=False)
    # if res:
    #     print(len(path))
    #     new_path = utils.interpolate(path)
    #     maze.execute(new_path)

    # random_state = [5.013068938601293, 8.060664273495597, 0.5680683758811989, 0.011601832020968475, 0.7839950843414254, -1.0492533437930283, 3.078759486697521, -2.1949954328823735, -0.141336182250964, -0.600275769871053, -1.6679128905619518]
    # maze.robot.set_state(random_state)

    # scaling = 1.0
    # while True:
    #     key = input()
    #     if key == "w":
    #         random_state[0] += 0.25
    #     elif key == "s":
    #         random_state[0] -= 0.25
    #     if key == "a":
    #         random_state[1] += 0.25
    #     elif key == "d":
    #         random_state[1] -= 0.25
    #     elif key == "q":
    #         scaling += 0.1
    #         maze.p.removeBody(maze.robot_id)
    #         maze.add_robot(random_state, scaling=scaling)
    #     elif key == "e":
    #         scaling -= 0.1
    #         maze.p.removeBody(maze.robot_id)
    #         maze.add_robot(random_state, scaling=scaling)

    #     maze.robot.set_state(random_state)
    #     print(random_state)
