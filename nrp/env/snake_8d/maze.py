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
    sys.path.insert(0, join(dirname(abspath(__file__)), '../../third_party/ompl/py-bindings'))
    print(sys.path)
    from ompl import util as ou
    from ompl import base as ob
    from ompl import geometric as og

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../../'))

from .my_planar_robot import MyPlanarRobot, MyPlanarRobotBase
from . import pb_ompl
from . import utils

# from config import ROOT_DI

# sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
CUR_DIR = osp.dirname(osp.abspath(__file__))
ROOT_DIR = osp.join(osp.dirname(osp.abspath(__file__)), "../../")
DATASET_DIR = osp.join(ROOT_DIR, "dataset")
LOCAL_OCC_GRID_SIZE = 4

# -------------- Settings ----------------
OCC_GRID_RESOLUTION = 0.1

class Snake8DEnv():
    EMPTY = 0
    GAP_ONLY = 1
    BOX_ONLY = 2
    GAP_AND_BOX = 3

    def __init__(self, gui=True):
        self.name = "snake_8d"
        self.utils = utils  # this is weird, but works

        self.obstacles = []

        if gui:
            self.p = bc.BulletClient(connection_mode=p.GUI)
        else:
            self.p = bc.BulletClient(connection_mode=p.DIRECT)
        self.p.setGravity(0, 0, -9.8)
        self.p.setTimeStep(1./240.)
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # load floor
        floor = os.path.join(pybullet_data.getDataPath(), "mjcf/ground_plane.xml")
        self.p.loadMJCF(floor)

        # load robot
        robot_model_path = osp.join(ROOT_DIR, "robot_model/snake_robot_8d.xacro")
        # robot_id = self.p.loadURDF(robot_model_path, (-1, -1, 0), globalScaling=0.4)
        robot_id = self.p.loadURDF(robot_model_path, (-1, -1, 0), globalScaling=1)
        robot = MyPlanarRobot(robot_id, self.p)
        self.robot = robot
        self.robot_base = MyPlanarRobotBase(robot_id, self.p)

        # set up pb_ompl
        self.pb_ompl_interface_base = pb_ompl.PbOMPLPRM(self.robot_base, self.obstacles, self.p)
        self.pb_ompl_interface = pb_ompl.PbOMPLPRM(self.robot, self.obstacles, self.p)

        # internal attributes
        self.goal_robot_id = None
        self.path = None
        self.approx_path = None
        self.sg_pairs = None

        self.obstacle_dict = {}

        # tmp mesh dir
        now = datetime.datetime.now()
        date_time = now.strftime("%m-%d-%Y-%H-%M-%S")
        self._tmp_mesh_dir = osp.join(CUR_DIR, "{}".format(date_time))

    def clear_obstacles(self):
        for obstacle in self.obstacles:
            self.p.removeBody(obstacle)
        self.occ_grid = None
        self.obstacles = []
        self.obstacle_dict = {}
        self.pb_ompl_interface.set_obstacles(self.obstacles)
        self.pb_ompl_interface_base.set_obstacles(self.obstacles)

    def add_box(self, box_pos, half_box_size):
        colBoxId = self.p.createCollisionShape(p.GEOM_BOX, halfExtents=half_box_size)
        box_id = self.p.createMultiBody(baseMass=0, baseCollisionShapeIndex=colBoxId, basePosition=box_pos)

        self.obstacles.append(box_id)

        return box_id

    def load_mesh(self, mesh_file):
        collision_id = self.p.createCollisionShape(p.GEOM_MESH, fileName=mesh_file, flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
        mesh_body_id = self.p.createMultiBody(baseCollisionShapeIndex=collision_id, basePosition=[0.05, 0.05, 0.05])
        self.obstacles.append(mesh_body_id)

        if self.robot is not None:
            self.pb_ompl_interface_base.set_obstacles(self.obstacles)
            self.pb_ompl_interface.set_obstacles(self.obstacles)

    def load_occupancy_grid(self, occ_grid, add_enclosing=False, add_box=False):
        print("Loading occ_grid")
        if add_box:
            for i in range(occ_grid.shape[0]):
                for j in range(occ_grid.shape[1]):
                    if occ_grid[i, j] > 0:
                        self.add_box([i * 0.1 + 0.05, j * 0.1 + 0.05, 0.25], [0.05, 0.05, 0.25])

        self.size = [occ_grid.shape[0] * OCC_GRID_RESOLUTION, occ_grid.shape[1] * OCC_GRID_RESOLUTION]
        self.occ_grid = occ_grid

        # add enclosing obstacles:
        if add_enclosing:
            self.add_box([-0.05, self.size[1] // 2, 0.25], [0.05, self.size[1] // 2 + 0.1, 0.25])
            self.add_box([self.size[0] + 0.05, self.size[1] // 2, 0.25], [0.05, self.size[1] // 2 + 0.1, 0.25])
            self.add_box([self.size[0] // 2, -0.05, 0.25], [self.size[0] // 2 + 0.1, 0.05, 0.25])
            self.add_box([self.size[0] // 2, self.size[0] + 0.05, 0.25], [self.size[0] // 2 + 0.1, 0.05, 0.25])

        # robot
        base_x_bounds = [0, self.size[0]]
        base_y_bounds = [0, self.size[1]]
        self.robot.set_base_bounds(base_x_bounds, base_y_bounds)
        self.robot_base.set_base_bounds(base_x_bounds, base_y_bounds)

        if add_box:
            self.pb_ompl_interface_base.set_obstacles(self.obstacles)
            self.pb_ompl_interface.set_obstacles(self.obstacles)

        # ompl
        bounds = ob.RealVectorBounds(self.robot_base.num_dim)
        joint_bounds = self.robot_base.get_joint_bounds()
        for i, bound in enumerate(joint_bounds):
            bounds.setLow(i, bound[0])
            bounds.setHigh(i, bound[1])
        self.pb_ompl_interface_base.space.setBounds(bounds)
        bounds = ob.RealVectorBounds(self.robot.num_dim)
        joint_bounds = self.robot.get_joint_bounds()
        for i, bound in enumerate(joint_bounds):
            bounds.setLow(i, bound[0])
            bounds.setHigh(i, bound[1])
        self.pb_ompl_interface.space.setBounds(bounds)

    def get_occupancy_grid(self):
        return self.occ_grid

    def get_obstacle_dict(self):
        return self.obstacle_dict

    def sample_start_goal(self, load = False):
        if load:
            print("Maze2D: loading start_goal from sg_paris.json!!!")
            with open(osp.join(ROOT_DIR, "sg_pairs.json"), 'r') as f:
                self.sg_pairs = json.load(f)

            sg = random.choice(self.sg_pairs)
            self.start = sg[0]
            self.goal = sg[1]
            # self.start = [-4,-3,0,0,0,0,0]
            # self.goal = [1,0,math.radians(-90),0,0,0,0]
            # self.goal = [-2,0,0,0,0,0,0]

            self.start = [0, 0, math.radians(180),0]
            # self.goal = [1,0,math.radians(-90),0,0,0,0]
            self.goal = [2,2, math.radians(-90),0]
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

    def get_local_occ_grid(self, state):
        base_x = state[0]
        base_y = state[1]

        # base_x = round(base_x, 1)
        # base_y = round(base_y, 1)

        small_occ_grid_resolution = 0.1
        small_occ_grid_size = int(LOCAL_OCC_GRID_SIZE / small_occ_grid_resolution) // 2
        idx_x = round(base_x / small_occ_grid_resolution)
        idx_y = round(base_y / small_occ_grid_resolution)

        min_y = max(0, idx_y - small_occ_grid_size)
        max_y = min(self.occ_grid.shape[1], idx_y + small_occ_grid_size)
        min_x = max(0, idx_x - small_occ_grid_size)
        max_x = min(self.occ_grid.shape[0], idx_x + small_occ_grid_size)

        min_y_1 = 0 if min_y != 0 else small_occ_grid_size - idx_y
        max_y_1 = 2 * small_occ_grid_size if max_y != self.occ_grid.shape[1] else self.occ_grid.shape[1] - idx_y + small_occ_grid_size
        min_x_1 = 0 if min_x != 0 else small_occ_grid_size - idx_x
        max_x_1 = 2 * small_occ_grid_size if max_x != self.occ_grid.shape[0] else self.occ_grid.shape[0] - idx_x + small_occ_grid_size

        # print(state, idx_x, min_x, max_x, min_x_1, max_x_1)
        # print(state, idx_y, min_y, max_y, min_y_1, max_y_1)

        local_occ_grid = np.ones((2*small_occ_grid_size, 2*small_occ_grid_size), dtype=np.uint8)
        local_occ_grid[min_x_1:max_x_1, min_y_1:max_y_1] = self.occ_grid[min_x:max_x, min_y:max_y]

        return local_occ_grid

    def construct_prm_base(self, allowed_time=5.0, clear=True):
        if clear:
            self.pb_ompl_interface_base.clear()
        self.pb_ompl_interface_base.construct_prm(allowed_time)

    def plan_base(self, allowed_time=5.0, interpolate=False):
        # self.pb_ompl_interface_2.clear()
        self.path = None
        self.approx_path = None

        self.robot_base.set_state(self.start[:2])
        res, path = self.pb_ompl_interface_base.plan(self.goal[:2], allowed_time, interpolate)
        if res:
            self.path = path
        elif path is not None:
            self.approx_path = path
        return res, path

    def construct_prm(self, allowed_time=5.0, clear=True):
        if clear:
            self.pb_ompl_interface.clear()
        self.pb_ompl_interface.construct_prm(allowed_time)

    def plan(self, allowed_time=2.0, interpolate=False):
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
        for q in path:
            self.robot.set_state(q)
            self.p.stepSimulation()
            time.sleep(0.01)

    def clear_obstacles_outside_local_occ_grid(self, state, tmp_mesh_file_name="tmp.obj"):
        base_x = state[0]
        base_y = state[1]

        # base_x = round(base_x, 1)
        # base_y = round(base_y, 1)

        small_occ_grid_resolution = 0.1
        small_occ_grid_size = int(LOCAL_OCC_GRID_SIZE / small_occ_grid_resolution) // 2
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
        new_occ_grid[0, :] = 1
        new_occ_grid[-1, :] = 1
        new_occ_grid[:, 0] = 1
        new_occ_grid[:, -1] = 1

        import trimesh
        new_occ_grid_tmp = np.expand_dims(new_occ_grid, -1)
        new_mesh = trimesh.voxel.ops.matrix_to_marching_cubes(new_occ_grid_tmp, pitch=0.1)
        # new_mesh.show()

        os.makedirs(self._tmp_mesh_dir, exist_ok=True)
        new_mesh_path = os.path.join(self._tmp_mesh_dir, tmp_mesh_file_name)

        if os.path.exists(new_mesh_path):
            os.remove(new_mesh_path)

        with open(new_mesh_path, 'w') as f:
            new_mesh.export(f, file_type='obj')

        self.clear_obstacles()
        self.load_mesh(new_mesh_path)
        self.load_occupancy_grid(new_occ_grid)

        return new_occ_grid, new_mesh_path

    def get_link_positions(self, robot_state):
        return self.utils.get_link_positions(robot_state)

if __name__ == '__main__':
    maze = Maze()

    mesh = osp.join(CUR_DIR, "../dataset/gibson/train/Allensville/env_small.obj")
    maze.load_mesh(mesh)

    occ_grid = np.loadtxt(osp.join(CUR_DIR, "../dataset/gibson/train/Allensville/occ_grid_small.txt"))
    maze.load_occupancy_grid(occ_grid)

    # low = maze.robot.get_joint_lower_bounds()
    # high = maze.robot.get_joint_higher_bounds()

    # res = False
    # while not res:
    #     random_state = [0] * maze.robot.num_dim
    #     for i in range(maze.robot.num_dim):
    #         random_state[i] = random.uniform(low[i], high[i])

    #     res = maze.pb_ompl_interface_2.is_state_valid(random_state)

    low = maze.robot_base.get_joint_lower_bounds()
    high = maze.robot_base.get_joint_higher_bounds()

    # res = False
    # while not res:
    #     random_state = [0] * maze.robot_base.num_dim
    #     for i in range(maze.robot_base.num_dim):
    #         random_state[i] = random.uniform(low[i], high[i])

    random_state = [0] * maze.robot_base.num_dim
    random_state[:2] = [6.640673453117626, 3.105043319628856]
    res = maze.pb_ompl_interface.is_state_valid(random_state)

    print(random_state, res)
    input()