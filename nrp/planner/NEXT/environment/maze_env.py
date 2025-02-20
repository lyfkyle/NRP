import numpy as np
import os.path as osp
from pathlib import Path
import json
import random
import networkx as nx
import math

from NEXT.algorithm import RRT_EPS
from .env_config import LIMITS, STICK_LENGTH

from nrp.env.fetch_11d.maze import Fetch11DEnv
from nrp.env.snake_8d.maze import Snake8DEnv
import utils

CUR_DIR = osp.dirname(osp.abspath(__file__))


class MazeEnv:
    """
    Interface class for maze environment
    """

    def __init__(self, dim):
        print("Initializing environment...")
        self.dim = dim
        self.collision_check_count = 0

        # load map from file
        map_file = "maze_files/mazes_15_%d_3000.npz" % dim
        print("loading mazes from %s" % map_file)
        with np.load(map_file) as f:
            self.maps = f["maps"]
            self.init_states = f["init_states"]
            self.goal_states = f["goal_states"]

        self.size = self.maps.shape[0]
        self.width = self.maps.shape[1]
        self.order = list(range(self.size))
        self.episode_i = 0

    def init_new_problem(self, index=None):
        """
        Initialize a new planning problem
        """
        if index is None:
            index = self.episode_i

        self.map = self.maps[self.order[index]]
        self.init_state = self.init_states[self.order[index]]
        self.goal_state = self.goal_states[self.order[index]]
        self.episode_i += 1
        self.collision_check_count = 0

        return self.get_problem()

    def get_problem(self):
        problem = {
            "map": self.map,
            "init_state": self.init_state,
            "goal_state": self.goal_state,
        }
        return problem

    def uniform_sample(self):
        """
        Uniformlly sample in the configuration space
        """
        sample = np.random.uniform(-LIMITS[: self.dim], LIMITS[: self.dim])
        return sample

    def distance(self, from_state, to_state):
        """
        Distance metric
        """
        diff = np.abs(to_state - from_state)
        if diff.ndim == 1:
            diff = diff.reshape(1, -1)

        if self.dim >= 3:
            diff[:, 2] = np.min(
                (diff[:, 2], np.abs(diff[:, 2] - 2 * LIMITS[2])), axis=0
            )
            assert (np.abs(diff[:, 2]) <= LIMITS[2]).all()

        return np.sqrt(np.sum(diff**2, axis=-1))

    def interpolate(self, from_state, to_state, ratio):
        diff = to_state - from_state

        if self.dim >= 3:
            if np.abs(diff[2]) > LIMITS[2]:
                if diff[2] > 0:
                    diff[2] -= 2 * LIMITS[2]
                else:
                    diff[2] += 2 * LIMITS[2]
            assert np.abs(diff[2]) <= LIMITS[2]

        new_state = from_state + diff * ratio

        if self.dim >= 3:
            if np.abs(new_state[2]) > LIMITS[2]:
                if new_state[2] > 0:
                    new_state[2] -= 2 * LIMITS[2]
                else:
                    new_state[2] += 2 * LIMITS[2]
            assert np.abs(new_state[2]) <= LIMITS[2]

        return new_state

    def in_goal_region(self, state):
        """
        Return whether a state(configuration) is in the goal region
        """
        return self.distance(state, self.goal_state) < RRT_EPS and self._state_fp(state)

    def step(self, state, action=None, new_state=None, check_collision=True):
        """
        Collision detection module
        """
        # must specify either action or new_state
        if action is not None:
            new_state = state + action

        new_state[:2] = new_state[:2].clip(-LIMITS[:-1], LIMITS[:-1])
        if self.dim >= 3:
            if np.abs(new_state[2]) > LIMITS[2]:
                if new_state[2] > 0:
                    new_state[2] -= 2 * LIMITS[2]
                else:
                    new_state[2] += 2 * LIMITS[2]
            assert np.abs(new_state[2]) <= LIMITS[2]

        action = new_state - state

        if not check_collision:
            return new_state, action

        done = False
        no_collision = self._edge_fp(state, new_state)
        if no_collision and self.in_goal_region(new_state):
            done = True

        return new_state, action, no_collision, done

    # =====================internal collision check module=======================

    # transform a state into a discretized grid coordinate
    def _transform(self, state, w=15):
        coord = ((np.array(state)[:2].flatten() + 1.0) * w / 2.0).astype(int)
        coord[coord > w - 1] = w - 1
        return coord

    def _end_points(coord=None, l=None, center=None, theta=None, a=None, b=None):
        if theta is None:
            theta = coord[2] / LIMITS[2] * np.pi
        orient = np.array([np.cos(theta), np.sin(theta)])
        if l is None:
            l = STICK_LENGTH

        if a is None and b is None:
            if center is None:
                center = np.array(coord[:2])
            a = center - l / 2.0 * orient
            b = center + l / 2.0 * orient
        else:
            if a is not None:
                b = a + l * orient
            if b is not None:
                a = b - l * orient

        return a, b

    def _valid_state(self, state):
        return (state >= -LIMITS[: state.size]).all() and (
            state <= LIMITS[: state.size]
        ).all()

    def _point_in_free_space(self, state):
        assert state.size == 2
        if not self._valid_state(state):
            return False

        return self.map[tuple(self._transform(state, self.width))] == 0

    def _stick_in_free_space(self, state):
        assert state.size == 3

        if not self._valid_state(state):
            return False

        a, b = MazeEnv._end_points(state)
        if not self._point_in_free_space(a) or not self._point_in_free_space(b):
            return False

        return self._iterative_check_segment(a, b)

    def _state_fp(self, state):
        assert state.size == 2 or state.size == 3 or state.size == 5
        self.collision_check_count += 1

        if state.size == 2:
            return self._point_in_free_space(state)
        elif state.size == 3:
            return self._stick_in_free_space(state)

    def _iterative_check_segment(self, left, right):
        assert left.size == 2 and right.size == 2

        left_coord = np.array(self._transform(left, self.width), dtype=int)
        right_coord = np.array(self._transform(right, self.width), dtype=int)
        if np.sum(np.abs(left_coord - right_coord)) > 1:
            mid = (left + right) / 2.0
            if not self._state_fp(mid):
                return False
            return self._iterative_check_segment(
                left, mid
            ) and self._iterative_check_segment(mid, right)

        return True

    def _edge_fp(self, state, new_state):
        assert state.size == new_state.size

        if not self._valid_state(state) or not self._valid_state(new_state):
            return False
        if not self._state_fp(state) or not self._state_fp(new_state):
            return False

        if state.size == 2:
            return self._iterative_check_segment(state, new_state)
        else:

            disp = new_state - state
            if np.abs(disp[2]) > LIMITS[2]:
                if disp[2] > 0:
                    disp[2] -= 2 * LIMITS[2]
                else:
                    disp[2] += 2 * LIMITS[2]
            assert np.abs(disp[2]) <= LIMITS[2]

            d = self.distance(state, new_state)
            K = int(d / 0.015)
            for k in range(1, K):
                c = state + k * 1.0 / K * disp

                if state.size == 3:
                    ca, cb = MazeEnv._end_points(c)
                    if not self._edge_fp(ca, cb):
                        return False

            return True


class MyMazeEnv(MazeEnv):
    RRT_EPS = 2.0
    LOCAL_ENV_SIZE = 2.0

    def __init__(self, env_name, dim,data_dir, test=False):
        print("Initializing environment...")
        self.dim = dim
        self.collision_check_count = 0

        if env_name == "fetch_11d":
            self._maze = Fetch11DEnv(gui=False)
        elif env_name == "snake_8d":
            self._maze = Snake8DEnv(gui=False)

        maze_dirs = []
        for path in Path(data_dir).rglob("env.obj"):
            maze_dirs.append(path.parent)
        if not test:
            self.maps = maze_dirs
        else:
            self.maps = []
            for i in range(len(maze_dirs)):
                self.maps.append(osp.join(data_dir, "{}".format(i)))

        # load map from file
        # map_file = 'maze_files/mazes_15_%d_3000.npz' % dim
        # print("loading mazes from %s" % map_file)
        # with np.load(map_file) as f:
        #     self.maps = f['maps']
        #     self.init_states = f['init_states']
        #     self.goal_states = f['goal_states']

        # self.size = self.maps.shape[0]
        self.size = len(self.maps)
        print(self.size)
        # self.width = self.maps.shape[1]
        self.order = list(range(self.size))
        self.episode_i = 0

        self.low = [
            0,
            0,
            math.radians(-180),
            math.radians(-180),
            math.radians(-180),
            math.radians(-180),
            math.radians(-180),
            math.radians(-180),
        ]
        self.high = [
            10,
            10,
            math.radians(180),
            math.radians(180),
            math.radians(180),
            math.radians(180),
            math.radians(180),
            math.radians(180),
        ]

    def init_new_problem(self, index=None, use_start_goal=False):
        """
        Initialize a new planning problem
        """
        if index is None:
            index = self.episode_i

        maze_dir = self.maps[self.order[index]]
        print("Init new problems on ", maze_dir, index)

        occ_grid = np.loadtxt(osp.join(maze_dir, "occ_grid_small.txt")).astype(np.uint8)
        mesh = osp.join(maze_dir, "env_small.obj")

        self._maze.clear_obstacles()
        self._maze.load_mesh(mesh)
        self._maze.load_occupancy_grid(occ_grid)

        base_x_bounds = [0, occ_grid.shape[0]]
        base_y_bounds = [0, occ_grid.shape[1]]
        print(occ_grid.shape)
        self._maze.robot.set_base_bounds(base_x_bounds, base_y_bounds)

        if not use_start_goal:
            G = nx.read_graphml(osp.join(maze_dir, "dense_g_small.graphml.xml"))
            start, goal, expert_path = self.sample_problems(G)
        else:
            with open(osp.join(maze_dir, "start_goal.json")) as f:
                start, goal = json.load(f)
                expert_path = None

        # expert_path_2 = utils.interpolate(expert_path)
        # utils.visualize_nodes_global(occ_grid_gt, expert_path_2, start, goal, show=False, save=True, file_name="viz.png")
        # for i in range(1, len(expert_path)):
        #     v1 = expert_path[i - 1]
        #     v2 = expert_path[i]
        #     assert utils.is_edge_free(self._maze, v1, v2)

        self.map = occ_grid
        self.map_orig = occ_grid
        self.init_state = np.array(start)
        self.goal_state = np.array(goal)
        self.episode_i += 1

        self.expert_path = expert_path

        if self.episode_i >= self.size:
            self.episode_i = 0
            # random.shuffle(self.order)

        self.collision_check_count = 0

        # self.low = self._maze.robot.get_joint_lower_bounds()
        # self.high = self._maze.robot.get_joint_higher_bounds()

        return self.get_problem()

    def uniform_sample(self):
        """
        Uniformlly sample in the configuration space
        """
        # random_state = [0] * maze.robot.num_dim
        # for i in range(maze.robot.num_dim):
        #     random_state[i] = random.uniform(low[i], high[i])
        sample = np.random.uniform(self.low[: self.dim], self.high[: self.dim])
        return sample

    def distance(self, from_state, to_state):
        """
        Distance metric
        """
        to_state = np.maximum(to_state, np.array(self.low))
        to_state = np.minimum(to_state, np.array(self.high))
        diff = np.abs(to_state - from_state)
        return np.sqrt(np.sum(diff**2, axis=-1))

    def interpolate(self, from_state, to_state, ratio):
        diff = to_state - from_state
        new_state = from_state + diff * ratio

        new_state = np.maximum(new_state, np.array(self.low))
        new_state = np.minimum(new_state, np.array(self.high))
        return new_state

    def in_goal_region(self, state):
        """
        Return whether a state(configuration) is in the goal region
        """
        return self.distance(state, self.goal_state) < 0.1 and self._state_fp(state)

    def step(self, state, action=None, new_state=None, check_collision=True):
        """
        Collision detection module
        """
        # must specify either action or new_state
        if action is not None:
            new_state = state + action

        new_state = np.maximum(new_state, np.array(self.low))
        new_state = np.minimum(new_state, np.array(self.high))

        action = new_state - state

        if not check_collision:
            return new_state, action

        done = False
        no_collision = self._edge_fp(state, new_state)
        if no_collision and self.in_goal_region(new_state):
            done = True

        return new_state, action, no_collision, done

        # RRT style extension
        # done = False
        # # no_collision = self._edge_fp(state, new_state)
        # res_new_state = np.array(utils.rrt_extend(self._maze, state, new_state))
        # # print("env.step", state, new_state, res_new_state)
        # if not np.allclose(res_new_state, state):
        #     no_collision = True
        #     action = new_state - state
        # else:
        #     no_collision = False
        # if no_collision and self.in_goal_region(res_new_state):
        #     done = True

        # return res_new_state, action, no_collision, done

    def _edge_fp(self, state, new_state):
        return utils.is_edge_free(self._maze, state, new_state)

    def _state_fp(self, state):
        return self._maze.pb_ompl_interface.is_state_valid(state.tolist())

    def sample_problems(self, G):
        # path = dict(nx.all_pairs_shortest_path(G))
        free_nodes = [n for n in G.nodes() if not G.nodes[n]["col"]]

        max_trial = 100
        i = 0
        s_name = None
        g_name = None
        while i < max_trial:
            s_name = random.choice(free_nodes)
            start_pos = utils.node_to_numpy(G, s_name)

            g_name = random.choice(free_nodes)
            goal_pos = utils.node_to_numpy(G, g_name)

            try:
                path = nx.shortest_path(G, source=s_name, target=g_name)
                # path_pos = [utils.node_to_numpy(G, n) for n in path]
            except:
                continue

            p = [utils.node_to_numpy(G, n).tolist() for n in path]
            # for x in p:
            #     x[0] += 2
            #     x[1] += 2

            if len(path) > 4 or utils.cal_path_len_base(p) > 5:
                break

            i += 1

        return start_pos, goal_pos, p
