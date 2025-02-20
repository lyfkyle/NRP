import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import json
import time
from copy import deepcopy

from nrp.planner.rrt import RRT
from nrp.planner.informed_rrt import InformedRRTStar
from nrp.planner.reeds_shepp_path import ReedsSheppPath





class DecomposedRRTPlanner:
    def __init__(self, optimal=False):
        self.log_dir = None
        self.log_extension_info = False

        if not optimal:
            self.algo = RRT(self.col_checker, self.heuristic, self.sample_uniform, self.expand_fn)
            self.add_intermediate_state = False
        else:
            self.algo = InformedRRTStar(self.col_checker, self.heuristic, self.sample_uniform, self.expand_fn)
            self.add_intermediate_state = False

        random.seed(0)

    def clear(self):
        self.env = None
        self.base_coord = [0, 0, 0]
        self.expansion_col_cnt = 0
        self.log_dir = None
        self.col_check_time = 0
        self.col_check_success_time = 0
        self.col_check_fail_time = 0

    def solve(self, env, start, goal, allowed_time, max_samples=float("inf"), mesh=None):
        self.clear()
        self.algo.clear()
        self.algo.env = env
        self.env = env

        path = self.algo.solve(start, goal, allowed_time, max_samples)

        return path

    def solve_step_expansion(self, env, start, goal, max_extensions, step_size=50, mesh=None):
        self.clear()
        self.algo.clear()
        self.algo.env = env
        self.env = env

        res = []
        i = 0
        for max_ext in range(step_size, max_extensions + 1, step_size):
            if i == 0:
                path = self.algo.solve(start, goal, float("inf"), max_ext)
            else:
                self.algo.max_extension_num = max_ext
                path = self.algo.continue_solve()

            success = len(path) > 0
            res.append((success, path))
            i += 1

        return res

    def solve_step_time(self, env, start, goal, max_time, step_size, mesh=None, reverse=False):
        self.clear()
        self.algo.clear()
        self.env = env
        self.algo.env = env
        self.base_coord = start[:3]

        res = []
        i = 0
        max_t = step_size
        start_time = time.time()
        while max_t <= max_time + 1e-4:
            if i == 0:
                path = self.algo.solve(start, goal, max_t, float("inf"))
            else:
                self.algo.allowed_time = start_time + max_t
                path = self.algo.continue_solve()

            success = len(path) > 0
            if reverse:
                path.reverse()
            res.append((success, path))
            i += 1
            max_t += step_size

        return res

    def col_checker(self, v1, v2):
        # valid = utils.is_edge_free(self.maze, v1, v2)
        valid = self.env.utils.is_edge_free(self.env, v1, v2)
        # if valid:
        #     return utils.calc_edge_len(v1, v2)
        # else:
        #     # return float('inf')
        #     return False
        return valid

    def heuristic(self, v1, v2):
        return np.array(self.env.utils.calc_edge_len(v1, v2))

    def sample_uniform(self, v, num_samples):
        if num_samples == 0:
            return []

        samples = []
        low = self.env.robot.get_joint_lower_bounds()
        high = self.env.robot.get_joint_higher_bounds()
        for _ in range(num_samples):
            random_state = self.base_coord + [0] * (self.env.robot.num_dim - 3)
            for i in range(3, self.env.robot.num_dim):
                random_state[i] = random.uniform(low[i], high[i])
            samples.append(random_state)

        return samples

    def expand_fn(self, v, g):
        sl_expansion_path = self.sl_expansion_fn(v, g)
        return [sl_expansion_path], None, None

    def sl_expansion_fn(self, v, g):
        start_time = time.time()
        sl_expansion_path = self.expand_rrt(v, g)
        end_time = time.time()

        self.col_check_time += end_time - start_time
        if len(sl_expansion_path) <= 1:
            self.col_check_fail_time += end_time - start_time
        else:
            self.col_check_success_time += end_time - start_time

        if not np.allclose(np.array(sl_expansion_path[-1]), np.array(g)):
            self.expansion_col_cnt += 1

        if self.log_extension_info:
            orig_path = self.env.utils.interpolate([v, g])
            self.dump_extension_information(orig_path, sl_expansion_path, g)

        # print("Extension num: {}".format(self.extend_cnt))

        return sl_expansion_path

    def dump_extension_information(self, path, final_path, g):
        expansion_data = {}
        expansion_data["local_planning_target"] = g
        expansion_data["path_intended"] = path
        expansion_data["path_actual"] = final_path

        if self.log_dir is not None:
            with open(osp.join(self.log_dir, "extension_data_{}.json".format(self.algo.num_expansions + 1)), "w") as f:
                json.dump(expansion_data, f)

    def expand_rrt(self, v, g):
        if self.add_intermediate_state:
            return self.env.utils.rrt_extend_intermediate(self.env, v, g)
        else:
            return self.env.utils.rrt_extend(self.env, v, g)


# adapted from https://github.com/dawnjeanh/motionplanning/blob/master/python/hybrid_a_star.py
class AStar(object):
    def __init__(self, start, end, map):
        self._s = start
        self._e = end
        self._map = map
        self._map_w, self._map_h = map.shape
        self._openset = dict()
        self._closeset = dict()

    def distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def is_collision(self, p):
        if not (0 <= p[0] < self._map_w and 0 <= p[1] < self._map_h):
            return True

        if self._map[int(p[0]), int(p[1])] == 1:
            return True

        return False

    def neighbor_nodes(self, x):
        plist = [
            (x[0] - 1, x[1] - 1),
            (x[0] - 1, x[1]),
            (x[0] - 1, x[1] + 1),
            (x[0], x[1] + 1),
            (x[0] + 1, x[1] + 1),
            (x[0] + 1, x[1]),
            (x[0] + 1, x[1] - 1),
            (x[0], x[1] - 1),
        ]
        for p in plist:
            if not self.is_collision(p):
                yield p

    def reconstruct_path(self):
        pt = self._e
        path = []
        while pt:
            path.append(pt)
            pt = self._closeset[pt]["camefrom"]
        return path[::-1]

    def run(self):
        h = self.distance(self._s, self._e)
        self._openset[self._s] = {"g": 0, "h": h, "f": h, "camefrom": None}
        while self._openset:
            x = min(self._openset, key=lambda key: self._openset[key]["f"])
            self._closeset[x] = deepcopy(self._openset[x])
            del self._openset[x]
            if self.distance(x, self._e) < 1.0:
                if x != self._e:
                    self._closeset[self._e] = {"camefrom": x}
                return True
            for y in self.neighbor_nodes(x):
                if y in self._closeset:
                    continue
                tentative_g_score = self._closeset[x]["g"] + self.distance(x, y)
                if y not in self._openset:
                    tentative_is_better = True
                elif tentative_g_score < self._openset[y]["g"]:
                    tentative_is_better = True
                else:
                    tentative_is_better = False
                if tentative_is_better:
                    h = self.distance(y, self._e)
                    self._openset[y] = {"g": tentative_g_score, "h": h, "f": tentative_g_score + h, "camefrom": x}

        return False


# Hybrid Astar, adapted from https://github.com/dawnjeanh/motionplanning/blob/master/python/hybrid_a_star.py
class HybridAStar(object):
    def __init__(self, br, tr, map_res):
        self._tr = tr / map_res  # turning radius
        self._br = br / map_res  # base radius
        self._openset = dict()
        self._closeset = dict()

    def plan(self, start, end, map):
        self._s = start
        self._e = end
        self._map = map
        self._map_w, self._map_h = map.shape
        self._openset = dict()
        self._closeset = dict()

        start_time = time.time()
        if self.run():
            path = self.reconstruct_path()
            # plot_trajectory(self._map, path, self._br)
        else:
            path = None

        end_time = time.time()
        path_short = tuple([x[:-1][::40] + x[-1:] for x in path])  # make sure end is in the trajectory
        return path_short, end_time - start_time

    def distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def neighbors(self, p):
        step = 5.0
        paths = [
            ["l", step / self._tr],
            ["s", step],
            ["r", step / self._tr],
            ["l", -step / self._tr],
            ["s", -step],
            ["r", -step / self._tr],
        ]
        for path in paths:
            xs, ys, yaws = ReedsSheppPath.gen_path(p, [path], r=self._tr, section=False)
            if not self.is_collision_rs_car(xs, ys, yaws):
                yield (round(xs[-1], 2), round(ys[-1], 2), round(yaws[-1], 2)), path, [xs, ys, yaws]

    def is_collision_rs_car(self, xs, ys, yaws):
        for pt in zip(xs, ys, yaws):
            # Check if the robot position is within the grid boundaries
            if not (self._br <= pt[0] < self._map_w - self._br and self._br <= pt[1] < self._map_h - self._br):
                return True

            # Check if the robot's circular boundary collides with obstacles
            for i in range(-int(self._br) - 1, int(self._br) + 1):
                for j in range(-int(self._br) - 1, int(self._br) + 1):
                    if i**2 + j**2 <= self._br and self._map[int(pt[0] + i), int(pt[1] + j)] == 1:
                        return True
        return False

    def h_cost(self, s):
        plan = AStar((s[0], s[1]), (self._e[0], self._e[1]), self._map)
        if plan.run():
            path = plan.reconstruct_path()
        d = 0
        for i in range(len(path) - 1):
            d += self.distance(path[i], path[i + 1])
        return d

    def h_cost_simple(self, s):
        return np.sqrt((s[0] - self._e[0]) ** 2 + (s[1] - self._e[1]) ** 2)

    def reconstruct_path(self):
        waypoint = []
        xs = []
        ys = []
        yaws = []
        pt = self._e
        while pt != self._s:
            waypoint.append(pt)
            pt = self._closeset[pt]["camefrom"]
        for pt in waypoint[::-1]:
            x, y, yaw = self._closeset[pt]["path"][1]
            xs += x
            ys += y
            yaws += yaw
        return xs, ys, yaws

    def run(self):
        # d = self.h_cost(self._s)
        d = self.h_cost_simple(self._s)

        self._openset[self._s] = {"g": 0, "h": d, "f": d, "camefrom": None, "path": []}
        # i = 0
        while self._openset:
            # print(i, len(self._openset))
            # i += 1
            x = min(self._openset, key=lambda key: self._openset[key]["f"])
            self._closeset[x] = deepcopy(self._openset[x])
            del self._openset[x]
            rspath = ReedsSheppPath(x, self._e, self._tr)
            rspath.calc_paths()
            path, _ = rspath.get_shortest_path()
            xs, ys, yaws = ReedsSheppPath.gen_path(x, path, self._tr, section=False)
            if len(xs) > 0 and not self.is_collision_rs_car(xs, ys, yaws):
                self._closeset[self._e] = {"camefrom": x, "path": [path, [xs, ys, yaws]]}
                return True
            for y, path, line in self.neighbors(x):
                if y in self._closeset:
                    continue
                tentative_g_score = self._closeset[x]["g"] + (
                    abs(path[1]) if path[0] == "s" else abs(path[1]) * self._tr
                )
                if y not in self._openset:
                    tentative_is_better = True
                elif tentative_g_score < self._openset[y]["g"]:
                    tentative_is_better = True
                else:
                    tentative_is_better = False
                if tentative_is_better:
                    # d = self.h_cost(y)
                    d = self.h_cost_simple(y)
                    self._openset[y] = {
                        "g": tentative_g_score,
                        "h": d,
                        "f": tentative_g_score + d,
                        "camefrom": x,
                        "path": [path, line],
                    }

        start_goal = ([self._s[0], self._e[0]], [self._s[0], self._e[0]], [self._s[0], self._e[0]])
        plot_trajectory(self._map, start_goal, self._br)
        return False


def plot_trajectory(map, path, vehicle_length):
    # Plot the occupancy grid
    plt.imshow(1 - map, cmap="gray", origin="upper")

    # Plot the path
    path_array = np.transpose(np.array(path))
    # ax.plot(path_array[::10, 1], path_array[::10, 0], marker='o', color='blue', markersize=0.1, label='Path')

    # Plot the vehicle at each step
    for state in path_array[::10]:
        x, y, yaw = state
        length = vehicle_length

        # Compute arrow components
        arrow_dx = length * np.sin(yaw)
        arrow_dy = length * np.cos(yaw)

        # Plot arrow at vehicle position
        # ax.quiver(y, x, arrow_dx, arrow_dy, angles='xy', scale_units='xy', scale=2, color='red', width=0.005)
        plt.arrow(y, x, arrow_dx, arrow_dy, length_includes_head=True, head_width=0.5, head_length=0.5, color="blue")

    plt.legend()
    plt.title("Hybrid A* Path and Trajectory")
    plt.xlabel("Column Index")
    plt.ylabel("Row Index")
    # plt.show()
    plt.savefig("/home/kma/whole-body-motion-planning/hybrid_astar_res.png", bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    # Example usage:
    # occupancy_grid = np.array([[0, 0, 0, 0, 0],
    #                             [0, 0, 0, 0, 0],
    #                             [0, 1, 0, 0, 0],
    #                             [0, 1, 0, 0, 0],
    #                             [0, 0, 0, 0, 0]])
    # start_state = (1, 1, math.pi/2)  # [x, y, yaw]
    # goal_state = (1, 3, math.pi/2)   # [x, y, yaw]

    occupancy_grid = np.zeros([60, 40])
    occupancy_grid[30, 10:20] = 1

    start_state = (10.1, 10.1, math.pi / 2)
    goal_state = (50.1, 30.1, -math.pi / 2)
    base_radius = 0.3
    turning_radius = 0.75
    map_resolution = 0.1

    hybrid_astar_planner = HybridAStar(base_radius, turning_radius, map_resolution)
    path, plan_time = hybrid_astar_planner.plan(start_state, goal_state, occupancy_grid)
    if path:
        print("Path:", path, "time:", plan_time)

        plot_trajectory(occupancy_grid, path, base_radius)

        from env.maze import Maze

        maze = Maze(gui=False)
        res_astar = []
        for i in range(len(path[0])):
            res_astar.append([path[0][i] / 10, path[1][i] / 10, path[2][i]] + list(maze.robot.rest_position))

        print(res_astar)

    else:
        path = ([start_state[0], goal_state[0]], [start_state[0], goal_state[0]], [start_state[0], goal_state[0]])
        plot_trajectory(occupancy_grid, path, base_radius)
