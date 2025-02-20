import os.path as osp
import sys
import json
import time
import torch
import random
import math
import numpy as np

from nrp.planner.rrt import RRT
from nrp.planner.informed_rrt import InformedRRTStar

CUR_DIR = osp.dirname(osp.abspath(__file__))

ADD_INTERMEDIATE_STATE = False


class RRTPlanner:
    def __init__(self, optimal=False):
        self.log_dir = None
        self.log_expansion_info = False

        if not optimal:
            self.algo = RRT(self.col_checker, self.heuristic, self.sample_uniform, self.expand_fn)
            self.add_intermediate_state = False
        else:
            self.algo = InformedRRTStar(self.col_checker, self.heuristic, self.sample_uniform, self.expand_fn)
            self.add_intermediate_state = False

        random.seed(0)

    def clear(self):
        self.env = None
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

    def solve_step_time(self, env, start, goal, max_time, step_size, mesh=None):
        self.clear()
        self.algo.clear()
        self.algo.env = env
        self.env = env

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
            res.append((success, path))

            i += 1
            max_t += step_size

        return res

    def col_checker(self, v1, v2):
        # start_time = time.time()
        valid = self.env.utils.is_edge_free(self.env, v1, v2)
        # end_time = time.time()
        # print("collision check takes: {}".format(end_time - start_time))
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
            random_state = [0] * self.env.robot.num_dim
            for i in range(self.env.robot.num_dim):
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

        if self.log_expansion_info:
            orig_path = self.env.utils.interpolate([v, g])
            self.dump_expansion_information(orig_path, sl_expansion_path, g)

        # print("Extension num: {}".format(self.extend_cnt))

        return sl_expansion_path

    def dump_expansion_information(self, path, final_path, g):
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
