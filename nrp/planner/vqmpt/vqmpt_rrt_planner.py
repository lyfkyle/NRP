import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))

import numpy as np
import argparse
import matplotlib.pyplot as plt
import math
import random
import json
import time
import torch
from torch.distributions import MultivariateNormal
from scipy import stats

from planner.rrt import RRT
from planner.rrt_connect import RRTConnect
from planner.informed_rrt import InformedRRTStar

from vqmpt.utils import get_search_dist, get_inference_models
# from env.snake_8d.utils import visualize_distributions
from env.fetch_11d.utils import visualize_distributions

DEBUG = True
def dbg_print(x):
    if DEBUG:
        print(x)

class RRT_VQMPT():
    def __init__(self, model_path, optimal=False, dim=8, n_e=2048, device=torch.device('cuda')):
        self.log_dir = None
        self.log_extension_info = False
        self.dim = dim
        self.num_keys = n_e
        self.device = device
        decoder_model_folder = osp.join(model_path, "stage_1")
        ar_model_folder = osp.join(model_path, "stage_2")

        self.quantizer_model, self.decoder_model, self.context_env_encoder, self.ar_model = \
            get_inference_models(decoder_model_folder, ar_model_folder, device, n_e=n_e, e_dim=dim)

        if not optimal:
            self.algo = RRT(self.col_checker, self.heuristic, self.sample_biased, self.extend_fn)
            self.add_intermediate_state = False
        else:
            self.algo = InformedRRTStar(self.col_checker, self.heuristic, self.sample_biased, self.extend_fn)
            self.add_intermediate_state = False

        random.seed(0)

    def clear(self):
        self.env = None
        self.extend_cnt = 0
        self.extension_col_cnt = 0
        self.log_dir = None
        self.col_check_time = 0
        self.col_check_success_time = 0
        self.col_check_fail_time = 0

    def solve(self, maze, start, goal, allowed_time, max_samples=float('inf'), mesh=None):
        self.clear()
        self.algo.clear()
        self.env = maze
        self.algo.env = maze

        path = self.algo.solve(start, goal, allowed_time, max_samples)

        return path

    def solve_step_extension(self, maze, start, goal, max_extensions, step_size=50, mesh=None):
        self.clear()
        self.algo.clear()
        self.env = maze
        self.algo.env = maze
        self.q_min = np.array(self.env.robot.get_joint_lower_bounds())
        self.q_max = np.array(self.env.robot.get_joint_higher_bounds())
        self.global_occ_grid = self.env.get_occupancy_grid()
        path_norm = (np.array([start, goal]) - self.q_min)/(self.q_max - self.q_min)

        search_dist_mu, search_dist_sigma = get_search_dist(
            path_norm,
            self.global_occ_grid,
            self.context_env_encoder,
            self.decoder_model,
            self.ar_model,
            self.quantizer_model,
            self.num_keys,
            device=self.device,
        )
        if search_dist_mu is not None:
            self.seq_num = search_dist_mu.shape[0]
            self.X = MultivariateNormal(search_dist_mu, search_dist_sigma)
            # visualize_distributions(self.global_occ_grid, search_dist_mu, search_dist_sigma, self.q_min, self.q_max, start, goal, show=False, save=True, file_name="vqmpt_distributions.png")
        else:
            self.X = None
            self.U = stats.uniform(np.zeros_like(self.q_min), np.ones_like(self.q_max))

        res = []
        i = 0
        for max_ext in range(step_size, max_extensions + 1, step_size):
            if i == 0:
                path = self.algo.solve(start, goal, float('inf'), max_ext)
            else:
                self.algo.max_extension_num = max_ext
                path = self.algo.continue_solve()

            success = len(path) > 0
            res.append((success, path))
            i += 1

        return res

    def solve_step_time(self, maze, start, goal, max_time, step_size, mesh=None):
        self.clear()
        self.algo.clear()
        self.env = maze
        self.algo.env = maze
        self.q_min = np.array(self.env.robot.get_joint_lower_bounds())
        self.q_max = np.array(self.env.robot.get_joint_higher_bounds())
        self.global_occ_grid = self.env.get_occupancy_grid()
        path_norm = (np.array([start, goal]) - self.q_min)/(self.q_max - self.q_min)

        get_dist_start_time = time.time()
        search_dist_mu, search_dist_sigma = get_search_dist(
            path_norm,
            self.global_occ_grid,
            self.context_env_encoder,
            self.decoder_model,
            self.ar_model,
            self.quantizer_model,
            self.num_keys,
            device=self.device,
        )
        if search_dist_mu is not None:
            self.seq_num = search_dist_mu.shape[0]
            self.X = MultivariateNormal(search_dist_mu, search_dist_sigma)
            # visualize_distributions(self.global_occ_grid, search_dist_mu, search_dist_sigma, self.q_min, self.q_max, start, goal, show=False, save=True, file_name="vqmpt_distributions.png")
        else:
            self.X = None
            self.U = stats.uniform(np.zeros_like(self.q_min), np.ones_like(self.q_max))
        get_dist_time = time.time() - get_dist_start_time

        res = []
        i = 0
        max_t = step_size
        start_time = time.time()
        while max_t <= max_time + 1e-4:
            if max_t <= get_dist_time:
                path = []
            elif i == 0:
                path = self.algo.solve(start, goal, max_t - get_dist_time, float('inf'))
                i += 1
            else:
                self.algo.allowed_time = start_time + max_t - get_dist_time
                path = self.algo.continue_solve()

            success = len(path) > 0
            res.append((success, path))
            max_t += step_size

        return res

    def col_checker(self, v1, v2):
        valid = self.env.utils.is_edge_free(self.env, v1, v2)
        # if valid:
        #     return self.env.utils.calc_edge_len(v1, v2)
        # else:
        #     return float('inf')
        return valid

    def heuristic(self, v1, v2):
        return self.env.utils.calc_edge_len(v1, v2)

    def get_random_samples(self):
        '''Generates a random sample from the list of points
        '''
        index = 0
        random_samples = np.random.permutation(self.X.sample()*(self.q_max-self.q_min)+self.q_min)

        while True:
            yield random_samples[index, :]
            index += 1
            if index==self.seq_num:
                random_samples = np.random.permutation(self.X.sample()*(self.q_max-self.q_min)+self.q_min)
                index = 0

    def sample_biased(self, v, num_samples):
        if num_samples == 0:
            return []

        samples = []
        # generator = self.get_random_samples()
        for _ in range(num_samples):
            if self.X is None:
                random_state = ((self.q_max-self.q_min)*self.U.rvs()+self.q_min)
            else:
                random_state = next(self.get_random_samples())
            samples.append(random_state)

        return samples

    def sample_uniform(self, v, num_samples):
        if num_samples == 0:
            return []

        low = self.env.robot.get_joint_lower_bounds()
        high = self.env.robot.get_joint_higher_bounds()
        samples = []
        for _ in range(num_samples):
            random_state = [0] * self.env.robot.num_dim
            for i in range(self.env.robot.num_dim):
                random_state[i] = random.uniform(low[i], high[i])
            samples.append(random_state)

        return samples

    def extend_fn(self, v, g):
        start_time = time.time()
        path, _ = self.local_extend_rrt(v, g)
        end_time = time.time()
        self.col_check_time += end_time - start_time
        if len(path) <= 1:
            self.col_check_fail_time += end_time - start_time
        else:
            self.col_check_success_time += end_time - start_time

        self.extend_cnt += 1

        if self.log_extension_info:
            orig_path = self.env.utils.interpolate([v, g])
            self.dump_extension_information(orig_path, path, g)

        # print("Extension num: {}".format(self.extend_cnt))

        return [path], None, None

    def dump_extension_information(self, path, final_path, g):
        if not np.allclose(np.array(path[-1]), np.array(final_path[-1])):
            self.extension_col_cnt += 1

        extension_data = {}
        extension_data["local_planning_target"] = g
        extension_data["path_intended"] = path
        extension_data["path_actual"] = final_path

        if self.log_dir is not None:
            with open(osp.join(self.log_dir, "extension_data_{}.json".format(self.extend_cnt)), 'w') as f:
                json.dump(extension_data, f)

    def rrt_extend(self, v, g):
        if self.add_intermediate_state:
            return self.env.utils.rrt_extend_intermediate(self.env, v, g)
        else:
            return self.env.utils.rrt_extend(self.env, v, g)

    def local_extend_rrt(self, v, g):
        path = self.rrt_extend(v, g)
        return path, []
