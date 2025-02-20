import os.path as osp
import sys
import json
import time
import torch
import random
import math
import numpy as np

from nrp.planner.informed_rrt import InformedRRTStar
from nrp.planner.rrt import RRT
from nrp.planner.local_sampler_d.local_sampler_d import LocalNeuralExpander8D, LocalNeuralExpander11D

CUR_DIR = osp.dirname(osp.abspath(__file__))

DEBUG = True
MAX_NUM_RECUR = 1
ADD_INTERMEDIATE_STATE = False


class NRP_d:
    """NRP using discriminative sampler."""

    def __init__(self, env, model_path, optimal=False, dim=8, occ_grid_dim=[1, 40, 40], device=torch.device("cuda")):
        self.dim = dim
        self.env = env
        self.nueral_select_cnt = 1
        self.log_dir = None
        self.collect_dataset = False
        self.log_extension_info = False
        self.use_online_samples = False
        self.ego_local_env = True
        self.only_sample_col_free = False
        self.sample_around_goal = False
        self.neural_expand_if_sl_fail = False
        self.local_env_size = 2.0

        self._init_sampler(env, occ_grid_dim, model_path, device)

        print(f"=============== RRT_NE_D Optimal: {optimal} =================")
        if not optimal:
            self.algo = RRT(self.col_checker, self.heuristic, self.sample_uniform, self.expand_fn)
            self.add_intermediate_state = False
        else:
            self.algo = InformedRRTStar(self.col_checker, self.heuristic, self.sample_uniform, self.expand_fn)
            self.add_intermediate_state = False

        self.max_num_recur = MAX_NUM_RECUR
        self.sl_bias = 0.01
        self.add_intermediate_state = ADD_INTERMEDIATE_STATE

        random.seed(0)

    def _init_sampler(self, env, occ_grid_dim, model_path, device):
        if env.name == "snake_8d":
            self.sampler = LocalNeuralExpander8D(env, self.dim, occ_grid_dim, model_path, device=device, global_mode=False)
        elif env.name == "fetch_11d":
            self.sampler = LocalNeuralExpander11D(env, self.dim, occ_grid_dim, model_path, device=device, global_mode=False)

    def clear(self):
        self.env = None
        self.global_occ_grid = None
        self.sample_memory = {}
        self.expansion_col_cnt = 0
        self.algo.clear()
        self.neural_expansion_time = 0
        self.neural_expansion_success_time = 0
        self.neural_expansion_fail_time = 0
        self.col_check_time = 0
        self.col_check_success_time = 0
        self.col_check_fail_time = 0
        self.neural_expansion_cnt = 0
        self.neural_expansion_col_cnt = 0
        self.selector_col_cnt = 0
        self.dataset_col_rate = 0
        self.dataset_diff_cnt = 0
        self.sampler.i = 0

    def solve(self, env, start, goal, allowed_time=2.0, max_samples=float("inf"), mesh=None):
        self.clear()
        self.algo.clear()
        self.algo.env = env
        self.env = env
        self.start = start
        self.goal = np.array(goal)
        self.global_occ_grid = self.env.get_occupancy_grid()
        self.sampler.set_mesh(mesh)
        self.low_bounds = env.robot.get_joint_lower_bounds()
        self.high_bounds = env.robot.get_joint_higher_bounds()

        self.sampler.warmup()

        path = self.algo.solve(start, goal, allowed_time, max_samples)

        return path

    def solve_step_extension(self, env, start, goal, max_extensions, step_size=50, mesh=None):
        self.clear()
        self.algo.clear()
        self.algo.env = env
        self.env = env
        self.start = start
        self.goal = np.array(goal)
        self.global_occ_grid = self.env.get_occupancy_grid()
        self.sampler.set_mesh(mesh)
        self.low_bounds = env.robot.get_joint_lower_bounds()
        self.high_bounds = env.robot.get_joint_higher_bounds()

        self.sampler.warmup()

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
        self.start = start
        self.goal = np.array(goal)
        self.global_occ_grid = self.env.get_occupancy_grid()
        self.sampler.set_mesh(mesh)
        self.low_bounds = env.robot.get_joint_lower_bounds()
        self.high_bounds = env.robot.get_joint_higher_bounds()

        self.sampler.warmup()

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
        # print(self.algo.graph.number_of_nodes())
        # self.env.utils.visualize_nodes(self.global_occ_grid, list(self.algo.graph.nodes()), None, None, self.start, self.goal)

        if num_samples == 0:
            return []

        samples = []
        low = self.env.robot.get_joint_lower_bounds()
        high = self.env.robot.get_joint_higher_bounds()
        for _ in range(num_samples):
            if self.sample_around_goal:
                noises = np.random.normal(scale=0.2, size=len(self.goal))
                random_state = np.array(self.goal) + noises
                random_state = random_state.tolist()
                if self.only_sample_col_free:
                    while not self.env.pb_ompl_interface.is_state_valid(random_state):
                        noises = np.random.normal(size=len(self.goal))
                        random_state = np.array(self.goal) + noises
                        random_state = random_state.tolist()
            else:
                random_state = [0] * self.env.robot.num_dim
                for i in range(self.env.robot.num_dim):
                    random_state[i] = random.uniform(low[i], high[i])

                if self.only_sample_col_free:
                    while not self.env.pb_ompl_interface.is_state_valid(random_state):
                        random_state = [0] * self.env.robot.num_dim
                        for i in range(self.env.robot.num_dim):
                            random_state[i] = random.uniform(low[i], high[i])

            samples.append(random_state)

        return samples

    def expand_fn(self, v, g):
        if random.uniform(0, 1) < self.sl_bias:
            sl_expansion_path = self.sl_expansion_fn(v, g)
            return [sl_expansion_path], None, None
        else:
            nerual_expansion_path = self.neural_expansion_fn(v, g)
            return [nerual_expansion_path], None, None

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

        # end_time = time.time()
        # print("sl_extend takes {}".format(end_time - start_time))

        return sl_expansion_path

    def neural_expansion_fn(self, v, g):
        start_time = time.time()

        cur_v = v
        path = [v]
        for _ in range(self.max_num_recur):
            recur_start_time = time.time()
            success = False

            local_g = self.env.utils.global_to_local(g, cur_v)
            local_v = self.env.utils.global_to_local(cur_v, cur_v)
            local_occ_grid = self.env.get_local_occ_grid(cur_v)

            # assert self.maze.pb_ompl_interface.is_state_valid(cur_v)
            local_path = self.sampler.neural_expand(local_v, local_g, local_occ_grid)
            global_path = [self.env.utils.local_to_global(x, cur_v) for x in local_path]

            if (
                global_path[-1][0] > self.low_bounds[0]
                and global_path[-1][0] < self.high_bounds[0]
                and global_path[-1][1] > self.low_bounds[1]
                and global_path[-1][1] < self.high_bounds[1]
            ):
                pass
            else:
                print("Neural expansion selects a samples outside env bounds. This should happen rarely")

            # Add to path
            path += global_path[1:]
            success = True

            # change cur_v if using ego view mode
            if self.ego_local_env:
                cur_v = global_path[-1]

            recur_end_time = time.time()
            if success:
                self.neural_expansion_success_time += recur_end_time - recur_start_time
            else:
                self.neural_expansion_fail_time += recur_end_time - recur_start_time

        self.neural_expansion_cnt += 1

        # print("Extension num: {}".format(self.algo.num_sampler_called))

        # extend towards global g.
        path.append(g)
        # print(len(path))

        end_time = time.time()
        self.neural_expansion_time += end_time - start_time
        # print("Calling neural networks takes : {}".format(end_time - start_time))

        # Collision check
        start_time = time.time()
        final_path = [v]
        for i in range(1, len(path)):
            v1 = path[i - 1]
            v2 = path[i]
            res_path = self.expand_rrt(v1, v2)
            if len(res_path) > 1:
                final_path += res_path[1:]

            if not np.allclose(np.array(res_path[-1]), np.array(v2)):
                if i == 1:
                    self.neural_expansion_col_cnt += 1

                self.expansion_col_cnt += 1
                # print("collision detected!!!")
                break

        # end_time = time.time()
        # self.col_check_time += end_time - start_time
        # print("Collision check takes : {}".format(end_time - start_time))

        # print(final_path)

        # Avoid float precision issues
        if np.allclose(np.array(final_path[-1]), np.array(g)):
            final_path[-1] = g

        # record down information
        if self.log_extension_info:
            self.dump_extension_information(path, final_path, g)

        if len(final_path) <= 1:
            self.col_check_fail_time += end_time - start_time
        else:
            self.col_check_success_time += end_time - start_time

        # print(final_path, path)

        return final_path

    def dump_extension_information(self, path, final_path, g):
        # self.env.utils.visualize_nodes_global(self.global_occ_grid, final_path, v, g)

        # calculate col rate
        if len(path) <= 1:
            selector_col = 0
        else:
            selector_col = 0
            for i in range(1, len(path) - 1):
                v1 = path[i - 1]
                v2 = path[i]
                is_free = self.env.utils.is_edge_free(self.env, v1, v2)
                if not is_free:
                    selector_col += 1

            # print(selector_col, len(path))
            self.selector_col_cnt += selector_col

        extension_data = {}
        extension_data["local_planning_target"] = g
        extension_data["path_intended"] = path
        extension_data["path_actual"] = final_path

        if self.log_dir is not None:
            with open(osp.join(self.log_dir, "extension_data_{}.json".format(self.algo.num_expansions + 1)), "w") as f:
                json.dump(extension_data, f)

    def expand_rrt(self, v, g):
        if self.add_intermediate_state:
            return self.env.utils.rrt_extend_intermediate(self.env, v, g)
        else:
            return self.env.utils.rrt_extend(self.env, v, g)

    # def convert_to_local_g(self, goal_pos):
    #     if math.fabs(goal_pos[0]) > 2.5 or math.fabs(goal_pos[1]) > 2.5:
    #         theta = math.atan2(goal_pos[1], goal_pos[0])
    #         if math.fabs(goal_pos[0]) > math.fabs(goal_pos[1]):
    #             tmp = 2.5 if goal_pos[0] > 0 else -2.5
    #             goal_pos[1] = math.tan(theta) * tmp
    #             goal_pos[0] = tmp
    #         else:
    #             tmp = 2.5 if goal_pos[1] > 0 else -2.5
    #             goal_pos[0] = tmp / math.tan(theta)
    #             goal_pos[1] = tmp

    #     return goal_pos[:2]

class NRPGlobal_d(NRP_d):
    def _init_sampler(self, env, occ_grid_dim, model_path, device):
        if env.name == "snake_8d":
            self.sampler = LocalNeuralExpander8D(env, self.dim, occ_grid_dim, model_path, device=device, global_mode=True)
        elif env.name == "fetch_11d":
            self.sampler = LocalNeuralExpander11D(env, self.dim, occ_grid_dim, model_path, device=device, global_mode=True)

    def neural_expansion_fn(self, v, g):
        start_time = time.time()
        low = np.array(self.env.robot.get_joint_lower_bounds())
        high = np.array(self.env.robot.get_joint_higher_bounds())

        cur_v = v
        path = [v]
        for _ in range(self.max_num_recur):
            recur_start_time = time.time()
            success = False

            # TODO only normalize if env is snake_8d. This discrepancy is not desired
            if self.env.name == "snake_8d":
                global_g = self.env.utils.normalize_state(np.array(g), low, high)
                global_v = self.env.utils.normalize_state(np.array(cur_v), low, high)
            else:
                global_g = np.array(g)
                global_v = np.array(cur_v)

            # assert self.maze.pb_ompl_interface.is_state_valid(cur_v)
            global_path = self.sampler.neural_expand(global_v, global_g, self.global_occ_grid)
            # print(local_path)

            if self.env.name == "snake_8d":
                global_path = [self.env.utils.unnormalize_state(np.array(x), low, high) for x in global_path]

            if (
                global_path[-1][0] > self.low_bounds[0]
                and global_path[-1][0] < self.high_bounds[0]
                and global_path[-1][1] > self.low_bounds[1]
                and global_path[-1][1] < self.high_bounds[1]
            ):
                pass
            else:
                # print(global_path[-1])
                print("Neural expansion selects a samples outside env bounds. This should happen rarely")

            # Add to path
            path += global_path[1:]
            success = True

            # change cur_v if using ego view mode
            if self.ego_local_env:
                cur_v = global_path[-1]

            recur_end_time = time.time()
            if success:
                self.neural_expansion_success_time += recur_end_time - recur_start_time
            else:
                self.neural_expansion_fail_time += recur_end_time - recur_start_time

        self.neural_expansion_cnt += 1

        # print("Extension num: {}".format(self.algo.num_sampler_called))

        # extend towards global g.
        path.append(g)
        # print(len(path))

        end_time = time.time()
        self.neural_expansion_time += end_time - start_time
        # print("Calling neural networks takes : {}".format(end_time - start_time))

        # Collision check
        start_time = time.time()
        final_path = [v]
        for i in range(1, len(path)):
            v1 = path[i - 1]
            v2 = path[i]
            res_path = self.expand_rrt(v1, v2)
            if len(res_path) > 1:
                final_path += res_path[1:]

            if not np.allclose(np.array(res_path[-1]), np.array(v2)):
                if i == 1:
                    self.neural_expansion_col_cnt += 1

                self.expansion_col_cnt += 1
                # print("collision detected!!!")
                break

        # end_time = time.time()
        # self.col_check_time += end_time - start_time
        # print("Collision check takes : {}".format(end_time - start_time))

        # print(final_path)

        # Avoid float precision issues
        if np.allclose(np.array(final_path[-1]), np.array(g)):
            final_path[-1] = g

        # record down information
        if self.log_extension_info:
            self.dump_extension_information(path, final_path, g)

        if len(final_path) <= 1:
            self.col_check_fail_time += end_time - start_time
        else:
            self.col_check_success_time += end_time - start_time

        # print(final_path, path)

        return final_path