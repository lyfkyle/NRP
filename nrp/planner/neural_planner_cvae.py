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
from nrp import ROOT_DIR

CUR_DIR = osp.dirname(osp.abspath(__file__))

DEBUG = True

MAX_NUM_RECUR = 1
LOCAL_ENV_SIZE = 2
NUM_OF_SAMPLES = 1
ADD_INTERMEDIATE_STATE = False


class GlobalSampler8D:
    def __init__(self, env, dim, occ_grid_dim, model_path, device):
        self.env = env
        self.i = 0
        self.robot_dim = dim
        self.occ_grid_dim = occ_grid_dim
        self.device = device
        self.uniform_samples = None

        self._init_model(model_path)

        self.visualize = False
        self.print_time = False
        self.num_of_samples = NUM_OF_SAMPLES

    def _init_model(self, model_path):
        from planner.cvae.model_8d import VAEInference

        self.fk = self.env.utils.FkTorch(self.device)

        print("Loading checkpoint {}".format(model_path))

        # define networks
        z_dim = 5
        linkpos_dim = 12
        state_dim = self.robot_dim + linkpos_dim
        goal_state_dim = self.robot_dim + linkpos_dim + 1
        context_dim = state_dim + goal_state_dim
        self.generator = VAEInference(z_dim, context_dim, state_dim)
        self.generator = torch.jit.script(self.generator)
        self.generator.load_state_dict(torch.load(model_path))
        self.generator.eval()
        print(self.device)
        self.generator.to(self.device)

    @torch.no_grad()
    def warmup(self):
        start_time = time.time()
        for _ in range(10):
            occ_grid_t = torch.zeros(
                (1, self.occ_grid_dim[0], self.occ_grid_dim[1], self.occ_grid_dim[2]), device="cuda"
            )
            start_t = torch.zeros((1, 20), device="cuda")
            goal_t = torch.zeros((1, 21), device="cuda")
            context_t = torch.cat((start_t, goal_t), dim=-1)
            self.generator(NUM_OF_SAMPLES, occ_grid_t, context_t)
        end_time = time.time()
        print("warmup takes {}".format(end_time - start_time))

    def set_robot_bounds(self, low, high):
        self.low = torch.tensor(low, device=self.device)
        self.high = torch.tensor(high, device=self.device)

    def enable_neural_select(self, enable):
        self.use_nueral_select = enable

    def set_uniform_samples(self, samples):
        self.uniform_samples = samples

    def set_mesh(self, mesh):
        """
        For visualization only
        """
        self.mesh = mesh

    def _preprocess_occ_grid(self, occ_grid_np):
        occ_grid_t = torch.tensor(occ_grid_np, device=self.device, dtype=torch.float).view(
            self.occ_grid_dim[0], self.occ_grid_dim[1], self.occ_grid_dim[2]
        )
        return occ_grid_t

    @torch.no_grad()
    def sample(self, v, g, occ_grid_np):
        if self.print_time:
            start_time = time.perf_counter()

        # convert to GPU
        start = torch.tensor(v, device=self.device, dtype=torch.float)
        occ_grid_t = self._preprocess_occ_grid(occ_grid_np)
        goal = torch.tensor(g, device=self.device, dtype=torch.float)

        if self.print_time:
            torch.cuda.synchronize()  # wait for mm to finish
            end_time = time.perf_counter()
            print("MyLocalSampler: convert to GPU takes {}".format(end_time - start_time))

        # visualize samples
        # if self.visualize:
        #     self.env.utils.visualize_nodes_local(occ_grid_np, samples.detach().cpu().numpy(), v, None,
        #         show=False, save=True, file_name = osp.join(self.log_dir, "all_samples_viz_{}.png".format(self.i)))

        # select
        tmp = torch.cat((start.view(1, -1), goal.view(1, -1)), dim=0)
        all_linkpos = self.fk.get_link_positions(tmp)
        start_linkpos = all_linkpos[0].view(-1)
        goal_linkpos = all_linkpos[1].view(-1)
        start_t = torch.cat((start, start_linkpos))
        if self.print_time:
            torch.cuda.synchronize()  # wait for mm to finish
            end_time = time.perf_counter()
            print("MyLocalSampler: forward kinematics takes {}".format(end_time - start_time))

        goal_direction = torch.atan2(goal[1], goal[0]).view(1)
        goal_t = torch.cat((goal, goal_linkpos, goal_direction))
        context_t = torch.cat((start_t, goal_t), dim=-1)
        samples = self.generator(NUM_OF_SAMPLES, occ_grid_t, context_t)[:, : self.robot_dim]

        if self.print_time:
            torch.cuda.synchronize()  # wait for mm to finish
            end_time = time.perf_counter()
            print("MyLocalSampler: sampling takes {}".format(end_time - start_time))

        # get best samples
        selected_sample = samples[0].cpu().numpy().tolist()

        if self.visualize:
            self.env.utils.visualize_nodes_local(
                occ_grid_np,
                [np.array(selected_sample)],
                v,
                g,
                show=False,
                save=True,
                file_name=osp.join(self.log_dir, "selected_samples_viz_{}.png".format(self.i)),
            )

        if self.print_time:
            torch.cuda.synchronize()  # wait for mm to finish
            end_time = time.perf_counter()
            dbg_print("MyLocalSampler: before return takes {}".format(end_time - start_time))

        self.i += 1

        return selected_sample


class GlobalSampler11D(GlobalSampler8D):
    def _init_model(self, model_path):
        from planner.cvae.model_11d import VAEInference
        from env.fetch_11d.fk.model import ProxyFkTorch

        linkpos_dim = 24
        fkmodel_path = osp.join(ROOT_DIR, "models/fetch_11d_approx_fk/model_fk_v2.pt")
        self.fk = ProxyFkTorch(self.robot_dim, linkpos_dim, fkmodel_path, self.device)

        # selector
        print("Loading checkpoint {}".format(model_path))

        # define networks
        z_dim = 8
        state_dim = self.robot_dim + linkpos_dim
        goal_state_dim = self.robot_dim + linkpos_dim + 1
        context_dim = state_dim + goal_state_dim
        self.generator = VAEInference(z_dim, context_dim, state_dim)
        self.generator = torch.jit.script(self.generator)
        self.generator.load_state_dict(torch.load(model_path))
        self.generator.eval()
        print(self.device)
        self.generator.to(self.device)

    @torch.no_grad()
    def warmup(self):
        start_time = time.time()
        for _ in range(10):
            occ_grid = torch.zeros((self.occ_grid_dim[0], self.occ_grid_dim[1], self.occ_grid_dim[2]), device="cuda")
            occ_grid_t = self.env.utils.add_pos_channels(occ_grid).unsqueeze(0)
            start_t = torch.zeros((1, 35), device="cuda")
            goal_t = torch.zeros((1, 36), device="cuda")
            context_t = torch.cat((start_t, goal_t), dim=-1)
            self.generator(NUM_OF_SAMPLES, occ_grid_t, context_t)
        end_time = time.time()
        print("warmup takes {}".format(end_time - start_time))

    def set_mesh(self, mesh):
        """
        For visualization only
        """
        self.mesh = mesh

    def _preprocess_occ_grid(self, occ_grid_np):
        occ_grid_t = torch.tensor(occ_grid_np, device=self.device, dtype=torch.float).view(
            self.occ_grid_dim[0], self.occ_grid_dim[1], self.occ_grid_dim[2]
        )
        occ_grid_t = self.env.utils.add_pos_channels(occ_grid_t)
        return occ_grid_t


class NeuralPlannerCVAE:
    """
    Purely discriminative
    Use selector to extend
    """

    def __init__(self, env, model_path, optimal=False, dim=8, occ_grid_dim=[1, 40, 40], device=torch.device("cuda")):
        self.dim = dim
        self.env = env
        self.log_dir = None
        self.log_extension_info = False

        if env.name == "snake_8d":
            self.sampler = GlobalSampler8D(env, self.dim, occ_grid_dim, model_path, device=device)
        elif env.name == "fetch_11d":
            self.sampler = GlobalSampler11D(env, self.dim, occ_grid_dim, model_path, device=device)

        print(f"===================== Optimal: {optimal} ======================")
        if not optimal:
            self.algo = RRT(self.col_checker, self.heuristic, self.sample_global, self.expand_fn)
            self.add_intermediate_state = False
        else:
            self.algo = InformedRRTStar(self.col_checker, self.heuristic, self.sample_global, self.expand_fn)
            self.add_intermediate_state = False

        self.uniform_bias = 0.5  # recommended by original paper

    def clear(self):
        self.env = None
        self.global_occ_grid = None
        self.expansion_col_cnt = 0
        self.neural_expansion_time = 0
        self.col_check_time = 0
        self.col_check_success_time = 0
        self.col_check_fail_time = 0

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

    def solve_step_expansion(self, env, start, goal, max_extensions, step_size=50, mesh=None):
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

    def sample_global(self, v, num_samples):
        # print(self.algo.graph.number_of_nodes())
        # self.env.utils.visualize_nodes(self.global_occ_grid, list(self.algo.graph.nodes()), None, None, self.start, self.goal)

        if num_samples == 0:
            return []

        samples = []
        low = self.env.robot.get_joint_lower_bounds()
        high = self.env.robot.get_joint_higher_bounds()

        if random.uniform(0, 1) < self.uniform_bias:
            for _ in range(num_samples):
                random_state = [0] * self.env.robot.num_dim
                for i in range(self.env.robot.num_dim):
                    random_state[i] = random.uniform(low[i], high[i])
                samples.append(random_state)
        else:
            v = self.start
            g = self.goal
            sample = self.sampler.sample(v, g, self.global_occ_grid)
            samples.append(sample)

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
