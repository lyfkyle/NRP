import os.path as osp
import sys
import json
import time
import torch
import random
import math
import numpy as np

from nrp import ROOT_DIR

LOCAL_ENV_SIZE = 2
NUM_OF_RANDOM_SAMPLE = 500


class LocalNeuralExpander8D:
    def __init__(self, env, dim, occ_grid_dim, model_path, device, global_mode=False):
        self.env = env
        self.i = 0
        self.robot_dim = dim
        self.occ_grid_dim = occ_grid_dim
        self.device = device
        self.uniform_samples = None

        self._init_model(model_path, global_mode)

        self.visualize = False
        # self.visualize_heuristic_selected_node = False
        self.print_time = False

        self.num_of_samples = NUM_OF_RANDOM_SAMPLE

    def _init_model(self, model_path, global_mode=False):
        from nrp.planner.local_sampler_d.model_8d import DiscriminativeSampler

        self.fk = self.env.utils.FkTorch(self.device)

        print("Loading checkpoint {}".format(model_path))
        self.sampler = DiscriminativeSampler(self.robot_dim, self.occ_grid_dim, model_path, global_mode=global_mode)
        self.sampler.eval()
        print(self.device)
        self.sampler.to(self.device)

    @torch.no_grad()
    def warmup(self):
        start_time = time.time()
        for _ in range(50):
            occ_grid = torch.zeros((1, self.occ_grid_dim[0], self.occ_grid_dim[1], self.occ_grid_dim[2]), device="cuda")
            start = torch.zeros((1, 20), device="cuda")
            goal = torch.zeros((1, 21), device="cuda")
            samples = torch.zeros((1, NUM_OF_RANDOM_SAMPLE, 20), device="cuda")
            self.sampler.get_sel_score(occ_grid, start, goal, samples, fixed_env=True)
        end_time = time.time()
        print("warmup takes {}".format(end_time - start_time))

    def set_sample_bounds(self, low, high):
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
    def neural_expand(self, v, g, occ_grid_np):
        if self.print_time:
            start_time = time.perf_counter()

        # convert to GPU
        # num_to_sample = min(100, num_samples * 2)
        num_to_sample = self.num_of_samples
        # z = torch.randn(num_to_sample, self.z_dim, device=device) # generate more samples first, since we prune later
        start = torch.tensor(v, device=self.device, dtype=torch.float)
        occ_grid_t = self._preprocess_occ_grid(occ_grid_np)
        goal = torch.tensor(g, device=self.device, dtype=torch.float)

        if self.print_time:
            torch.cuda.synchronize()  # wait for mm to finish
            end_time = time.perf_counter()
            print("MyLocalSampler: convert to GPU takes {}".format(end_time - start_time))

        # sampling
        # start_time = time.perf_counter()
        samples = torch.rand((num_to_sample, self.robot_dim), device=self.device)
        samples = samples * (self.high - self.low) + self.low
        # samples = torch.cat((samples, start.view(1, self.robot_dim)), dim=0) # add current node to samples

        # if goal is in local env, allow sampler to select goal directly
        if self.env.utils.is_robot_within_local_env(v, g, LOCAL_ENV_SIZE):
            samples = torch.cat((goal.view(1, self.robot_dim), samples), dim=0)

        if self.print_time:
            torch.cuda.synchronize()  # wait for mm to finish
            end_time = time.perf_counter()
            print("MyLocalSampler: after random sampling takes {}".format(end_time - start_time))

        # select
        path = [v]
        tmp = torch.cat((start.view(1, -1), goal.view(1, -1), samples), dim=0)
        all_linkpos = self.fk.get_link_positions(tmp)
        start_linkpos = all_linkpos[0].view(-1)
        goal_linkpos = all_linkpos[1].view(-1)
        sample_linkpos = all_linkpos[2:]
        start_t = torch.cat((start, start_linkpos))
        samples_t = torch.cat((samples, sample_linkpos), dim=-1)
        if self.print_time:
            torch.cuda.synchronize()  # wait for mm to finish
            end_time = time.perf_counter()
            print("MyLocalSampler: forward kinematics takes {}".format(end_time - start_time))

        goal_direction = torch.atan2(goal[1], goal[0]).view(1)
        goal_t = torch.cat((goal, goal_linkpos, goal_direction))

        occ_grid_batch = occ_grid_t.unsqueeze(0)
        start_batch = start_t.unsqueeze(0)
        goal_batch = goal_t.unsqueeze(0)
        samples_batch = samples_t.unsqueeze(0)  # 1 x N x dim
        sel_scores = self.sampler.get_sel_score(
            occ_grid_batch, start_batch, goal_batch, samples_batch, fixed_env=True
        ).view(-1)
        best_indice = torch.argmax(sel_scores)

        if self.print_time:
            torch.cuda.synchronize()  # wait for mm to finish
            end_time = time.perf_counter()
            print("MyLocalSampler: filtering 2 takes {}".format(end_time - start_time))

        if self.visualize:
            all_samples_list = samples.cpu().numpy().tolist()
            self.env.utils.visualize_nodes_local(
                occ_grid_np,
                all_samples_list,
                v,
                None,
                show=False,
                save=True,
                file_name=osp.join(self.log_dir, "all_samples_viz_{}.png".format(self.i)),
            )

        if self.print_time:
            torch.cuda.synchronize()  # wait for mm to finish
            end_time = time.perf_counter()
            print("MyLocalSampler: calling selector takes {}".format(end_time - start_time))

        # get best samples
        # start_time  = time.time()
        selected_sample = samples[best_indice].view(-1).cpu().numpy().tolist()
        # torch.cuda.synchronize() # wait for mm to finish
        # end_time = time.perf_counter()
        # dbg_print("MyLocalSampler: convert back to cpu takes {}".format(end_time - start_time))

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

        path.append(selected_sample)

        if self.print_time:
            torch.cuda.synchronize()  # wait for mm to finish
            end_time = time.perf_counter()
            print("MyLocalSampler: before return takes {}".format(end_time - start_time))

        self.i += 1

        return path


class LocalNeuralExpander11D(LocalNeuralExpander8D):
    def _init_model(self, model_path, global_mode=False):
        from nrp.planner.local_sampler_d.model_11d import DiscriminativeSampler
        from nrp.env.fetch_11d.fk.model import ProxyFkTorch

        linkpos_dim = 24
        fkmodel_path = osp.join(ROOT_DIR, "models/fetch_11d_approx_fk/model_fk_v2.pt")
        self.fk = ProxyFkTorch(self.robot_dim, linkpos_dim, fkmodel_path, self.device)

        # selector
        print("Loading checkpoint {}".format(model_path))

        self.sampler = DiscriminativeSampler(self.robot_dim, self.occ_grid_dim, model_path, global_mode=global_mode)

        self.sampler.eval()
        print(self.device)
        self.sampler.to(self.device)

    @torch.no_grad()
    def warmup(self):
        start_time = time.time()
        for _ in range(50):
            occ_grid = torch.zeros((self.occ_grid_dim[0], self.occ_grid_dim[1], self.occ_grid_dim[2]), device="cuda")
            occ_grid_t = self.env.utils.add_pos_channels(occ_grid).unsqueeze(0)
            start_t = torch.zeros((1, 35), device="cuda")
            goal_t = torch.zeros((1, 36), device="cuda")
            samples = torch.zeros((1, NUM_OF_RANDOM_SAMPLE, 35), device="cuda")
            self.sampler.get_sel_score(occ_grid_t, start_t, goal_t, samples, fixed_env=True)
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
