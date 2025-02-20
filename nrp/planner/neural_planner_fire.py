import os.path as osp
import sys
import json
import time
import torch
import random
import math
import numpy as np
import pickle

from nrp.planner.informed_rrt import InformedRRTStar
from nrp.planner.fire.model import FireModel
from nrp.planner.rrt import RRT
from nrp.planner.fire.fire import Fire

CUR_DIR = osp.dirname(osp.abspath(__file__))

DATA_DIR = osp.join(CUR_DIR, "fire/dataset/fire_database")

DEBUG = True


SIGMA = 0.2
ADD_INTERMEDIATE_STATE = False

class FireSampler:
    def __init__(self, env, dim, occ_grid_dim, model_path, device):
        self.env = env
        self.i = 0
        self.robot_dim = dim
        self.occ_grid_dim = occ_grid_dim
        self.device = device
        self.uniform_samples = None

        self.fk = self.env.utils.FkTorch(device)

        # define networks
        self._fire_similarity_model = FireModel(self.robot_dim)
        # self._fire_similarity_model = torch.jit.script(self._fire_similarity_model)
        self._fire_similarity_model.load_state_dict(torch.load(model_path))
        self._fire_similarity_model.eval()
        print(device)
        self._fire_similarity_model.to(device)

        self.visualize = False
        self.print_time = False

        # load existing entries
        self._fire_database = Fire(env)
        self._fire_database.load_existing_database(DATA_DIR)

    @torch.no_grad()
    def synthesize_sampling_distributions(self, env, start, goal):
        def retrieve_relevant_experience(target):
            bs = 512
            prim_latent_value = []
            q_proj = self._fire_database.get_fetch_projection(target)
            for idx in range(0, len(self._primitives), bs):
                print(f"primitives: {idx}/{len(self._primitives)}, {len(prim_latent_value)}")
                center_poss = []
                occ_grids = []
                q_targets = []
                q_projs = []
                for i in range(idx, min(len(self._primitives), idx + bs)):
                    center_poss.append(self._primitives[i][1])
                    occ_grid_t = torch.tensor(self._primitives[i][0], dtype=torch.float32).view(
                        self.occ_grid_dim, self.occ_grid_dim, self.occ_grid_dim
                    )
                    occ_grid_t = self.env.utils.add_pos_channels(occ_grid_t)
                    occ_grids.append(occ_grid_t.cpu().numpy().tolist())
                    q_targets.append(goal)
                    q_projs.append(self._fire_database.normalize_projections(self._primitives[i], q_proj))

                cur_bs = len(center_poss)
                occ_grid_t = torch.tensor(occ_grids, dtype=torch.float32, device=self.device).view(
                    cur_bs, 4, self.occ_grid_dim, self.occ_grid_dim, self.occ_grid_dim
                )
                center_pos_t = torch.tensor(center_poss, dtype=torch.float32, device=self.device).view(cur_bs, -1)
                q_target_t = torch.tensor(q_targets, dtype=torch.float32, device=self.device).view(cur_bs, -1)
                q_proj_t = torch.tensor(q_projs, dtype=torch.float32, device=self.device).view(cur_bs, -1)  # flatten

                z = self._fire_similarity_model(occ_grid_t, center_pos_t, q_target_t, q_proj_t).cpu().numpy()

                z1 = np.expand_dims(z, axis=1)  # N x 1 x 8
                z2 = np.expand_dims(np.array(self._entry_latent_value), axis=0)  # 1 x N x 8
                dist = np.linalg.norm(z1 - z2, axis=-1)
                indices = np.nonzero(dist < 0.5)
                for idx2 in indices[1]:
                    self._relevant_database_indices.add(idx2)

                prim_latent_value += z.tolist()

            # for idx1, z1 in enumerate(prim_latent_value):
            # z1 = np.expand_dims(np.array(prim_latent_value), axis=1)  # N x 1 x 8
            # z2 = np.expand_dims(np.array(self._entry_latent_value), axis=0)  # 1 x N x 8
            # dist = z1 - z2
            # # dist = np.array(z1).reshape(1, 8) - np.array(self._entry_latent_value)
            # dist = np.linalg.norm(dist, axis=-1).reshape(-1)
            # print(dist.shape)
            # indices = np.nonzero(dist < 0.5)[1]
            # for idx2 in indices:
            #     self._relevant_database_indices.add(idx2)

            # for idx2, z2 in enumerate(self._entry_latent_value):
            #     if np.linalg.norm(np.array(z1) - np.array(z2)) < 0.5:  # if nearby in latent space.
            #         self._relevant_database_entries.append(self._fire_database.database_entries[idx2])

        self._primitives = self._fire_database.extract_primitives(env.occ_grid)
        with open(osp.join(DATA_DIR, "latent_value.pkl"), "rb") as f:
            self._entry_latent_value = pickle.load(f)

        print(f"Database size is {len(self._entry_latent_value)}")
        assert len(self._entry_latent_value) == len(self._fire_database.database_entries)

        self._relevant_database_indices = set()
        retrieve_relevant_experience(goal)
        goal_relevant_entries_num = len(self._relevant_database_indices)
        print(f"Retrieves {goal_relevant_entries_num} database entries for goal as target")
        retrieve_relevant_experience(start)
        print(
            f"Retrieves {len(self._relevant_database_indices) - goal_relevant_entries_num} database entries for start as target"
        )

        return self._relevant_database_indices

    def sample(self):
        database_entry_idx = random.choice(list(self._relevant_database_indices))
        database_entry = self._fire_database.database_entries[database_entry_idx]
        q_sample = np.random.normal(np.array(database_entry.q), SIGMA)
        return q_sample

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


class NeuralPlannerFire:
    """
    Purely discriminative
    Use selector to extend
    """

    def __init__(self, env, model_path, optimal=False, dim=11, occ_grid_dim=20, device=torch.device("cuda")):
        self.dim = dim
        self.env = env
        self.log_dir = None
        self.log_extension_info = False

        self.sampler = FireSampler(env, self.dim, occ_grid_dim, model_path, device=device)

        print(f"===================== Optimal: {optimal} ======================")
        if not optimal:
            self.algo = RRT(self.col_checker, self.heuristic, self.sample_global, self.expand_fn)
            self.add_intermediate_state = False
        else:
            self.algo = InformedRRTStar(self.col_checker, self.heuristic, self.sample_global, self.expand_fn)
            self.add_intermediate_state = False

        self.uniform_bias = 0.2

    def clear(self):
        self.env = None
        self.global_occ_grid = None
        self.expansion_col_cnt = 0
        self.algo.clear()
        self.neural_expansion_time = 0
        self.col_check_time = 0
        self.sampler.i = 0
        self.extend_cnt = 0
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

        self.sampler.synthesize_sampling_distributions(env, start, goal)

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

        self.sampler.synthesize_sampling_distributions(env, start, goal)

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

        start_time = time.time()
        self.sampler.synthesize_sampling_distributions(env, start, goal)
        end_time = time.time()
        self.offline_preprocess_time = end_time - start_time

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
            sample = self.sampler.sample()
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
