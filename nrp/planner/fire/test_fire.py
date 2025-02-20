import os.path as osp
import sys

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), "../../"))

import numpy as np
import random
import torch
import pickle

# from planner.informed_rrt import InformedRRTStar
from env.fetch_11d.maze import Fetch11DEnv
from planner.fire.model import FireModel
from planner.rrt import RRT
from planner.fire.fire import Fire, FireEntry

CUR_DIR = osp.dirname(osp.abspath(__file__))

DATA_DIR = osp.join(CUR_DIR, "dataset/fire_database")

DEBUG = True


def dbg_print(x):
    if DEBUG:
        print(x)


SIGMA = 0.2


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
        self._primitives = self._fire_database.extract_primitives(env.occ_grid)
        with open(osp.join(DATA_DIR, "latent_value.pkl"), "rb") as f:
            self._entry_latent_value = pickle.load(f)

        assert len(self._entry_latent_value) == len(self._fire_database.database_entries)

        bs = 512
        self._prim_latent_value = []
        q_proj = self._fire_database.get_fetch_projection(goal)

        for idx in range(0, len(self._primitives), bs):
            print(f"primitives: {idx}/{len(self._primitives)}, {len(self._prim_latent_value)}")
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

            z = self._fire_similarity_model(occ_grid_t, center_pos_t, q_target_t, q_proj_t).cpu().numpy().tolist()
            self._prim_latent_value += z

            # q_proj = self._fire_database.get_fetch_projection(start)
            # q_proj_normalized = self._fire_database.normalize_projections(primitive, q_proj)

            # q_target_t = torch.Tensor(start)
            # q_proj_t = torch.Tensor(q_proj_normalized).view(-1)  # flatten
            # z = self._fire_similarity_model(occ_grid_t, center_pos_t, q_target_t, q_proj_t).cpu().numpy()
            # self._entry_latent_value.append(z)

        assert len(self._prim_latent_value) == len(self._primitives)

        self._relevant_database_entries = []
        # for idx1, z1 in enumerate(self._prim_latent_value):
        for _ in range(5):
            idx1 = random.randint(0, len(self._prim_latent_value) - 1)
            z1 = self._prim_latent_value[idx1]
            self._fire_database.visualize_primitives(
                self._primitives[idx1][0], self._primitives[idx1][1], None, q_target=goal, name=f"query_{idx1}"
            )

            for idx2, z2 in enumerate(self._entry_latent_value):
                if np.linalg.norm(np.array(z1) - np.array(z2)) < 0.5:  # if nearby in latent space.
                    self._relevant_database_entries.append(self._fire_database.database_entries[idx2])

                    # visualize
                    self._fire_database.visualize_primitives(
                        self._fire_database.database_entries[idx2].occ_grid,
                        self._fire_database.database_entries[idx2].center_pos,
                        self._fire_database.database_entries[idx2].q,
                        self._fire_database.database_entries[idx2].q_target,
                        name=f"relevant_{idx1}_{idx2}",
                    )
                    input()

        print(f"Retrieves {len(self._relevant_database_entries)} database entries")

        return self._relevant_database_entries

    def sample(self):
        database_entry = random.choice(self._relevant_database_entries)
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
        self.env = None
        self.log_dir = None
        self.log_extension_info = False
        self.ego_local_env = True
        self.env = env

        self.sampler = FireSampler(env, self.dim, occ_grid_dim, model_path, device=device)

    def clear(self):
        self.env = None
        self.global_occ_grid = None
        self.extension_col_cnt = 0
        self.neural_extend_time = 0
        self.col_check_time = 0
        self.sampler.i = 0
        self.extend_cnt = 0
        self.col_check_success_time = 0
        self.col_check_fail_time = 0

    def solve(self, maze, start, goal, allowed_time=2.0, max_samples=float("inf"), mesh=None):
        self.clear()
        self.env = maze
        self.start = start
        self.goal = np.array(goal)
        self.global_occ_grid = self.env.get_occupancy_grid()
        self.sampler.set_mesh(mesh)
        self.low_bounds = maze.robot.get_joint_lower_bounds()
        self.high_bounds = maze.robot.get_joint_higher_bounds()

        relevant_entries = self.sampler.synthesize_sampling_distributions(maze, start, goal)

        # self.env.utils.visualize_nodes_global(None, tmp_occ_grid, viz_q, show=False, save=True, file_name=osp.join(CUR_DIR, f"{name}.png"))


test_env_dir = osp.join(CUR_DIR, "../../dataset/fetch_11d/gibson/mytest/Collierville")

similarity_model_path = osp.join(CUR_DIR, "models/fire.pt")

env = Fetch11DEnv(gui=False)
env.clear_obstacles()
env.load_occupancy_grid(env.utils.get_occ_grid(test_env_dir))
env.load_mesh(env.utils.get_mesh_path(test_env_dir))
planner = NeuralPlannerFire(env, similarity_model_path)

G = env.utils.get_prm(test_env_dir)
free_nodes = [n for n in G.nodes() if not G.nodes[n]["col"]]
start_node = random.choice(free_nodes)
start = env.utils.node_to_numpy(G, start_node)
goal_node = random.choice(free_nodes)
goal = env.utils.node_to_numpy(G, goal_node)

planner.solve(env, start, goal)
