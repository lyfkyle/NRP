
import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../../'))

import numpy as np
import pickle
from dataclasses import dataclass

from nrp.env.fetch_11d.env import Fetch11DEnv
from planner.fire.shortcut import random_shortcut

CUR_DIR = osp.dirname(osp.abspath(__file__))
DATA_DIR = osp.join(CUR_DIR, "dataset/fire_database")


DEBUG = False
PRIM_SIZE = 20

@dataclass
class FireEntry:
    occ_grid: int
    center_pos: int
    q_target: int
    q_proj: int
    q: int
    prev_q: int
    next_q: int
    proj_num: int
    env_name: str = ""


class Fire():
    def __init__(self, env):
        self.env = env
        self.prim_size = PRIM_SIZE
        self.occ_grid_res = 0.1
        self.entries = []
        self.entry_idx = 0
        self.sigma = 0.2

    def process_experience(self, trajs, occ_grid, mesh, env_name=""):
        self.env.clear_obstacles()
        self.env.load_mesh(mesh)
        self.env.load_occupancy_grid(occ_grid)

        primitives = self.extract_primitives(occ_grid)

        # debug primitive
        if DEBUG:
            print(len(primitives))
            for i, p in enumerate(primitives[:10]):
                self.visualize_primitives(p, name=f"prim_{i}")
            input()

        for i, traj in enumerate(trajs):
            # debug
            # self.env.utils.visualize_nodes_global(mesh, occ_grid, traj, show=False, save=True, file_name=osp.join(CUR_DIR, "cur_traj.png"), gui=True)
            # nearby_primitives = self._get_nearby_primitives(primitives, traj[6])
            # q_proj = self._get_fetch_projection(traj[6])
            # for i, nearby_primitive in enumerate(nearby_primitives):
            #     print(traj[6], q_proj, nearby_primitive[1])
            #     self._visualize_primitives(nearby_primitive, traj[6], name = f"nearby_{i}")

            # return

            print(f"Fire, processing traj {i}/{len(trajs)}, traj_len : {len(traj)}")
            self.create_entries(primitives, traj, env_name=env_name)

    def extract_primitives(self, occ_grid):
        primitives = []
        for i in range(0, occ_grid.shape[0] - self.prim_size, int(self.prim_size / 4)):
            for j in range(0, occ_grid.shape[1] - self.prim_size, int(self.prim_size / 4)):
                for k in range(0, occ_grid.shape[2], self.prim_size):
                    local_occ_grid = occ_grid[i:i+self.prim_size, j:j+self.prim_size, k:k+self.prim_size]
                    if local_occ_grid.max() > 0:
                        primitives.append((local_occ_grid, np.array([i+self.prim_size/2, j+self.prim_size/2, k+self.prim_size/2]) * self.occ_grid_res))

        return primitives

    def create_entries(self, primitives, traj, env_name=""):
        # print("Fire: Processing path into entries: ", traj)

        q_target = traj[-1]
        for i, q in enumerate(traj):
            if i == 0 or i == len(traj) - 1:  # ignore start and goal
                continue

            prev_q = traj[i - 1] if i > 0 else q
            next_q = traj[i + 1] if i < len(traj) - 1 else q
            q_proj = self.get_fetch_projection(q)

            for prim_idx, primitive in enumerate(primitives):
                res, proj_num = self._contains(primitive, q_proj)
                if res:
                    q_proj_normalized = self.normalize_projections(primitive, q_proj)
                    entry = FireEntry(primitive[0], primitive[1], q_target, q_proj_normalized, q, prev_q, next_q, proj_num, env_name=env_name)

                    if entry.occ_grid.shape[0] == self.prim_size and entry.occ_grid.shape[1] == self.prim_size and entry.occ_grid.shape[2] == self.prim_size:
                        with open(osp.join(DATA_DIR, f"entry_{self.entry_idx}.pkl"), "wb") as f:
                            pickle.dump(entry, f)
                            self.entry_idx += 1

                    # self.entries.append(entry)
                    # if DEBUG:
                    #     self.visualize_primitives(primitive[0], primitive[1], q, f"q{i}_prim{prim_idx}")

    def _contains(self, primitive, projections):
        primitive_center = np.array(primitive[1])
        primitivie_half_size = self.prim_size * self.occ_grid_res / 2
        for proj_num, projection in enumerate(projections):
            if (np.absolute(np.array(projection) - primitive_center) <= primitivie_half_size).all():
                return True, proj_num

        return False, None

    def get_fetch_projection(self, q, fetch_radius=0.28):
        projections = self.env.get_link_positions(q).reshape(-1, 3).tolist()

        # Append base proections
        projections.append([q[0] - fetch_radius, q[1], 0.0])
        projections.append([q[0] + fetch_radius, q[1], 0.0])
        projections.append([q[0], q[1] - fetch_radius, 0.0])
        projections.append([q[0], q[1] + fetch_radius, 0.0])

        # print(q, projections)
        return projections

    def normalize_projections(self, primitive, projections):
        primitive_center = np.array(primitive[1]).reshape(1, 3)
        return np.array(projections) - primitive_center

    def _sample_valid(self, prev_q, q, next_q):
        return self.env.utils.is_edge_free(self.env, prev_q, q) and self.env.utils.is_edge_free(self.env, q, next_q)

    def visualize_primitives(self, prim_occ_grid, prim_center_pos, q=None, q_target=None, name=""):
        tmp_occ_grid = np.zeros_like(self.env.occ_grid)
        c = (prim_center_pos / self.occ_grid_res).astype(np.int32)
        half_size = int(self.prim_size / 2)
        print(c, half_size)
        for i in range(prim_occ_grid.shape[0]):
            for j in range(prim_occ_grid.shape[1]):
                for k in range(prim_occ_grid.shape[2]):
                    tmp_occ_grid[c[0] - half_size + i, c[1] - half_size + j, c[2] - half_size + k] = prim_occ_grid[i, j, k]

        tmp_occ_grid[c[0] - half_size, :, 0] = 1
        tmp_occ_grid[c[0] + half_size, :, 0] = 1
        tmp_occ_grid[:, c[1] - half_size, 0] = 1
        tmp_occ_grid[:, c[1] + half_size, 0] = 1

        print(tmp_occ_grid.max())
        if q is not None:
            viz_q = [q]
        else:
            viz_q = []

        viz_start_pos = [c[0], c[1], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0,0]
        self.env.utils.visualize_nodes_global(None, tmp_occ_grid, viz_q, start_pos=viz_start_pos, goal_pos=q_target, show=False, save=True, file_name=osp.join(CUR_DIR, f"{name}.png"))

    def _get_nearby_primitives(self, primitives, q):
        nearby_primitives = []
        for primitive in primitives:
            if (np.absolute(primitive[1][:2] - q[:2]) < 0.5).all():
                nearby_primitives.append(primitive)
        return nearby_primitives

    def load_existing_database(self, database_path):
        data_cnt = 86932
        print("start loading")
        entries = []
        for entry_idx in range(data_cnt):
            with open(osp.join(database_path, f"entry_{entry_idx}.pkl"), "rb") as f:
                entry = pickle.load(f)
            entries.append(entry)
        self.database_entries = entries
        return entries


if __name__ == '__main__':
    env = Fetch11DEnv(gui=False)
    fire = Fire(env)

    data_dir = osp.join(CUR_DIR, "dataset/model_fire_shortcut")

    env_num = 25
    traj_num = 20
    for env_idx in range(env_num):
        trajs = []
        for traj_idx in range(traj_num):
            file_path = osp.join(data_dir, "data_{}_{}.pkl".format(env_idx, traj_idx))
            with open(file_path, 'rb') as f:
                env_dir, traj = pickle.load(f)[0]
                trajs.append(traj)

        env_name = str(env_dir).split("/")[-1]
        print(env_dir, env_name)
        occ_grid = env.utils.get_occ_grid(env_dir)
        mesh = env.utils.get_mesh_path(env_dir)
        fire.process_experience(trajs, occ_grid, mesh, env_name=env_name)

        # print(fire.entries)