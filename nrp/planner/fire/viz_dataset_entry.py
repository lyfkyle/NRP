import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../../'))

import numpy as np
import pickle
from dataclasses import dataclass
import random

from env.fetch_11d.maze import Fetch11DEnv
from fire import Fire

CUR_DIR = osp.dirname(osp.abspath(__file__))
DATA_DIR = osp.join(CUR_DIR, "dataset/fire_database")
train_env_dir = osp.join(CUR_DIR, "../../dataset/fetch_11d/gibson/train")

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


if __name__ == '__main__':
    env = Fetch11DEnv(gui=False)
    fire = Fire(env)

    data_dir = osp.join(CUR_DIR, "dataset/model_fire_shortcut")

    data_cnt = 355356

    for idx in range(10):
        entry_idx = random.randint(0, data_cnt)

        with open(osp.join(DATA_DIR, f"entry_{entry_idx}.pkl"), "rb") as f:
            entry = pickle.load(f)

        env_dir = osp.join(train_env_dir, entry.env_name)
        occ_grid = env.utils.get_occ_grid(env_dir)
        mesh = env.utils.get_mesh_path(env_dir)
        env.clear_obstacles()
        env.load_occupancy_grid(occ_grid)
        env.load_mesh(mesh)
        fire.visualize_primitives(entry.occ_grid, entry.center_pos, entry.q, name=f"{idx}")
        env.utils.visualize_nodes_global(mesh, occ_grid, [entry.q], show=False, save=True, file_name=osp.join(CUR_DIR, f"{idx}_full.png"))