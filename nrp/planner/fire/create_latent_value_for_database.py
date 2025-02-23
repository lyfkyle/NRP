import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../../'))

import numpy as np
import matplotlib.pyplot as plt
import math
import random
import torch
import pickle
import networkx as nx

# from planner.informed_rrt import InformedRRTStar
from nrp.env.fetch_11d.env import Fetch11DEnv
from planner.fire.model import FireModel
from planner.rrt import RRT
from planner.fire.fire import Fire, FireEntry

CUR_DIR = osp.dirname(osp.abspath(__file__))

device = "cuda"
similarity_model_path = osp.join(CUR_DIR, "models/fire.pt")
DATA_DIR = osp.join(CUR_DIR, "dataset/fire_database")
occ_grid_dim = 20

env = Fetch11DEnv(gui=False)
fire_database = Fire(env)
fire_database.load_existing_database(DATA_DIR)

fire_similarity_model = FireModel(11)
# self._fire_similarity_model = torch.jit.script(self._fire_similarity_model)
fire_similarity_model.load_state_dict(torch.load(similarity_model_path))
fire_similarity_model.eval()
print(device)
fire_similarity_model.to(device)

entry_latent_value = []
bs = 512
with torch.no_grad():
    for idx in range(0, len(fire_database.database_entries), bs):
        print(f"database: {idx}/{len(fire_database.database_entries)}, {len(entry_latent_value)}")
        q_targets = []
        center_poss = []
        q_projs = []
        occ_grids = []
        for i in range(idx, min(len(fire_database.database_entries), idx + bs)):
            q_targets.append(fire_database.database_entries[i].q_target)
            center_poss.append(fire_database.database_entries[i].center_pos)
            q_projs.append(fire_database.database_entries[i].q_proj)
            occ_grid_t = torch.tensor(fire_database.database_entries[i].occ_grid, dtype=torch.float32).view(occ_grid_dim, occ_grid_dim, occ_grid_dim)
            occ_grid_t = env.utils.add_pos_channels(occ_grid_t)
            occ_grids.append(occ_grid_t.cpu().numpy().tolist())

        cur_bs = len(q_targets)
        q_target_t = torch.tensor(q_targets, dtype=torch.float32, device=device).view(cur_bs, -1)
        center_pos_t = torch.tensor(center_poss, dtype=torch.float32, device=device).view(cur_bs, -1)
        q_proj_t = torch.tensor(q_projs, dtype=torch.float32, device=device).view(cur_bs, -1)  # flatten
        occ_grid_t = torch.tensor(occ_grids, dtype=torch.float32, device=device).view(cur_bs, 4, occ_grid_dim, occ_grid_dim, occ_grid_dim)

        # print(occ_grid_t.shape, center_pos_t.shape, q_target_t.shape, q_proj_t.shape)
        z = fire_similarity_model(occ_grid_t, center_pos_t, q_target_t, q_proj_t).cpu().numpy().tolist()

        entry_latent_value += z

with open(osp.join(DATA_DIR, "latent_value.pkl"), "wb") as f:
    pickle.dump(entry_latent_value, f)