import os.path as osp
import sys
import os
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../../'))

import numpy as np
import pickle
from dataclasses import dataclass
from pathlib import Path
import torch.multiprocessing as mp
import json
import random

from env.fetch_11d.maze import Fetch11DEnv
from fire import Fire

CUR_DIR = osp.dirname(osp.abspath(__file__))
DATA_DIR = osp.join(CUR_DIR, "dataset/fire_database")
train_env_dir = osp.join(CUR_DIR,   "../../dataset/fetch_11d/gibson/train")
out_data_dir = osp.join(CUR_DIR, "dataset/similarity_dataset")

if not osp.exists(out_data_dir):
    os.mkdir(out_data_dir)


prim_size = 20
occ_grid_res = 0.1
sigma = 0.2
# data_cnt = 354672
data_cnt = 86932

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

def is_sample_valid(env, prev_q, q, next_q):
    return env.utils.is_edge_free(env, prev_q, q) and env.utils.is_edge_free(env, q, next_q)

def create_dataset(env_entries, env_name, data_counter):
    print(f"create data for {env_name}")
    # self.similar_pairs = []
    # self.disimilar_pairs = []
    env = Fetch11DEnv(gui=False)
    # data_cnt = 0

    # env_entries = [e for e in entries if e.env_name == env_name]

    env_dir = osp.join(train_env_dir, env_name)
    occ_grid = env.utils.get_occ_grid(env_dir)
    mesh = env.utils.get_mesh_path(env_dir)
    env.clear_obstacles()
    env.load_occupancy_grid(occ_grid)
    env.load_mesh(mesh)

    pos_res = []
    neg_res = []
    tried = set()
    while len(pos_res) < 2000:
        if len(tried) == len(env_entries):
            print(len(tried))
            break

        idx1 = random.randint(0, len(env_entries)-1)
        if idx1 in tried:
            continue
        tried.add(idx1)

        e1 = env_entries[idx1]

    # for idx1, e1 in enumerate(env_entries):
        for idx2, e2 in enumerate(env_entries):
            if idx2 <= idx1:
                continue

            if idx2 % 1000 == 0:
                print(f"{env_name}: processing {idx1}-{idx2}, num_pos: {len(pos_res)}")

            is_similar = 0
            if e1.proj_num == e2.proj_num:  # same proj_num
                if np.linalg.norm(e1.center_pos - e2.center_pos) < prim_size * occ_grid_res:
                    if np.linalg.norm(np.array(e1.q) - np.array(e2.q)) < 11 * sigma:
                        for _ in range(100):
                            q_sample = np.random.normal(np.array(e2.q), sigma)
                            if is_sample_valid(env, e1.prev_q, q_sample, e1.next_q):
                                is_similar = 1
                                break

            # self.disimilar_pairs.append((e1, e2))
            if is_similar:
                pos_res.append([env_name, idx1, idx2, 1])
            else:
                neg_res.append([env_name, idx1, idx2, 0])

    if len(pos_res) > 2000:
        pos_res = pos_res[:2000]
    if len(neg_res) > 2000:
        random.shuffle(neg_res)
        neg_res = neg_res[:2000]

    res = pos_res + neg_res
    for data_idx, data in enumerate(res):
        with open(osp.join(out_data_dir, f"data_{env_name}_{data_idx}.pkl"), "wb") as f:
            pickle.dump(data, f)


if __name__ == '__main__':
    # env = Fetch11DEnv(gui=False)
    # fire = Fire(env)

    data_dir = osp.join(CUR_DIR, "dataset/model_fire_shortcut")

    print("start loading")
    entries = []
    for entry_idx in range(data_cnt):
        with open(osp.join(DATA_DIR, f"entry_{entry_idx}.pkl"), "rb") as f:
            entry = pickle.load(f)
        entries.append(entry)

        # env_dir = osp.join(train_env_dir, entry.env_name)
        # occ_grid = env.utils.get_occ_grid(env_dir)
        # mesh = env.utils.get_mesh_path(env_dir)
        # env.clear_obstacles()
        # env.load_occupancy_grid(occ_grid)
        # env.load_mesh(mesh)
        # fire.visualize_primitives(entry.occ_grid, entry.center_pos, entry.q, name=f"{idx}")
        # env.utils.visualize_nodes_global(mesh, occ_grid, [entry.q], show=False, save=True, file_name=osp.join(CUR_DIR, f"{idx}_full.png"))

    print("finish loading")

    process_num = 25
    manager = mp.Manager()
    # dataset_dict = manager.dict()
    data_counter = manager.Value("i", 0)

    train_env_dirs = []
    for p in Path(train_env_dir).rglob('env_final.obj'):
        train_env_dirs.append(p.parent)

    j = 0
    while j < len(train_env_dirs):
        processes = []
        print("Running on env {} to {}".format(j, min(len(train_env_dirs), j + process_num)))
        for i in range(j, min(len(train_env_dirs), j + process_num)):
            env_name = str(train_env_dirs[i]).split("/")[-1]
            env_entries = [e for e in entries if e.env_name == env_name]
            p = mp.Process(target=create_dataset, args=(env_entries, env_name, data_counter), daemon=True)
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        j += process_num

    # create_dataset(entries)
