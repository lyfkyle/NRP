import os.path as osp
import sys
import os
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../../'))

import pickle
import json
from pathlib import Path
import random
from dataclasses import dataclass

CUR_DIR = osp.dirname(osp.abspath(__file__))
DATA_DIR = osp.join(CUR_DIR, "dataset/fire_database")
train_env_dir = osp.join(CUR_DIR, "../../dataset/fetch_11d/gibson/train")
in_data_dir = osp.join(CUR_DIR, "dataset/similarity_dataset")
out_data_dir = osp.join(CUR_DIR, "dataset/similarity_dataset2")
if not osp.exists(out_data_dir):
    os.mkdir(out_data_dir)

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


# data_cnt = 354672
data_cnt = 86932
print("start loading")
entries = []
for entry_idx in range(data_cnt):
    with open(osp.join(DATA_DIR, f"entry_{entry_idx}.pkl"), "rb") as f:
        entry = pickle.load(f)
    entries.append(entry)

train_env_dirs = []
for p in Path(train_env_dir).rglob('env_final.obj'):
    train_env_dirs.append(p.parent)

all_data = []
for train_env_dir in train_env_dirs:
    env_name = str(train_env_dir).split("/")[-1]
    env_entries = [e for e in entries if e.env_name == env_name]

    data_idx = 0
    while osp.exists(osp.join(in_data_dir, f"data_{env_name}_{data_idx}.pkl")):
        with open(osp.join(in_data_dir, f"data_{env_name}_{data_idx}.pkl"), "rb") as f:
            data = pickle.load(f)

        all_data.append(data)
        data_idx += 1

print(len(all_data))

for data_idx, data in enumerate(all_data):
    with open(osp.join(out_data_dir, f"data_{data_idx}.pkl"), "wb") as f:
        pickle.dump(data, f)


