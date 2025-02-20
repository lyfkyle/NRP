import shutil
import os
import os.path as osp
from pathlib import Path

CUR_DIR = osp.dirname(osp.abspath(__file__))
data_dir = os.path.join(CUR_DIR, "../wbmp8dof_v2/dataset/gibson")

env_dirs = []
for path in Path(data_dir).rglob('env_small.obj'):
    env_dirs.append(path.parent)

for env_dir in env_dirs:
    new_env_dir = str(env_dir).replace("wbmp8dof_v2", "wbmp8dof")
    # os.makedirs(new_env_dir)
    # shutil.copy(os.path.join(env_dir, "env_small.obj"), os.path.join(new_env_dir, "env_small.obj"))
    # shutil.copy(os.path.join(env_dir, "occ_grid_small.txt"), os.path.join(new_env_dir, "occ_grid_small.txt"))
    # shutil.copy(os.path.join(env_dir, "dense_free_small.png"), os.path.join(new_env_dir, "dense_free_small.png"))
    # shutil.copy(os.path.join(env_dir, "dense_g_small.graphml"), os.path.join(new_env_dir, "dense_g_small.graphml"))

    assert os.path.exists(os.path.join(new_env_dir, "dense_g_small.graphml"))