import os.path as osp
import os
import random
import pickle
from collections import defaultdict


from nrp.env.fetch_11d import utils
from nrp.env.fetch_11d.maze import Fetch11DEnv

CUR_DIR = osp.dirname(osp.abspath(__file__))
NUM_OF_COL_SAMPLE = 100
LOCAL_ENV_SIZE = 2.0
PATH_LEN_DIFF_THRESHOLD = 2.0
PROCESS_NUM = 40

env = Fetch11DEnv(gui=False)


def convert(pos):
    pos[0] += 2
    pos[1] += 2


# constants
robot_dim = 11
linkpos_dim = 24
occ_grid_dim = 40
goal_dim = robot_dim + linkpos_dim + 1
model_name = "fetch_11d"

# train_env_dir = osp.join(ROOT_DIR, "env/fetch_11d/dataset/gibson_01/train")
# train_env_dirs = []
# for p in Path(train_env_dir).rglob("env.obj"):
#     train_env_dirs.append(p.parent)
data_dir = osp.join(CUR_DIR, "dataset/train_01_out_g")
# output dir
output_dir = osp.join(CUR_DIR, "dataset/train_01_out_g_viz")
if not os.path.exists(output_dir):
    os.mkdir(output_dir)


datas = []
for filename in os.listdir(data_dir):
    f = os.path.join(data_dir, filename)
    if filename.endswith("pkl"):
        datas.append(filename)

env_path_cnt = defaultdict(int)

# for i, data in enumerate(datas):
for _ in range(5):
    i = random.randint(0, len(datas))

    # print(data)

    file_path = osp.join(data_dir, f"data_{i}.pkl")
    with open(file_path, "rb") as f:
        loaded_data = pickle.load(f)

    occ_grid, local_start_pos, local_goal_pos, expert_path, env_idx, global_start_pos = loaded_data
    global_goal_pos = utils.local_to_global(local_goal_pos, global_start_pos)
    sampled_pos = expert_path[1].copy()
    global_sampled_pos = utils.local_to_global(sampled_pos, global_start_pos)

    convert(sampled_pos)
    convert(local_start_pos)
    convert(local_goal_pos)
    for p in expert_path:
        convert(p)

    expert_path = utils.interpolate(expert_path)
    utils.visualize_nodes_global(
        None,
        occ_grid,
        [],
        local_start_pos,
        local_goal_pos,
        sampled_pos,
        show=False,
        save=True,
        file_name=osp.join(output_dir, "viz_path_{}.png".format(i)),
    )

    env_dir = utils.TRAIN_ENV_DIRS[env_idx]
    utils.visualize_nodes_global(
        utils.get_mesh_path(env_dir),
        utils.get_occ_grid(env_dir),
        [],
        global_start_pos,
        global_goal_pos,
        global_sampled_pos,
        show=False,
        save=True,
        file_name=osp.join(output_dir, "viz_path_{}_global.png".format(i)),
    )