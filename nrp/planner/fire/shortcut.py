import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../../'))

import random
import pickle

from nrp.env.fetch_11d.env import Fetch11DEnv

CUR_DIR = osp.dirname(osp.abspath(__file__))

def random_shortcut(env, traj, max_trial=100):
    new_traj = traj.copy()

    if env.utils.is_edge_free(env, traj[0], traj[-1]):
        return [traj[0], traj[-1]]

    for _ in range(max_trial):
        path_len = len(new_traj)
        p1 = random.randint(0, path_len - 1)
        p2 = random.randint(0, path_len - 1)

        if abs(p1-p2) < 2:
            continue

        if env.utils.is_edge_free(env, new_traj[p1], new_traj[p2]):
            del new_traj[p1+1:p2]

    print(f"shortcut: orig len {len(traj)}, new len {len(new_traj)}")
    return new_traj


if __name__ == '__main__':
    env = Fetch11DEnv(gui=False)
    data_dir = osp.join(CUR_DIR, "dataset/model_fire")
    new_data_dir = osp.join(CUR_DIR, "dataset/model_fire_shortcut")

    env_num = 25
    traj_num = 100
    for env_idx in range(env_num):
        for traj_idx in range(traj_num):
            print(f"shortcut: processing env {env_idx}, traj {traj_idx}")
            file_path = osp.join(data_dir, "data_{}_{}.pkl".format(env_idx, traj_idx))
            with open(file_path, 'rb') as f:
                env_dir, traj = pickle.load(f)[0]

            occ_grid = env.utils.get_occ_grid(env_dir)
            mesh = env.utils.get_mesh_path(env_dir)
            env.clear_obstacles()
            env.load_mesh(mesh)
            env.load_occupancy_grid(occ_grid)

            interpolated = env.utils.interpolate(traj, step_size = 2.0)
            # env.utils.visualize_nodes_global(env.mesh, env.occ_grid, interpolated, show=False, save=True, file_name=osp.join("orig.png"))
            simplified_traj = random_shortcut(env, interpolated)
            # interpolated = env.utils.interpolate(simplified_traj)
            # env.utils.visualize_nodes_global(env.mesh, env.occ_grid, interpolated, show=False, save=True, file_name=osp.join("simplified.png"))

            file_path = osp.join(new_data_dir, "data_{}_{}.pkl".format(env_idx, traj_idx))
            with open(file_path, 'wb') as f:
                pickle.dump([(env_dir, simplified_traj)], f)