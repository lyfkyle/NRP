import os.path as osp
import json
import networkx as nx
from multiprocessing import Process

from nrp.env.fetch_11d import utils

CUR_DIR = osp.dirname(osp.abspath(__file__))

# env_dirs = [osp.join(CUR_DIR, f"../env/fetch_11d/dataset/gibson_01/test/Ihlen")]
# test_env_dirs = [
#     '/data/yunlu/work/whole-body-motion-planning/nrp/env/fetch_11d/dataset/gibson/test/Wiconisco',
#     '/data/yunlu/work/whole-body-motion-planning/nrp/env/fetch_11d/dataset/gibson/test/Markleeville',
#     '/data/yunlu/work/whole-body-motion-planning/nrp/env/fetch_11d/dataset/gibson/test/Azusa',
#     '/data/yunlu/work/whole-body-motion-planning/nrp/env/fetch_11d/dataset/gibson/test/Corozal',
#     '/data/yunlu/work/whole-body-motion-planning/nrp/env/fetch_11d/dataset/gibson/test/Ihlen'
# ]

# env_dirs = utils.TEST_ENV_DIRS
# env_dirs = utils.TRAIN_ENV_DIRS

def calculate_criticality(env_dirs, env_idx):
    env_dir = env_dirs[env_idx]
    G = utils.get_prm(env_dir)
    print(G.number_of_nodes())

    criticality = nx.betweenness_centrality(G, weight="weight")
    with open(osp.join(env_dir, "criticality.json"), "w") as f:
        json.dump(criticality, f)

    criticality_list = [(criticality[n], list(utils.node_to_numpy(G, n))) for n in G.nodes()]
    criticality_list.sort(key=lambda item: item[0], reverse=True)
    critical_node_poss = [x[1] for x in criticality_list[:10]]

    mesh_path = utils.get_mesh_path(env_dir)
    occ_grid = utils.get_occ_grid(env_dir)
    # env.clear_obstacles()
    # env.load_mesh(mesh_path)
    # env.load_occupancy_grid(occ_grid)
    utils.visualize_nodes_global(mesh_path, occ_grid, critical_node_poss, None, None, show=False, save=True, file_name=osp.join(env_dir, "critical_point.png"))

    with open(osp.join(env_dir, "critical_points.json"), "w") as f:
        json.dump(critical_node_poss, f)


if __name__ == "__main__":
    env_dirs = utils.TEST_ENV_DIRS
    # env_dirs = utils.TRAIN_ENV_DIRS

    # split into processes
    max_process_num = 25
    env_num = len(env_dirs)
    print(env_num)
    process_num = min(env_num, max_process_num)
    j = 0
    while j < env_num:
        processes = []
        for i in range(j, min(env_num, j + process_num)):
            p = Process(target=calculate_criticality, args=(env_dirs, i))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        j += process_num
