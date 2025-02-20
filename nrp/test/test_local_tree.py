import os
import os.path as osp
from collections import defaultdict
import random
from datetime import datetime
import json
import numpy as np
import argparse
import networkx as nx
import math
import itertools
from sklearn.neighbors import NearestNeighbors
import random
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from multiprocessing import Process, Manager

from nrp import ROOT_DIR
from nrp.env.fetch_11d.maze import Fetch11DEnv
from nrp.env.fetch_11d import utils
from nrp.planner.local_sampler_g.local_sampler_g import LocalNeuralExpander11D
from nrp.env.fetch_11d import prm_utils

PRM_CONNECT_RADIUS = 2.0
CUR_DIR = osp.dirname(osp.abspath(__file__))


def merge_images(file1, file2, file3):
    """Merge two images into one, displayed side by side
    :param file1: path to first image file
    :param file2: path to second image file
    :return: the merged Image object
    """
    image1 = Image.open(file1)
    image2 = Image.open(file2)
    image3 = Image.open(file3)

    (width, height) = image1.size
    image2 = image2.resize((width, height))
    image3 = image3.resize((width, height))

    result_width = width * 3
    result_height = height

    result = Image.new("RGB", (result_width, result_height))
    result.paste(im=image1, box=(0, 0))
    result.paste(im=image2, box=(width, 0))
    result.paste(im=image3, box=(width * 2, 0))

    return result


def get_nearest_neighbors(G, v, distance_func, n_neighbors: int = 1):
    all_nodes = list(G.nodes())
    all_node_poss = [utils.node_to_numpy(G, n) for n in all_nodes]

    n_neighbors = min(n_neighbors, len(all_node_poss))

    all_vertices = np.array(all_node_poss)
    v = np.array(v).reshape(1, -1)

    # NOTE: here +1 is to remove itself.
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm="ball_tree", metric=distance_func).fit(all_vertices)
    distances, indices = nbrs.kneighbors(v)
    # print("indices {}".format(indices))
    nbr_nodes = [all_nodes[i] for i in indices.ravel()]
    return nbr_nodes[1:]


def process_env(env_dirs, env_idx, total_col_status, generate=False, critical=False):
    local_sampler = LocalNeuralExpander11D(env, dim, occ_grid_dim, model_path, device="cuda")

    env_dir = env_dirs[env_idx]
    if critical:
        local_tree_test_dir = osp.join(env_dir, "local_tree_test_critical")
        if not os.path.exists(local_tree_test_dir):
            os.makedirs(local_tree_test_dir)

        with open(osp.join(env_dir, "critical_points.json"), "r") as f:
            critical_points = json.load(f)
    else:
        local_tree_test_dir = osp.join(env_dir, "local_tree_test")
        if not os.path.exists(local_tree_test_dir):
            os.makedirs(local_tree_test_dir)

    mesh_path = utils.get_mesh_path(env_dir)
    occ_grid = utils.get_occ_grid(env_dir)
    orig_G = utils.get_prm(env_dir)

    free_nodes = utils.get_free_nodes(orig_G)

    for start_idx in tqdm(range(1)):
        env.clear_obstacles()
        env.load_mesh(mesh_path)
        env.load_occupancy_grid(occ_grid)

        if generate:
            # Random start_pos
            if not critical:
                idx = random.randint(0, len(free_nodes) - 1)
                start_node = free_nodes[idx]
                start_pos = utils.node_to_numpy(orig_G, start_node)
                with open(osp.join(env_dir, "local_start_pos_idx.json"), "w") as f:
                    json.dump([idx], f)

            # manual start_pos
            # start_pos = [3.8614785267395497, 14.246539012849077, 0.5567384275465974, 0.2018445170299941, 0.6209316222927534, 0.6815991004423627, -2.739821649091807, -0.6093650433908302, 2.6945333703853915, 0.31623371346259344, 1.6331295139934348]
            # for node in orig_G.nodes():
            #     node_pos = utils.node_to_numpy(orig_G, node)
            #     if np.allclose(node_pos, start_pos):
            #         start_node = node
            #         break

            # critical start pos
            if critical:
                start_pos = critical_points[start_idx]
                for node in orig_G.nodes():
                    node_pos = utils.node_to_numpy(orig_G, node)
                    if np.allclose(node_pos, start_pos):
                        start_node = node
                        break

            print(start_pos)
            # print(goal_pos)
            # start_pos[0] = 2.5
            # start_pos[1] = 7
            # start_pos[2] = 0

            tmp_mesh_file_name = f"tmp_{env_idx}_{start_idx}.obj"
            global_occ_grid, new_mesh_path = env.clear_obstacles_outside_local_occ_grid(
                start_pos, 2, tmp_mesh_file_name
            )

            # orig_G = utils.get_prm(env_dir)
            # G, outside_nodes = generate_new_prm(
            #     orig_G,
            #     env,
            #     start_node,
            #     local_env_size,
            # )
            G_without_goal, outside_nodes = prm_utils.generate_new_prm(
                orig_G,
                env,
                start_node,
                local_env_size=2,
            )
            nx.write_graphml(G_without_goal, osp.join(local_tree_test_dir, "dense_g_local.graphml"))
        else:
            if not critical:
                with open(osp.join(local_tree_test_dir, "local_start_pos_idx.json"), "r") as f:
                    idx = json.load(f)[0]
                start_node = free_nodes[idx]
                start_pos = utils.node_to_numpy(orig_G, start_node)
            else:
                start_pos = critical_points[start_idx]
                for node in orig_G.nodes():
                    node_pos = utils.node_to_numpy(orig_G, node)
                    if np.allclose(node_pos, start_pos):
                        start_node = node
                        break

            G_without_goal = nx.read_graphml(osp.join(local_tree_test_dir, "dense_g_local.graphml"))

        relative_path_optimality = []
        expert_wp_tree = nx.Graph()
        sampled_wp_tree = nx.Graph()
        expert_wp_tree_s_node = "n{}".format(expert_wp_tree.number_of_nodes() + 1)
        expert_wp_tree.add_node(expert_wp_tree_s_node, coords=",".join(map(str, start_pos)), col=False)
        sampled_wp_tree_s_node = "n{}".format(sampled_wp_tree.number_of_nodes() + 1)
        sampled_wp_tree.add_node(sampled_wp_tree_s_node, coords=",".join(map(str, start_pos)), col=False)

        sampled_goal_indices = []
        for goal_idx in range(50):
            if generate:
                idx = random.randint(0, len(free_nodes) - 1)
                sampled_goal_indices.append(idx)
            else:
                with open(osp.join(local_tree_test_dir, "local_goal_pos_idx.json"), "r") as f:
                    idx = json.load(f)[goal_idx]

            goal_node = free_nodes[idx]
            goal_pos = utils.node_to_numpy(orig_G, goal_node)

            local_occ_grid = env.get_local_occ_grid(start_pos, local_env_size=2)
            local_start_pos = utils.global_to_local(start_pos, start_pos)
            local_goal_pos = utils.global_to_local(goal_pos, start_pos)
            local_path = local_sampler.neural_expand(local_start_pos, local_goal_pos, local_occ_grid)
            global_path = [utils.local_to_global(p, start_pos) for p in local_path]

            # interpolate between start and waypoint
            waypoint_pos = global_path[1]
            path_to_wp = utils.rrt_extend_path(env, global_path[0:2], intermediate=True)
            reached = np.allclose(np.array(waypoint_pos), np.array(path_to_wp[-1]))
            if reached:
                total_col_status.append(0)
            else:
                total_col_status.append(1)
            # print("path_to_wp_res: ", waypoint_pos, path_to_wp[-1], reached)

            # add goal node to G
            if generate:
                cur_G = G_without_goal.copy()
                cur_G, goal_node = prm_utils.add_goal_node_to_prm(cur_G, env, start_node, goal_pos, local_env_size=2)
                nx.write_graphml(cur_G, osp.join(local_tree_test_dir, f"dense_g_local_{goal_idx}.graphml"))
            else:
                cur_G = nx.read_graphml(osp.join(local_tree_test_dir, f"dense_g_local_{goal_idx}.graphml"))
                for node in cur_G.nodes():
                    node_pos = utils.node_to_numpy(cur_G, node)
                    if np.allclose(np.array(node_pos), np.array(goal_pos)):
                        goal_node = node
                        break

            # utils.visualize_tree_simple(global_occ_grid, cur_G, start_pos, goal_pos, show=False, save=True, file_name=osp.join(res_dir, "tree_{}.png".format(theta)))

            # connect waypoint to nearest edges
            if reached:
                wp_node = "n{}".format(cur_G.number_of_nodes() + 1)
                cur_G.add_node(wp_node, coords=",".join(map(str, waypoint_pos)), col=False)
                nbr_nodes = get_nearest_neighbors(cur_G, waypoint_pos, utils.calc_edge_len, n_neighbors=10)
                for nbr_node in nbr_nodes:
                    nbr_pos = utils.node_to_numpy(cur_G, nbr_node)
                    if utils.is_edge_free(env, waypoint_pos, nbr_pos):
                        cur_G.add_edge(wp_node, nbr_node, weight=utils.calc_edge_len(waypoint_pos, nbr_pos))
                try:
                    expert_node_wp_to_goal = nx.shortest_path(cur_G, wp_node, goal_node)
                    expert_path_wp_to_goal = [utils.node_to_numpy(cur_G, n) for n in expert_node_wp_to_goal]
                    expert_path_wp_to_goal = utils.interpolate(expert_path_wp_to_goal)
                    final_global_path = path_to_wp + expert_path_wp_to_goal
                except Exception as e:
                    final_global_path = path_to_wp
            else:
                final_global_path = path_to_wp

            # utils.visualize_nodes_global(new_mesh_path, global_occ_grid, global_path, start_pos, goal_pos, show=False, save=True, file_name=osp.join(CUR_DIR, "visualize_local_sampler/viz_{}.png".format(theta)))
            # utils.visualize_nodes_global(new_mesh_path, global_occ_grid, final_global_path, start_pos, goal_pos, sample_pos=waypoint_pos, show=False, save=True, file_name=osp.join(res_dir, "viz_{}.png".format(theta)))

            try:
                expert_node_path = nx.shortest_path(cur_G, start_node, goal_node)
                expert_path = [utils.node_to_numpy(cur_G, n) for n in expert_node_path]
                expert_wp_pos = expert_path[1]
                expert_path = utils.interpolate(expert_path)
            except Exception as e:
                print("no expert path to sampled goal pos")
                continue

            # utils.visualize_nodes_global(new_mesh_path, global_occ_grid, expert_path, start_pos, goal_pos, show=False, save=True, file_name=osp.join(res_dir, f"viz_expert_{env_dir}_{goal_idx}.png"))
            wp_node = "n{}".format(sampled_wp_tree.number_of_nodes() + 1)
            sampled_wp_tree.add_node(wp_node, coords=",".join(map(str, path_to_wp[-1])), col=False)
            sampled_wp_tree.add_edge(sampled_wp_tree_s_node, wp_node)
            if not reached:
                sampled_wp_node = "n{}".format(sampled_wp_tree.number_of_nodes() + 1)
                sampled_wp_tree.add_node(sampled_wp_node, coords=",".join(map(str, waypoint_pos)), col=True)
                sampled_wp_tree.add_edge(wp_node, sampled_wp_node)

            expert_wp_pos = utils.expand_until_local_env_edge(start_pos, expert_wp_pos)
            expert_wp_node = "n{}".format(expert_wp_tree.number_of_nodes() + 1)
            expert_wp_tree.add_node(expert_wp_node, coords=",".join(map(str, expert_wp_pos)), col=False)
            expert_wp_tree.add_edge(expert_wp_tree_s_node, expert_wp_node)

            optimal_path_len = utils.calc_path_len(expert_path)
            if np.allclose(np.array(final_global_path[-1]), np.array(goal_pos)):
                final_path_len = utils.calc_path_len(final_global_path)
                relative_path_len = optimal_path_len / final_path_len
            else:
                final_path_len = float("inf")
                relative_path_len = 0

            relative_path_optimality.append(relative_path_len)
            print(final_path_len, optimal_path_len, relative_path_len)

        if generate:
            with open(osp.join(local_tree_test_dir, "local_goal_pos_idx.json"), "w") as f:
                json.dump(sampled_goal_indices, f)

            continue

        print(np.mean(np.array(relative_path_optimality)))

        img1 = osp.join(res_dir, f"expert_wp_tree_{env_idx}_{start_idx}.png")
        img2 = osp.join(res_dir, f"sampled_wp_tree_{env_idx}_{start_idx}.png")
        img3 = osp.join(res_dir, f"sampled_wp_tree_{env_idx}_{start_idx}_with_col_edges.png")
        utils.visualize_tree_simple(occ_grid, expert_wp_tree, start_pos, None, show=False, save=True, file_name=img1)
        utils.visualize_tree_simple(occ_grid, sampled_wp_tree, start_pos, None, show=False, save=True, file_name=img2)
        utils.visualize_tree_simple(
            occ_grid, sampled_wp_tree, start_pos, None, draw_col_edges=True, show=False, save=True, file_name=img3
        )

        res_img = merge_images(img1, img2, img3)
        res_img.save(os.path.join(res_dir, f"tree_compare_{env_idx}_{start_idx}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--model", default="")
    parser.add_argument("--env", default="fetch_11d")
    parser.add_argument("--name", default="")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--critical", action="store_true")
    args = parser.parse_args()

    train_env_dirs = utils.TRAIN_ENV_DIRS
    test_env_dirs = utils.TEST_ENV_DIRS
    # test_env_dirs = [
    #     '/data/yunlu/work/whole-body-motion-planning/nrp/env/fetch_11d/dataset/gibson/test/Wiconisco',
    #     '/data/yunlu/work/whole-body-motion-planning/nrp/env/fetch_11d/dataset/gibson/test/Markleeville',
    #     '/data/yunlu/work/whole-body-motion-planning/nrp/env/fetch_11d/dataset/gibson/test/Azusa',
    #     '/data/yunlu/work/whole-body-motion-planning/nrp/env/fetch_11d/dataset/gibson/test/Corozal',
    #     '/data/yunlu/work/whole-body-motion-planning/nrp/env/fetch_11d/dataset/gibson/test/Ihlen'
    # ]

    env = Fetch11DEnv(gui=False)

    dim = 11
    occ_grid_dim = [40, 40, 20]
    local_env_size = 2

    if args.name == "":
        name = datetime.now().timestamp()
    else:
        name = args.name

    # model_path = osp.join(CUR_DIR, "../planner/local_sampler_g/models/fetch_11d/cvae_sel.pt")
    model_path = osp.join(CUR_DIR, "../train/fetch_11d/models/sampler_g_01_critical.pt")
    res_dir = osp.join(CUR_DIR, f"test_local_tree/{name}")
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    max_process_num = 20

    # --------- Train env --------
    # split into processes
    env_num = len(train_env_dirs)
    print(env_num)
    process_num = min(env_num, max_process_num)

    total_col_status = Manager().list()
    j = 0
    while j < env_num:
        processes = []
        for i in range(j, min(env_num, j + process_num)):
            p = Process(target=process_env, args=(train_env_dirs, i, total_col_status, args.generate, args.critical))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        j += process_num

    total_col_status = np.array(total_col_status)
    col_rate = np.sum(total_col_status) / len(total_col_status)
    print(f"Col rate: {col_rate}")

    # --------- test env --------
    # env_num = len(test_env_dirs)
    # print(env_num)
    # process_num = min(env_num, max_process_num)

    # total_col_status = Manager().list()
    # j = 0
    # while j < env_num:
    #     processes = []
    #     for i in range(j, min(env_num, j + process_num)):
    #         p = Process(target=process_env, args=(test_env_dirs, i, total_col_status))
    #         p.start()
    #         processes.append(p)

    #     for p in processes:
    #         p.join()

    #     j += process_num

    # total_col_status = np.array(total_col_status)
    # col_rate = np.sum(total_col_status) / len(total_col_status)
    # print(f"Col rate: {col_rate}")

    # process_env(test_env_dirs, 4, total_col_status)
