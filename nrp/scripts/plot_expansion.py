import os
import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))

import numpy as np
import argparse
import random
import json
from PIL import Image
import networkx as nx

from env.snake_8d.maze import Snake8DEnv
from env.fetch_11d.maze import Fetch11DEnv

CUR_DIR = osp.dirname(osp.abspath(__file__))


def merge_images(file1, file2, file3, file4):
    """Merge two images into one, displayed side by side
    :param file1: path to first image file
    :param file2: path to second image file
    :return: the merged Image object
    """
    image1 = Image.open(file1)
    image2 = Image.open(file2)
    image3 = Image.open(file3)
    image4 = Image.open(file4)

    (width, height) = image1.size
    image2 = image2.resize((width, height))
    image3 = image3.resize((width, height))
    image4 = image4.resize((width, height))

    result_width = width * 4
    result_height = height

    result = Image.new('RGB', (result_width, result_height))
    result.paste(im=image1, box=(0, 0))
    result.paste(im=image2, box=(width, 0))
    result.paste(im=image3, box=(width * 2, 0))
    result.paste(im=image4, box=(width * 3, 0))
    return result


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--env', default='snake_8d')
parser.add_argument('--name', default='ext_test')
parser.add_argument('--planner', default='nrp_d')
parser.add_argument('--replace', action="store_true")
args = parser.parse_args()

planner_res_dir = osp.join(CUR_DIR, "../test/eval_res/{}/{}/{}".format(args.env, args.name, args.planner))

if args.env == "snake_8d":
    env = Snake8DEnv(gui=False)
    dim = 8
    occ_grid_dim = [1, 40, 40]
    base_step_size = 0.5
elif args.env == "fetch_11d":
    env = Fetch11DEnv(gui=False)
    dim = 11
    occ_grid_dim = [40, 40, 20]
    base_step_size = 1.0



# for env_num in range(50, 250, 10):
envs = [3]
plan_nums = [i for i in range(20)] # 13
# plan_nums = [13]
ext_nums = [1]

for env_idx in envs:
    env.clear_obstacles()
    env_dir = osp.join(CUR_DIR, f"../env/{args.env}/dataset/{args.name}/{env_idx}")
    occ_grid = env.utils.get_occ_grid(env_dir)
    mesh_path = env.utils.get_mesh_path(env_dir)

    env.load_mesh(mesh_path)
    env.load_occupancy_grid(occ_grid, add_enclosing=True)

    with open(osp.join(env_dir, "start_goal.json")) as f:
        start_goal = json.load(f)

    for plan_num in plan_nums:
        env_data_dir = os.path.join(planner_res_dir, f"{env_idx}/{plan_num}")
        if not os.path.exists(env_data_dir):
            continue

        output_dir = env_data_dir
        for ext_num in ext_nums:
            extension_data_path = osp.join(env_data_dir, "extension_data_{}.json".format(ext_num))
            if osp.exists(extension_data_path):
                with open(extension_data_path, "r") as f:
                    extension_data = json.load(f)

                img1 = os.path.join(output_dir, "ext_{}_{}.png".format(env_idx, ext_num))
                img2 = os.path.join(output_dir, "path_intended_{}_{}.png".format(env_idx, ext_num))
                img3 = os.path.join(output_dir, "path_actual_full_{}_{}.png".format(env_idx, ext_num))
                img4 = os.path.join(output_dir, "tree_{}_{}.png".format(env_idx, ext_num))
                img5 = os.path.join(output_dir, "sl_path_actual_{}_{}.png".format(env_idx, ext_num))
                img6 = os.path.join(output_dir, "mixed_path_intended_{}_{}.png".format(env_idx, ext_num))
                img7 = os.path.join(output_dir, "mixed_path_actual_{}_{}.png".format(env_idx, ext_num))

                path_intended = extension_data["path_intended"]
                start_pos = path_intended[0]
                sampled_pos = path_intended[1]
                target_pos = path_intended[-1]

                # Expert base + arm sl
                ebsa_path = env.utils.get_ebsa_path(occ_grid, start_pos, target_pos)

                if ebsa_path:
                    # print(expert_path)
                    # ebsa_path_viz = env.utils.interpolate(ebsa_path, step_size=0.2)
                    # ebsa_path_viz = ebsa_path[::7] + [ebsa_path[-1]]
                    env.utils.visualize_nodes_global(mesh_path, occ_grid, ebsa_path, start_pos, target_pos, viz_edge=True, show=False, save=True, file_name=img6)

                    ebsa_path_actual = env.utils.rrt_extend_path(env, ebsa_path, step_size=0.2, intermediate=True)
                    env.utils.visualize_nodes_global(mesh_path, occ_grid, ebsa_path_actual, start_pos, target_pos, viz_edge=True, edge_path=ebsa_path,  show=False, save=True, file_name=img7)

                # local_occ_grid = maze.get_local_occ_grid(start_pos)
                # local_start_pos = utils.global_to_local(start_pos, start_pos)
                # local_target_pos = utils.global_to_local(target_pos, start_pos)
                # local_selected_pos = utils.global_to_local(path_intended[1], start_pos)
                env.utils.visualize_nodes_global(mesh_path, occ_grid, [], start_pos, target_pos, show=False, save=True, file_name=img1)

                path_intended_interpolated = env.utils.interpolate_base(path_intended, step_size=base_step_size)
                env.utils.visualize_nodes_global(mesh_path, occ_grid, path_intended_interpolated, start_pos, target_pos, sample_pos=sampled_pos, viz_edge=True, show=False, save=True, file_name=img2)

                # # path_actual = extension_data["path_actual"]
                # path_actual = [start_pos, path_intended[1], extension_data["path_actual"][-1]]
                # utils.visualize_nodes_global(occ_grid, path_actual, start_pos, target_pos, show=False, save=True, file_name=img3)

                # res_img = merge_images(img1, img2, img3)
                # res_img.save(os.path.join(output_dir, "path_compare_{}.png".format(ext_num)))

                # utils.visualize_nodes_global(occ_grid, [path_intended[1]], start_pos, target_pos, show=False, save=True, file_name=img1)

                # path_actual = env.utils.interpolate_base(extension_data["path_actual"], step_size=base_step_size)
                path_actual = env.utils.interpolate(extension_data["path_actual"])
                env.utils.visualize_nodes_global(mesh_path, occ_grid, path_actual, start_pos, target_pos, sample_pos=sampled_pos, viz_edge=True, edge_path=path_intended_interpolated, show=False, save=True, file_name=img3)

                sl_path_intended = env.utils.interpolate([start_pos, target_pos])
                sl_path_actual = env.utils.rrt_extend_intermediate(env, start_pos, target_pos, step_size=0.2)
                # sl_path_actual.append(sl_path_intended[len(sl_path_actual)])
                env.utils.visualize_nodes_global(mesh_path, occ_grid, sl_path_actual, start_pos, target_pos, viz_edge=True, edge_path=sl_path_intended, show=False, save=True, file_name=img5)

                # tree = nx.read_graphml(osp.join(env_data_dir, "tree_nodes_{}.graphml".format(ext_num - 1)))
                # tree_nodes = tree.nodes()
                # tree_nodes = [tuple(utils.string_to_numpy(x[1:-1])) for x in tree_nodes]
                # tree_nodes_base = [list(node)[:2] for node in tree_nodes]
                # print(len(tree_nodes_base))
                # utils.visualize_nodes_global(occ_grid, tree_nodes_base, start_goal[0], start_goal[1], show=False, save=True, file_name=img4)
                # env.utils.visualize_tree(mesh_path, occ_grid, tree, start_goal[0], target_pos=target_pos, string=True, show=False, save=True, file_name=img4)

                # ad-hoc
                # path_actual = env.utils.interpolate([start_pos, sampled_pos], step_size=1.0)
                # env.utils.visualize_nodes_global(mesh_path, occ_grid, path_actual, start_pos, target_pos, sample_pos=sampled_pos, viz_edge=True, show=False, save=True, file_name=os.path.join(output_dir, "actual_path_zoom_{}_{}.png".format(env_idx, ext_num)))
