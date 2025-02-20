import os.path as osp
import os
import shutil
import numpy as np
from PIL import Image

CUR_DIR = osp.dirname(osp.abspath(__file__))

def merge_images(files):
    """Merge two images into one, displayed side by side
    :param file1: path to first image file
    :param file2: path to second image file
    :return: the merged Image object
    """
    images = []
    for file in files:
        images.append(Image.open(file))

    image1 = images[0]
    (width, height) = image1.size

    result_width = width * 7
    result_height = height * 2

    result = Image.new('RGB', (result_width, result_height))
    for i in range(7):
        result.paste(im=images[i], box=(i * width, 0))
    for i in range(7, 14):
        result.paste(im=images[i], box=((i - 7) * width, height))

    return result

if __name__ == '__main__':
    data_dir = os.path.join(CUR_DIR, "planner/eval_res/tree_viz")
    output_dir = os.path.join(CUR_DIR, "planner/eval_res/qualitative/viz_tree")
    base_data_dir = os.path.join(data_dir, "rrt")
    neural_data_dir = os.path.join(data_dir, "neural")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    env_num = 250
    for env_idx in range(0, env_num, 25):
        print("Analyzing env {}".format(env_idx))
        env_dir = osp.join(CUR_DIR, "dataset/test_env/{}".format(env_idx))

        occ_grid = np.loadtxt(osp.join(env_dir, "occ_grid_small.txt")).astype(np.uint8)

        images = []
        for ext_num in range(0, 301, 50):
            print("loading tree nodes at step {}".format(ext_num))
            images.append(osp.join(base_data_dir, "{}/tree_nodes_{}.png".format(env_idx, ext_num)))
            shutil.copy(osp.join(base_data_dir, "{}/tree_nodes_{}.png".format(env_idx, ext_num)), osp.join(output_dir, "rrt/tree_nodes_{}_{}.png".format(env_idx, ext_num)))

        for ext_num in range(0, 301, 50):
            print("loading tree nodes at step {}".format(ext_num))
            images.append(osp.join(neural_data_dir, "{}/tree_nodes_{}.png".format(env_idx, ext_num)))
            shutil.copy(osp.join(neural_data_dir, "{}/tree_nodes_{}.png".format(env_idx, ext_num)), osp.join(output_dir, "rrt-ne/tree_nodes_{}_{}.png".format(env_idx, ext_num)))

        print(len(images))
        res_img = merge_images(images)
        res_img.save(os.path.join(output_dir, "viz_tree_compare_{}.png".format(env_idx)))