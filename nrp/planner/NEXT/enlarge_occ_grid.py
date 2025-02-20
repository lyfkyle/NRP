import os
import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))

import numpy as np
import inspect
import trimesh
import cv2
from pathlib import Path

import utils

dir_current = os.path.dirname(
    os.path.abspath(
        inspect.getfile(
            inspect.currentframe())))

data_dir = os.path.join(dir_current, "../dataset/gibson")

def show(chair_mesh, chair_voxels, colors=(1, 1, 1, 0.3)):
    scene = chair_mesh.scene()
    scene.add_geometry(chair_voxels.as_boxes(colors=colors))
    scene.show()

if __name__ == '__main__':
    max_x = 0
    max_y = 0
    min_x = float('inf')
    min_y = float('inf')
    for path in Path(data_dir).rglob('mesh_z_up.obj'):
        print(path)

        orig_occ_grid = np.loadtxt(os.path.join(path.parent, "occ_grid_fixed.txt"))

        max_x = max(orig_occ_grid.shape[0], max_x)
        max_y = max(orig_occ_grid.shape[1], max_y)
        min_x = min(orig_occ_grid.shape[0], min_x)
        min_y = min(orig_occ_grid.shape[1], min_y)

    print(max_x, max_y)
    print(min_x, min_y)

    # width = max(max_x, max_y)
    # for path in Path(data_dir).rglob('mesh_z_up.obj'):
    #     print(path)
    #     large_occ_grid = np.ones((width, width)).astype(np.uint8)

    #     orig_occ_grid = np.loadtxt(os.path.join(path.parent, "occ_grid_fixed.txt"))
    #     print(orig_occ_grid.shape)

    #     for i in range(orig_occ_grid.shape[0]):
    #         for j in range(orig_occ_grid.shape[1]):
    #             large_occ_grid[i, j] = orig_occ_grid[i, j]
    #     print(large_occ_grid.shape)

    #     # np.savetxt(os.path.join(path.parent, "occ_grid_large.txt"), large_occ_grid)
    #     # print("saved to {}".format(path.parent))

    #     # utils.visualize_nodes_global(orig_occ_grid, [], None, None, show=False, save=True, file_name=os.path.join(path.parent, "env.png"))
    #     # print("here")
    #     # utils.visualize_nodes_global(large_occ_grid, [], None, None, show=False, save=True, file_name=os.path.join(path.parent, "env_large.png"))

    #     import matplotlib.pyplot as plt
    #     plt.imshow(large_occ_grid)
    #     plt.savefig(os.path.join(path.parent, "env_large.png"))