import os
import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))

import numpy as np
import inspect
import trimesh
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
    # min_x = float('inf')
    # min_y = float('inf')
    # for path in Path(data_dir).rglob('mesh_z_up.obj'):
    #     print(path)

    #     orig_occ_grid = np.loadtxt(os.path.join(path.parent, "occ_grid_fixed.txt"))

    #     min_x = min(orig_occ_grid.shape[0], min_x)
    #     min_y = min(orig_occ_grid.shape[1], min_y)

    # print(min_x, min_y)

    width = 100
    for path in Path(data_dir).rglob('mesh_z_up.obj'):
        print(path)
        small_occ_grid = np.ones((width, width)).astype(np.uint8)

        orig_occ_grid = np.loadtxt(os.path.join(path.parent, "occ_grid_fixed.txt"))
        orig_occ_grid_cx = orig_occ_grid.shape[0] // 2
        orig_occ_grid_cy = orig_occ_grid.shape[1] // 2
        print(orig_occ_grid.shape)

        for i in range(small_occ_grid.shape[0]):
            for j in range(small_occ_grid.shape[1]):
                x = orig_occ_grid_cx - int(width / 2) + i
                y = orig_occ_grid_cy - int(width / 2) + j
                if x >= 0 and x < orig_occ_grid.shape[0] and y >= 0 and y < orig_occ_grid.shape[1]:
                    small_occ_grid[i, j] = orig_occ_grid[x , y]
        print(small_occ_grid.shape)

        # make border to be filled
        small_occ_grid[0, :] = 1
        small_occ_grid[-1, :] = 1
        small_occ_grid[:, 0] = 1
        small_occ_grid[:, -1] = 1

        np.savetxt(os.path.join(path.parent, "occ_grid_small.txt"), small_occ_grid)
        print("saved to {}".format(path.parent))

        # utils.visualize_nodes_global(orig_occ_grid, [], None, None, show=False, save=True, file_name=os.path.join(path.parent, "env.png"))
        # print("here")
        # utils.visualize_nodes_global(small_occ_grid, [], None, None, show=False, save=True, file_name=os.path.join(path.parent, "env_large.png"))

        import matplotlib.pyplot as plt
        plt.imshow(small_occ_grid)
        plt.savefig(os.path.join(path.parent, "env_small.png"))

        # small_occ_grid = np.repeat(np.expand_dims(small_occ_grid, 2), 2, axis=2) # repeat z 5 times
        small_occ_grid = np.expand_dims(small_occ_grid, 2)
        new_mesh = trimesh.voxel.ops.matrix_to_marching_cubes(small_occ_grid, pitch=0.1)
        # new_mesh.show()

        with open(os.path.join(path.parent, "env_small.obj"), 'w') as f:
            new_mesh.export(f, file_type='obj')