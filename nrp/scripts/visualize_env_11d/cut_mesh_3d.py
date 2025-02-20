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

data_dir = os.path.join(dir_current, "../dataset/Connellsville")


def show(chair_mesh, chair_voxels, colors=(1, 1, 1, 0.3)):
    scene = chair_mesh.scene()
    scene.add_geometry(chair_voxels.as_boxes(colors=colors))
    scene.show()

if __name__ == '__main__':
    for path in Path(data_dir).rglob('mesh_z_up.obj'):
        print(path)

        chair_mesh = trimesh.load(path)
        if isinstance(chair_mesh, trimesh.scene.Scene):
            chair_mesh = trimesh.util.concatenate([
                trimesh.Trimesh(mesh.vertices, mesh.faces)
                for mesh in chair_mesh.geometry.values()])

        voxelized = chair_mesh.voxelized(pitch=0.1)
        voxelized.fill()
        voxelized.strip()

        z_plane = int((0.2 - voxelized.translation[2]) // 0.1)

        m = np.array(voxelized.matrix)
        m_cut = m[:, :, z_plane:z_plane + 20].astype(float)  # 2m height
        m_cut[0, :, :] = 1.0
        m_cut[-1, :, :] = 1.0
        m_cut[:, 0, :] = 1.0
        m_cut[:, -1, :] = 1.0
        print(m_cut.shape)
        print(m_cut.max())
        print(m_cut.min())

        fixed_occ_grid = np.zeros((m_cut.shape[0], m_cut.shape[1], m_cut.shape[2]), dtype=np.uint8)
        for i in range(fixed_occ_grid.shape[0]):
            for j in range(fixed_occ_grid.shape[1]):
                for k in range(fixed_occ_grid.shape[2]):
                    fixed_occ_grid[i, j, k] = m_cut[i, j, k]

        new_mesh = trimesh.voxel.ops.matrix_to_marching_cubes(fixed_occ_grid, pitch=0.1)
        # new_mesh.show()

        with open(os.path.join(path.parent, "env_final.obj"), 'w') as f:
            new_mesh.export(f, file_type='obj')

        voxelized = new_mesh.voxelized(pitch=0.1)
        voxelized.fill()
        voxelized.strip()

        with open(os.path.join(path.parent, "occ_grid_final.npy"), 'wb') as f:
            np.save(f, fixed_occ_grid)

        utils.visualize_nodes_global(os.path.join(path.parent, "env_final.obj"), occ_g=fixed_occ_grid, curr_node_posns=[], start_pos=None, goal_pos=None, show=False, save=True, file_name=os.path.join(path.parent, "env_final.png"))

        # assert occ_grid.shape[2] == 20ls
