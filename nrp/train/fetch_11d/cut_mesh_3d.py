import os.path as osp
import sys
import os

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), "../"))

import os
import numpy as np
import inspect
import trimesh
from pathlib import Path

from env.fetch_11d import utils

dir_current = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
data_dir = os.path.join(dir_current, "../env/fetch_11d/dataset/gibson/train")


def show(chair_mesh, chair_voxels, colors=(1, 1, 1, 0.3)):
    scene = chair_mesh.scene()
    scene.add_geometry(chair_voxels.as_boxes(colors=colors))
    scene.show()


if __name__ == "__main__":
    for path in Path(data_dir).rglob("env_large.obj"):
        print(path)

        # skip existing
        if osp.exists(os.path.join(path.parent, "env.png")):
            continue

        chair_mesh = trimesh.load(path)
        if isinstance(chair_mesh, trimesh.scene.Scene):
            chair_mesh = trimesh.util.concatenate(
                [trimesh.Trimesh(mesh.vertices, mesh.faces) for mesh in chair_mesh.geometry.values()]
            )

        # convert to occ_grid
        voxelized = chair_mesh.voxelized(pitch=0.08)
        voxelized.fill()
        voxelized.strip()
        occ_grid = voxelized.matrix

        # Cut z-plane
        # z_plane = int((0.2 - voxelized.translation[2]) // 0.1)
        # m = np.array(voxelized.matrix)
        # m_cut = m[:, :, z_plane:z_plane + 20].astype(float)  # 2m height
        # m_cut[0, :, :] = 1.0
        # m_cut[-1, :, :] = 1.0
        # m_cut[:, 0, :] = 1.0
        # m_cut[:, -1, :] = 1.0
        # print(m_cut.shape)
        # print(m_cut.max())
        # print(m_cut.min())

        # Rotate
        # fixed_occ_grid = np.zeros((occ_grid.shape[1], occ_grid.shape[0], occ_grid.shape[2]), dtype=np.uint8)
        # for i in range(fixed_occ_grid.shape[0]):
        #     for j in range(fixed_occ_grid.shape[1]):
        #         for k in range(fixed_occ_grid.shape[2]):
        #             fixed_occ_grid[i, j, k] = occ_grid[-j - 1, i, k]
        # occ_grid = fixed_occ_grid

        # Extract out middle of the env
        print(occ_grid.shape)
        desired_h = 150
        desired_w = 150
        orig_h = occ_grid.shape[0]
        orig_w = occ_grid.shape[1]
        h_diff = desired_h // 2 - orig_h // 2
        i_min = max(0, h_diff)
        i_max = min(desired_h, desired_h // 2 + orig_h // 2)
        w_diff = desired_w // 2 - orig_w // 2
        j_min = max(0, w_diff)
        j_max = min(desired_w, desired_w // 2 + orig_w // 2)
        print(i_min, i_max, j_min, j_max, h_diff, w_diff)
        fixed_occ_grid_2 = np.ones((desired_h, desired_w, 20), dtype=np.uint8)
        for i in range(i_min, i_max):
            for j in range(j_min, j_max):
                for k in range(fixed_occ_grid_2.shape[2]):
                    fixed_occ_grid_2[i, j, k] = occ_grid[i - h_diff, j - w_diff, k]
        fixed_occ_grid_2[0, :, :] = 1
        fixed_occ_grid_2[-1, :, :] = 1
        fixed_occ_grid_2[:, 0, :] = 1
        fixed_occ_grid_2[:, -1, :] = 1
        occ_grid = fixed_occ_grid_2

        # m_cut_img = 255 * np.expand_dims(m_cut, -1).astype(np.uint8)
        # cv2.imwrite(os.path.join(path.parent, "grid.png"), m_cut_img)
        # np.savetxt(os.path.join(path.parent, "occ_grid.txt"), m_cut)
        # assert fixed_occ_grid.shape[2] == 20
        # with open(os.path.join(path.parent, "occ_grid_large.npy"), 'wb') as f:
        #     np.save(f, fixed_occ_grid)

        new_mesh = trimesh.voxel.ops.matrix_to_marching_cubes(occ_grid, pitch=0.1)
        # new_mesh.show()

        with open(os.path.join(path.parent, "env.obj"), "w") as f:
            new_mesh.export(f, file_type="obj")

        voxelized = new_mesh.voxelized(pitch=0.1)
        voxelized.fill()
        voxelized.strip()

        m = np.array(voxelized.matrix)
        # fixed_occ_grid = np.zeros((m.shape[0], m.shape[1], 20), dtype=np.uint8)
        fixed_occ_grid = np.zeros((desired_h, desired_w, 20), dtype=np.uint8)
        print(fixed_occ_grid.shape)
        for i in range(fixed_occ_grid.shape[0]):
            for j in range(fixed_occ_grid.shape[1]):
                for k in range(20):
                    fixed_occ_grid[i, j, k] = m[i, j, k]

        with open(os.path.join(path.parent, "occ_grid.npy"), "wb") as f:
            np.save(f, fixed_occ_grid)

        utils.visualize_nodes_global(
            os.path.join(path.parent, "env.obj"),
            occ_g=fixed_occ_grid,
            curr_node_posns=[],
            start_pos=None,
            goal_pos=None,
            show=False,
            save=True,
            file_name=os.path.join(path.parent, "env.png"),
        )

        # break
        # assert occ_grid.shape[2] == 20ls
