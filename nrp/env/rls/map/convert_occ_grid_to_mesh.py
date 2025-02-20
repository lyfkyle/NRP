import trimesh
import numpy as np


def occ_grid_to_mesh(occ_grid):
    new_mesh = trimesh.voxel.ops.matrix_to_marching_cubes(occ_grid, pitch=0.1)

    with open("rls_mesh_v4.obj", "w") as f:
        new_mesh.export(f, file_type="obj")


def mesh_to_occ_grid(mesh_path):
    mesh = trimesh.load(mesh_path)
    voxelized = mesh.voxelized(pitch=0.1)
    voxelized.fill()
    voxelized.strip()

    m = np.array(voxelized.matrix)

    with open("occ_grid.npy", "wb") as f:
        np.save(f, m)


with open("occ_grid_v4.npy", "rb") as f:
    occ_grid = np.load(f)

occ_grid_to_mesh(occ_grid)
