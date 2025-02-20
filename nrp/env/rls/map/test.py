import os

import pybullet as p
import numpy as np

OCC_GRID_RESOLUTION = 0.1


def add_box(box_pos, half_box_size):
    colBoxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_box_size)
    visualBoxId = p.createVisualShape(
        p.GEOM_BOX, halfExtents=half_box_size, rgbaColor=[0.5, 0.5, 0.5, 0.2]
    )
    box_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=colBoxId,
        baseVisualShapeIndex=visualBoxId,
        basePosition=box_pos,
    )

    return box_id


def load_occupancy_grid(occ_grid, add_enclosing=False):
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    for i in range(occ_grid.shape[0]):
        for j in range(occ_grid.shape[1]):
            for k in range(occ_grid.shape[2]):
                if occ_grid[i, j, k] > 0:
                    add_box(
                        [i * 0.1 + 0.05, j * 0.1 + 0.05, k * 0.1 + 0.05],
                        [0.05, 0.05, 0.05],
                    )
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    occ_grid = occ_grid
    size = [
        occ_grid.shape[0] * OCC_GRID_RESOLUTION,
        occ_grid.shape[1] * OCC_GRID_RESOLUTION,
        occ_grid.shape[2] * OCC_GRID_RESOLUTION,
    ]

    # add enclosing obstacles:
    if add_enclosing:
        add_box([-0.05, size[1] // 2, 1], [0.05, size[1] // 2 + 0.1, 1])
        add_box([size[0] + 0.05, size[1] // 2, 1.0], [0.05, size[1] // 2 + 0.1, 1])
        add_box([size[0] // 2, -0.05, 1], [size[0] // 2 + 0.1, 0.05, 1])
        add_box([size[0] // 2, size[0] + 0.05, 1], [size[0] // 2 + 0.1, 0.05, 1])


def load_mesh(mesh_file):
    collision_id = p.createCollisionShape(
        p.GEOM_MESH, fileName=mesh_file, flags=p.GEOM_FORCE_CONCAVE_TRIMESH
    )
    visualBoxId = p.createVisualShape(
        p.GEOM_MESH, fileName=mesh_file, rgbaColor=[0.2, 0.2, 0.2, 0.5]
    )
    mesh_body_id = p.createMultiBody(
        baseCollisionShapeIndex=collision_id,
        baseVisualShapeIndex=visualBoxId,
        basePosition=[0, 0, 0.06],
    )  # Raise mesh to prevent robot from spawning inside the hollow mesh


p.connect(p.GUI)
p.setGravity(0, 0, -9.8)
p.setTimeStep(1.0 / 240.0)

with open("rls_occ_grid.npy", "rb") as f:
    occ_grid = np.load(f)

# This takes very long
# load_occupancy_grid(occ_grid)

load_mesh("rls_mesh.obj")

input()
