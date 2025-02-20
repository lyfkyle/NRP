import os
import os.path as osp

import numpy as np
import trimesh

CUR_DIR = osp.dirname(osp.abspath(__file__))


def show(chair_mesh, chair_voxels, colors=(1, 1, 1, 0.3)):
    scene = chair_mesh.scene()
    scene.add_geometry(chair_voxels.as_boxes(colors=colors))
    scene.show()


path = os.path.join(CUR_DIR, "../env/fetch_11d/dataset/kitchen/PROVANS.obj")

chair_mesh = trimesh.load(path)
if isinstance(chair_mesh, trimesh.scene.Scene):
    chair_mesh = trimesh.util.concatenate(
        [trimesh.Trimesh(mesh.vertices, mesh.faces) for mesh in chair_mesh.geometry.values()]
    )

matrix = np.eye(4)
matrix[:2, :2] /= 1000
chair_mesh.apply_transform(matrix)

voxelized = chair_mesh.voxelized(pitch=0.1)
voxelized.fill()
voxelized.strip()

print(voxelized.translation)
# z_plane = int((-1.2 - voxelized.translation[2]) // 0.1)

m = np.array(voxelized.matrix)

m_cut = m[:, :, 2:].astype(np.float32)
# m_cut[:, :, 0] = 0.0
# m_cut[0, :, :] = 1.0
# m_cut[-1, :, :] = 1.0
# m_cut[:, 0, :] = 1.0
# m_cut[:, -1, :] = 1.0
print(m_cut.shape)
print(m_cut.max())
print(m_cut.min())

fixed_occ_grid = m_cut
# fixed_occ_grid = np.zeros((m_cut.shape[1], m_cut.shape[0], m_cut.shape[2]), dtype=np.uint8)
# for i in range(fixed_occ_grid.shape[0]):
#     for j in range(fixed_occ_grid.shape[1]):
#         for k in range(fixed_occ_grid.shape[2]):
#             fixed_occ_grid[i, j, k] = m_cut[m_cut.shape[0] - j - 1, i, k]

# for x_cut in range(fixed_occ_grid.shape[0]):
#     if fixed_occ_grid[x_cut, fixed_occ_grid.shape[1] // 2, fixed_occ_grid.shape[2] // 2]:
#         break

# for y_cut in range(fixed_occ_grid.shape[1]):
#     if fixed_occ_grid[fixed_occ_grid.shape[0] // 2, 0, fixed_occ_grid.shape[2] // 2]:
#         break

# print(x_cut, y_cut)
# y_cut = 4  # hardcode

# fixed_occ_grid = fixed_occ_grid[x_cut:, y_cut:]
# fixed_occ_grid[:, :, 0] = 0.0
# fixed_occ_grid[0, :, :] = 1.0
# fixed_occ_grid[-1, :, :] = 1.0
# fixed_occ_grid[:, 0, :] = 1.0
# fixed_occ_grid[:, -1, :] = 1.0

# m_cut_img = 255 * np.expand_dims(m_cut, -1).astype(np.uint8)
# cv2.imwrite(os.path.join(path.parent, "grid.png"), m_cut_img)
# np.savetxt(os.path.join(path.parent, "occ_grid.txt"), m_cut)
# assert fixed_occ_grid.shape[2] == 20
# with open(os.path.join(path.parent, "occ_grid_large.npy"), 'wb') as f:
#     np.save(f, fixed_occ_grid)

new_mesh = trimesh.voxel.ops.matrix_to_marching_cubes(fixed_occ_grid, pitch=0.1)
# new_mesh.show()

with open(os.path.join(CUR_DIR, "rls_fixed.obj"), "w") as f:
    new_mesh.export(f, file_type="obj")

voxelized = new_mesh.voxelized(pitch=0.1)
voxelized.fill()
voxelized.strip()

m = np.array(voxelized.matrix)
fixed_occ_grid = np.zeros((m.shape[0], m.shape[1], 20), dtype=np.uint8)
print(fixed_occ_grid.shape)
for i in range(fixed_occ_grid.shape[0]):
    for j in range(fixed_occ_grid.shape[1]):
        for k in range(min(fixed_occ_grid.shape[2], m.shape[2])):
            fixed_occ_grid[i, j, k] = m[i, j, k]

with open(os.path.join(CUR_DIR, "occ_grid.npy"), "wb") as f:
    np.save(f, fixed_occ_grid)

# with open(os.path.join(path.parent, "occ_grid_large.npy"), 'rb') as f:
#     occ_grid = np.load(f)

# assert occ_grid.shape[2] == 20ls
