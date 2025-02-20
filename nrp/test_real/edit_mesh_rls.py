import os
import sys
import os.path as osp
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))

import numpy as np
import trimesh

from rls.maze import Maze

CUR_DIR = osp.dirname(osp.abspath(__file__))

mesh_path = os.path.join(CUR_DIR, "rls_tmp.obj")

with open(os.path.join(CUR_DIR, "map/occ_grid.npy"), 'rb') as f:
    occ_grid = np.load(f)

print(occ_grid.shape)

# remove part of table
occ_grid[105:110, 5:35, :7] = 0
# remove fetch dock
occ_grid[60:80, 6:15, :] = 0
# remove fatty wall
occ_grid[73:75, 49:60, :] = 0

new_occ_grid = np.zeros((121, 77, 20))
for i in range(occ_grid.shape[0]):
    new_occ_grid[int(i / (125 / 121))] = occ_grid[i]

# edit middle table
new_occ_grid[52:60, 20:40, :] = 0

# edit kitchen table
tmp = np.copy(new_occ_grid)
new_occ_grid[66:70, 40:70, :] = 0
new_occ_grid[68:72, 40:70, :] = tmp[66:70, 40:70, :]
scale = 16 / 18
new_occ_grid[75:110, 35:65, :] = 0
for i in range(75, 110):
    new_occ_grid[79 + int((i - 75) * scale), 35:65, : ] = tmp[i, 35:65, :]
new_occ_grid[80:111, 54:57, :] = 0
new_occ_grid[83:85, 49:55, :] = 0
new_occ_grid[80:111, 35:38, :] = 0
new_occ_grid[93:100, 35:42, :] = 0
new_occ_grid[93:110, 35:45, :5] = 0
new_occ_grid[102, 43, :] = 1
new_occ_grid[102, 42:55, 15] = 1

# edit dining table
tmp = np.copy(new_occ_grid)
new_occ_grid[79:110, 10:33, :] = 0
new_occ_grid[80:111, 10:33, :] = tmp[79:110, 10:33, :]
new_occ_grid[106:114, 10:33, :] = 0
new_occ_grid[114:119, 10:30, :] = 0
new_occ_grid[110:114, 30:42, :] = 0
new_occ_grid[80:111, 30, :] = 0
new_occ_grid[80:111, 13:15, :] = 0

new_mesh = trimesh.voxel.ops.matrix_to_marching_cubes(new_occ_grid, pitch=0.1)
# new_mesh.show()

with open(mesh_path, 'w') as f:
    new_mesh.export(f, file_type='obj')

with open(os.path.join(CUR_DIR, "occ_grid_tmp.npy"), 'wb') as f:
    np.save(f, new_occ_grid)

maze = Maze()
maze.load_mesh(mesh_path)
# maze.load_occupancy_grid(occ_grid, add_box=True)

input()

