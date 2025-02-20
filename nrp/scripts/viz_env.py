import os
import os.path as osp
import sys
import json

from nrp.env.fetch_11d.maze import Fetch11DEnv
from nrp.env.fetch_11d import utils
from nrp.env.rls import utils as rls_utils

CUR_DIR = osp.dirname(osp.abspath(__file__))
print(CUR_DIR)
# data_dir = os.path.join(CUR_DIR, "../env/fetch_11d/dataset/test_env_01/211")
data_dir = os.path.join(CUR_DIR, "../env/rls/dataset/test_env/0")
# maze_dir = os.path.join(data_dir, "Connellsville")  # we choose this environment for visualization
# maze_dir = os.path.join(data_dir, "Azusa")  # we choose this environment for visualization
maze_dir = data_dir
env = Fetch11DEnv(gui=False, add_robot=True)

# mesh = os.path.join(data_dir, "PROVANS.obj")
# if mesh is not None:
#     col_id = maze.p.createCollisionShape(
#         maze.p.GEOM_MESH, fileName=mesh, flags=maze.p.GEOM_FORCE_CONCAVE_TRIMESH, meshScale=[0.001, 0.001, 0.001]
#     )
#     visualBoxId = maze.p.createVisualShape(maze.p.GEOM_MESH, fileName=mesh, meshScale=[0.001, 0.001, 0.001])
#     maze.p.createMultiBody(baseCollisionShapeIndex=col_id, baseVisualShapeIndex=visualBoxId, baseOrientation=[ 0.7071068, 0, 0, 0.7071068 ])

mesh = os.path.join(maze_dir, "env.obj")
mesh = rls_utils.get_mesh_path(None)
env.load_mesh(mesh)

# with open(osp.join(maze_dir, "start_goal.json")) as f:
#     start_goal = json.load(f)

# env.add_robot(start_goal[1])
# print(env.pb_ompl_interface.is_state_valid(start_goal[1]))

occ_grid = rls_utils.get_occ_grid(maze_dir)
utils.visualize_nodes_global(mesh, occ_grid, [], start_pos=None, goal_pos=None, show=False, save=True, file_name="tmp.png")

# while True:
#     print(maze.robot.get_cur_state())
#     robot_state_str = input()
#     robot_state_list = robot_state_str.split()
#     robot_state = [float(x) for x in robot_state_list]
#     maze.robot.set_state(robot_state)
input()



# jointPoses = list(maze.p.calculateInverseKinematics(maze.robot_id, maze.robot.eef_joint_idx, targetPosition=[-0.5, -1, 1], residualThreshold=0.01))
# jointPoses.insert(2, 0)
# jointPoses[0] = -1.0
# jointPoses[1] = -1.0
# print(jointPoses)
# maze.robot.set_state(jointPoses)
# input()