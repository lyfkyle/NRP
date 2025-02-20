import sys
import os.path as osp
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), "../"))

from env.maze_2d import Maze2D

if __name__ == '__main__':
    maze = Maze2D()
    maze.random_obstacles(mode=3)
    maze.sample_start_goal()
    maze.plan2(interpolate=True)
    maze.execute2()
    print("here")