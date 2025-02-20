import os
import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))

import numpy as np
from NEXT.algorithm import RRT_EPS

STICK_LENGTH = 1.5 * 2 / 15
LIMITS = np.array([1., 1., 8.*RRT_EPS])