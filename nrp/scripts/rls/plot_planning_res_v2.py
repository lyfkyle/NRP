import os
import os.path as osp
import sys

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math
import torch
import json

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))
import utils

CUR_DIR = osp.dirname(osp.abspath(__file__))

planners = ["rrt", "rrt_is", "neural_d", "neural_g"]
legends = ["RRT", "RRT-IS", "NRP-RRT-d", "NRP-RRT-g"]

fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols=2, figsize=(12, 5)) # unpack

num_extensions = np.arange(0, 501, 25)
for i, planner in enumerate(planners):
    planner_success = []
    for repeat in range(10):
        with open(osp.join(CUR_DIR, "eval_res/test_ext/{}/result_{}.json".format(planner, repeat)), "r") as f:
            planner_success_tmp = json.load(f)
            planner_success_cnt = planner_success_tmp["base_success_list"] if i < 2 else planner_success_tmp["success_list"]
            planner_success.append([0] + planner_success_cnt)

    planner_success_np = np.array(planner_success)
    planner_success_mean = np.mean(planner_success_np, axis=0)
    planner_success_std = np.std(planner_success_np, axis=0)
    print(planner_success_mean, planner_success_std)

    p = ax1.plot(num_extensions, planner_success_mean, 'o-', label=legends[i], color=utils.c_map[legends[i]])
    ax1.fill_between(num_extensions, (planner_success_mean-planner_success_std), (planner_success_mean+planner_success_std), color=p[0].get_color(), alpha=.1)

ax1.set_title("Number of Successful Queries against Expansions")
# ax1.set_ylim([0, 10.0])
ax1.set_yticks(np.arange(0, 10.1, 2))
# ax1.set_xticks(num_extensions)
ax1.set_xticks(np.arange(0, 501, 50))
ax1.set_xticks(np.arange(0, 501, 25), minor=True)   # set minor ticks on x-axis
ax1.set_yticks(np.arange(0, 10, 1), minor=True)   # set minor ticks on y-axis
ax1.grid()
ax1.grid(which='minor', alpha=0.3)
ax1.set_xlabel("Number of expansions")
ax1.set_ylabel("Number of successful queries")

planning_time = np.arange(0, 10.1, 0.5)
for i, planner in enumerate(planners):
    planner_success = []
    for repeat in range(10):
        with open(osp.join(CUR_DIR, "eval_res/test_time/{}/result_{}.json".format(planner, repeat)), "r") as f:
            planner_success_tmp = json.load(f)
            planner_success_cnt = planner_success_tmp["base_success_list"] if i < 2 else planner_success_tmp["success_list"]
            planner_success.append([0] + planner_success_cnt)

    planner_success_np = np.array(planner_success)
    planner_success_mean = np.mean(planner_success_np, axis=0)
    planner_success_std = np.std(planner_success_np, axis=0)
    print(planner_success_mean, planner_success_std)

    p = ax2.plot(planning_time, planner_success_mean, 'o-', label=legends[i], color=utils.c_map[legends[i]])
    ax2.fill_between(planning_time, (planner_success_mean-planner_success_std), (planner_success_mean+planner_success_std), color=p[0].get_color(), alpha=.1)


# rrt_success = np.array([0, 34, 53, 59, 63, 65, 69, 72, 72, 73, 73, 74, 75, 78, 78, 78])
# rrt_is_success = np.array([0, 103, 123, 136, 143, 147, 154, 158, 160, 165, 167, 167, 171, 173, 174, 175])
# cvae_success = np.array([0, 58, 72, 80, 87, 88, 90, 95, 99, 100, 103, 104, 106, 108, 109, 112])
# next_success = np.array([0, 60, 79, 88, 98, 108, 109, 118, 120, 123, 124, 125, 127, 128, 131, 135])
# rrt_ne_d_success = np.array([0, 146, 160, 173, 182, 185, 190, 190, 192, 194, 196, 197, 203, 206, 210, 210])
# rrt_ne_g_success = np.array([0, 130, 159, 174, 186, 190, 196, 200, 203, 204, 206, 207, 210, 210, 210, 211])
# ax2.plot(planning_time, np.array(rrt_success) / 250, 'o-', label='RRT')
# ax2.plot(planning_time, np.array(rrt_is_success) / 250, 'o-', label='RRT-IS')
# ax2.plot(planning_time, np.array(cvae_success) / 250, 'o-', label='CVAE')
# ax2.plot(planning_time, np.array(next_success) / 250, 'o-', label='NEXT')
# # ax2.plot(planning_time, np.array(rrt_ne_g_global_success) / 250, 'o-', label='RRT-NE-global-g')
# ax2.plot(planning_time, np.array(rrt_ne_d_success) / 250, 'o-', label='RRT-NE-d')
# ax2.plot(planning_time, np.array(rrt_ne_g_success) / 250, 'o-', label='RRT-NE-g')

ax2.legend(loc='lower center', ncols=len(planners), bbox_to_anchor=(-0.1, -0.22), fancybox=True, shadow=True)
ax2.set_title("Number of Successful Queries Against Planning Time")
# ax2.set_ylim([0, 10.0])
ax2.set_yticks(np.arange(0, 10.1, 2))
ax2.set_xticks(np.arange(11))
# ax2.set_xticks(planning_time)
ax2.set_xticks(planning_time, minor=True)   # set minor ticks on x-axis
ax2.set_yticks(np.arange(0, 10, 1), minor=True)   # set minor ticks on y-axis
ax2.grid()
ax2.grid(which='minor', alpha=0.3)
ax2.set_xlabel("Planning time (s)")
ax2.set_ylabel("Number of successful queries")

plt.savefig(osp.join(CUR_DIR, "planning_res_v3.png"), bbox_inches="tight")

