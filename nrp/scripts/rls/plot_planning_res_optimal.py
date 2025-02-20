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

planners = ["rrt_star", "bit","neural_d_star", "neural_g_star"]
legends = ["IRRT*", "BIT*", "IRRT*-NE-d", "IRRT*-NE-g"]

fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols=2, figsize=(12, 5)) # unpack

num_extensions = np.arange(0, 301, 25)
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

    p = ax1.plot(num_extensions, planner_success_mean, 'o-', label=legends[i])
    ax1.fill_between(num_extensions, (planner_success_mean-planner_success_std), (planner_success_mean+planner_success_std), color=p[0].get_color(), alpha=.1)

ax1.set_title("Successful Planning Queries Against Extension Number")
ax1.set_xticks(num_extensions)
ax1.set_xlabel("Number of extensions")
ax1.set_ylabel("Number of Successful Queries")

planning_time = np.arange(11)
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

    p = ax2.plot(planning_time, planner_success_mean, 'o-', label=legends[i])
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

ax2.legend(loc='center left', bbox_to_anchor=(1.04, 0.5), fancybox=True, shadow=True)
ax2.set_title("Successful Planning Queries Against Planning Time")
ax2.set_xticks(planning_time)
ax2.set_xlabel("Planning time (s)", fontsize=18)
ax2.set_ylabel("Number of Successful Queries", fontsize=18)

plt.savefig("planning_res_optimal_v2.png", bbox_inches="tight")

