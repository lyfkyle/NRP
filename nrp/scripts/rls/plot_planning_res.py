import os
import os.path as osp

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math
import torch
import json

CUR_DIR = osp.dirname(osp.abspath(__file__))

planners = ["rrt", "rrt_is", "neural_d", "neural_g"]
legends = ["RRT", "RRT-IS", "RRT-NE-d", "RRT-NE-g"]

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

ax2.legend(loc='center left', bbox_to_anchor=(1.04, 0.5), fancybox=True, shadow=True)
ax2.set_title("Successful Planning Queries Against Planning Time")
ax2.set_xticks(planning_time)
ax2.set_xlabel("Planning time (s)")
ax2.set_ylabel("Number of Successful Queries")

plt.savefig("planning_res_v2.png", bbox_inches="tight")

