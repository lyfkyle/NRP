import os
import os.path as osp

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math
import torch
import json
import utils

CUR_DIR = osp.dirname(osp.abspath(__file__))

def convert(planner_success_np):
    res = np.zeros((planner_success_np.shape[0], 21))
    for i in range(res.shape[1]):
        if i % 2 == 0:
            res[:, i] = planner_success_np[:, i * 5 // 2]
            if i != res.shape[1] - 1:
                res[:, i+1] = (planner_success_np[:, i * 5 // 2 + 2] + planner_success_np[:, i * 5 // 2 + 3]) / 2.0

    return res

planners = ["neural_d", "neural_g", "cvae", "rrt", "rrt_is"]
legends = ["NRP-RRT-d", "NRP-RRT-g", "CVAE-RRT", "RRT", "RRT-IS"]

fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols=2, figsize=(12, 5)) # unpack

num_extensions = np.arange(0, 301, 25)

for i, planner in enumerate(planners):
    planner_success = []
    for repeat in range(10):
        with open(osp.join(CUR_DIR, "planner/eval_res/test_ext/{}/result_{}.json".format(planner, repeat)), "r") as f:
            planner_success_tmp = json.load(f)
            planner_success_cnt = planner_success_tmp["base_success_list"] if i >= 3 else planner_success_tmp["success_list"]
            planner_success.append([0] + planner_success_cnt)

    planner_success_np = np.array(planner_success)
    planner_success_mean = np.mean(planner_success_np, axis=0)
    planner_success_std = np.std(planner_success_np, axis=0)
    print(planner_success_mean, planner_success_std)

    p = ax1.plot(num_extensions, planner_success_mean / 250, 'o-', label=legends[i], color=utils.c_map[legends[i]])
    ax1.fill_between(num_extensions, (planner_success_mean-planner_success_std) / 250, (planner_success_mean+planner_success_std) / 250, color=p[0].get_color(), alpha=.1)

ax1.set_title("Planning Success Rate Against Expansions")
# ax1.legend(loc="upper left")
ax1.set_yticks(np.arange(0, 1.1, 0.2))
ax1.set_xticks(num_extensions)
ax1.set_xticks(np.arange(0, 301, 12.5), minor=True)   # set minor ticks on x-axis
ax1.set_yticks(np.arange(0, 1.1, 0.1), minor=True)   # set minor ticks on y-axis
ax1.grid()
ax1.grid(which='minor', alpha=0.3)
ax1.set_xlabel("Number of vertex expansions")
ax1.set_ylabel("Success rate")

planning_time = np.arange(0, 10.1, 0.5)
for i, planner in enumerate(planners):
    planner_success = []
    for repeat in range(10):
        with open(osp.join(CUR_DIR, "planner/eval_res/test_time/{}/result_{}.json".format(planner, repeat)), "r") as f:
            planner_success_tmp = json.load(f)
            planner_success_cnt = planner_success_tmp["base_success_list"] if i >= 3 else planner_success_tmp["success_list"]
            planner_success.append([0] + planner_success_cnt)

    planner_success_np = convert(np.array(planner_success))
    planner_success_mean = np.mean(planner_success_np, axis=0)
    planner_success_std = np.std(planner_success_np, axis=0)
    print(planner_success_mean, planner_success_std)

    p = ax2.plot(planning_time, planner_success_mean / 250, 'o-', label=legends[i], ms=4, color=utils.c_map[legends[i]])
    ax2.fill_between(planning_time, (planner_success_mean-planner_success_std) / 250, (planner_success_mean+planner_success_std) / 250, color=p[0].get_color(), alpha=.1)


handles, labels = ax2.get_legend_handles_labels()
# order = [0, 1, 2, 3]
# ax2.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='center left', bbox_to_anchor=(1.04, 0.5), fancybox=True, shadow=True)
ax2.legend(loc="lower center", ncols=len(planners), bbox_to_anchor=(-0.1, -0.22), fancybox=True, shadow=True)
ax2.set_title("Planning Success Rate Against Planning Time")
ax2.set_yticks(np.arange(0, 1.1, 0.2))
ax2.set_xticks(np.arange(11))
ax2.set_xticks(planning_time, minor=True)   # set minor ticks on x-axis
ax2.set_yticks(np.arange(0, 1.1, 0.1), minor=True)   # set minor ticks on y-axis
ax2.grid()
ax2.grid(which='minor', alpha=0.3)
ax2.set_xlabel("Planning time (s)")
ax2.set_ylabel("Success rate")

plt.savefig("planning_res_v2.png", bbox_inches="tight")

