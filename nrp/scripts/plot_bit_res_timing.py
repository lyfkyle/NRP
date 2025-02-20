import os
import os.path as osp

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math
import torch
import json

c_map = {
    "BIT*-15": "tab:green",
    "BIT*-25": "tab:red",
    "BIT*-50": "tab:blue",
    "BIT*-100": "tab:pink",
    "BIT*-200": "tab:orange",
}


def convert(planner_success_np):
    if planner_success_np.shape[1] == 21:
        return planner_success_np
    res = np.zeros((planner_success_np.shape[0], 21))
    for i in range(res.shape[1]):
        if i % 2 == 0:
            res[:, i] = planner_success_np[:, i * 5 // 2]
            if i != res.shape[1] - 1:
                res[:, i+1] = (planner_success_np[:, i * 5 // 2 + 2] + planner_success_np[:, i * 5 // 2 + 3]) / 2.0

    return res

CUR_DIR = osp.dirname(osp.abspath(__file__))



# fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols=2, figsize=(12, 5)) # unpack
fig, ax2 = plt.subplots(nrows = 1, ncols=1, figsize=(6, 5)) # unpack

# 11D
planning_time = np.arange(0, 10.1, 0.5)
planners = ["bit_star_15", "bit_star_25", "bit_star_50", "bit_star_100", "bit_star_200"]
legends = ["BIT*-15", "BIT*-25", "BIT*-50", "BIT*-100", "BIT*-200"]

for i, planner in enumerate(planners):
    planner_success = []
    for repeat in range(10):
        with open(osp.join(CUR_DIR, "../test/eval_res/fetch_11d/debug_bit/{}/result_{}.json".format(planner, repeat)), "r") as f:
            planner_success_tmp = json.load(f)
            planner_success_cnt = planner_success_tmp["base_success_list"]
            planner_success.append([0] + planner_success_cnt)

    planner_success_np = np.array(planner_success)
    planner_success_mean = np.mean(planner_success_np, axis=0)
    planner_success_std = np.std(planner_success_np, axis=0)
    print(planner_success_mean, planner_success_std)

    p = ax2.plot(planning_time, planner_success_mean / 250, 'o-', label=legends[i], ms=4, color=c_map[legends[i]])
    ax2.fill_between(planning_time, (planner_success_mean-planner_success_std) / 250, (planner_success_mean+planner_success_std) / 250, color=p[0].get_color(), alpha=.1)


handles, labels = ax2.get_legend_handles_labels()
# order = [0, 1, 2, 3]
# ax2.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='center left', bbox_to_anchor=(1.04, 0.5), fancybox=True, shadow=True)
ax2.legend(loc="lower center", ncols=(len(planners)+1)//2, bbox_to_anchor=(-0.18, -0.35), fancybox=True, shadow=True, fontsize=14)
# ax2.legend(loc="lower center", ncols=len(planners), bbox_to_anchor=(-0.1, -0.3), fancybox=True, shadow=True, fontsize=14)
# ax2.legend(loc="lower right", fancybox=True, shadow=True)
ax2.set_title("11D-Fetch", fontsize=18)
ax2.set_yticks(np.arange(0, 1.1, 0.2))
ax2.set_xticks(np.arange(11))
ax2.set_xticks(planning_time, minor=True)   # set minor ticks on x-axis
ax2.set_yticks(np.arange(0, 1.1, 0.1), minor=True)   # set minor ticks on y-axis
ax2.grid()
ax2.grid(which='minor', alpha=0.3)
ax2.set_xlabel("Planning time (s)", fontsize=18)
ax2.set_ylabel("Planning Success rate", fontsize=18)

plt.savefig(osp.join(CUR_DIR, "bit_res_timing.png"), bbox_inches="tight")
