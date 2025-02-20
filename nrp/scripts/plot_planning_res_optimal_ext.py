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
    "NRP-d": "tab:orange",
    "NRP-g": "tab:red",
    "NRP*-d": "tab:orange",
    "NRP*-g": "tab:red",
    "NRP-d-global": "tab:cyan",
    "NRP-g-global": "tab:pink",
    "CVAE-RRT": "tab:cyan",
    "CVAE-IRRT*": "tab:cyan",
    "IRRT*": "tab:blue",
    "BIT*": "tab:green",
    "NEXT": "tab:purple",
    "RRT": "tab:gray",
    "RRT-IS": "tab:olive",
    "Decoupled": "tab:brown",
    "FIRE": "tab:pink",
    "FIRE*": "tab:pink",
    "VQMPT": "tab:brown",
    "VQMPT*": "tab:brown",
}


def convert(planner_success_np):
    res = np.zeros((planner_success_np.shape[0], 21))
    for i in range(res.shape[1]):
        if i % 2 == 0:
            res[:, i] = planner_success_np[:, i * 5 // 2]
            if i != res.shape[1] - 1:
                res[:, i+1] = (planner_success_np[:, i * 5 // 2 + 2] + planner_success_np[:, i * 5 // 2 + 3]) / 2.0

    return res

CUR_DIR = osp.dirname(osp.abspath(__file__))

# planners = ["neural_d_star", "neural_g_star", "cvae_star", "next", "rrt_star", "bit_star", "vqmpt_rrt_star"]
# legends = ["NRP*-d", "NRP*-g", "CVAE-IRRT*", "NEXT", "IRRT*", "BIT*", "VQMPT*"]
planners = ["neural_d_star_no_col", "neural_g_star", "cvae_star", "next", "rrt_star", "bit_star"]
legends = ["NRP*-d", "NRP*-g", "CVAE-IRRT*", "NEXT", "IRRT*", "BIT*"]


fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols=2, figsize=(12, 5)) # unpack

num_extensions = np.arange(0, 501, 25)

for i, planner in enumerate(planners):
    planner_success = []
    for repeat in range(10):
        with open(osp.join(CUR_DIR, "../test/eval_res/snake_8d/test_ext_500/{}/result_{}.json".format(planner, repeat)), "r") as f:
            planner_success_tmp = json.load(f)
            planner_success_cnt = planner_success_tmp["base_success_list"] if i >= 4 else planner_success_tmp["success_list"]
            planner_success.append([0] + planner_success_cnt)

    # planner_success_np = convert(np.array(planner_success))
    planner_success_np = np.array(planner_success)
    planner_success_mean = np.mean(planner_success_np, axis=0)
    planner_success_std = np.std(planner_success_np, axis=0)
    print(planner_success_mean, planner_success_std)

    p = ax1.plot(num_extensions, planner_success_mean / 250, 'o-', label=legends[i], ms=4, color=c_map[legends[i]])
    ax1.fill_between(num_extensions, (planner_success_mean-planner_success_std) / 250, (planner_success_mean+planner_success_std) / 250, color=p[0].get_color(), alpha=.1)


handles, labels = ax1.get_legend_handles_labels()
# order = [0, 1, 2, 3]
# ax2.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='center left', bbox_to_anchor=(1.04, 0.5), fancybox=True, shadow=True)
# ax1.legend(loc="lower center", ncols=len(planners), bbox_to_anchor=(0.5, -0.22), fancybox=True, shadow=True)
# ax1.legend(loc="lower right", fancybox=True, shadow=True)
ax1.set_title("8D-Snake", fontsize=18)
ax1.set_yticks(np.arange(0, 1.1, 0.2))
ax1.set_xticks(np.arange(0, 501, 50))
ax1.set_xticks(num_extensions, minor=True)   # set minor ticks on x-axis
ax1.set_yticks(np.arange(0, 1.1, 0.1), minor=True)   # set minor ticks on y-axis
ax1.grid()
ax1.grid(which='minor', alpha=0.3)
ax1.set_xlabel("Number of vertex expansions", fontsize=18)
ax1.set_ylabel("Planning Success rate", fontsize=18)

num_extensions = np.arange(0, 501, 25)
# planners = ["neural_d_star", "neural_g_star", "cvae_star", "next", "fire_star", "rrt_star", "bit_star", "vqmpt_rrt_star"]
# legends = ["NRP*-d", "NRP*-g", "CVAE-IRRT*", "NEXT", "FIRE*", "IRRT*", "BIT*", "VQMPT*"]
planners = ["neural_d_star_no_col", "neural_g_star", "cvae_star", "next", "fire_star", "rrt_star", "bit_star_100"]
legends = ["NRP*-d", "NRP*-g", "CVAE-IRRT*", "NEXT", "FIRE*", "IRRT*", "BIT*"]

for i, planner in enumerate(planners):
    planner_success = []
    for repeat in range(10):
        with open(osp.join(CUR_DIR, "../test/eval_res/fetch_11d/test_ext_v3/{}/result_{}.json".format(planner, repeat)), "r") as f:
            planner_success_tmp = json.load(f)
            planner_success_cnt = planner_success_tmp["base_success_list"] if i >= 5 else planner_success_tmp["success_list"]
            planner_success.append([0] + planner_success_cnt)

    planner_success_np = np.array(planner_success)
    planner_success_mean = np.mean(planner_success_np, axis=0)
    planner_success_std = np.std(planner_success_np, axis=0)
    print(planner_success_mean, planner_success_std)

    p = ax2.plot(num_extensions, planner_success_mean / 250, 'o-', label=legends[i], ms=4, color=c_map[legends[i]])
    ax2.fill_between(num_extensions, (planner_success_mean-planner_success_std) / 250, (planner_success_mean+planner_success_std) / 250, color=p[0].get_color(), alpha=.1)


handles, labels = ax1.get_legend_handles_labels()
# order = [0, 1, 2, 3]
# ax2.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='center left', bbox_to_anchor=(1.04, 0.5), fancybox=True, shadow=True)
ax2.legend(loc="lower center", ncols=(len(planners)+1)//2, bbox_to_anchor=(-0.18, -0.35), fancybox=True, shadow=True, fontsize=14)
# ax2.legend(loc="lower center", ncols=len(planners), bbox_to_anchor=(-0.1, -0.3), fancybox=True, shadow=True, fontsize=14)
# ax2.legend(loc="lower right", fancybox=True, shadow=True)
ax2.set_title("11D-Fetch", fontsize=18)
ax2.set_yticks(np.arange(0, 1.1, 0.2))
ax2.set_xticks(np.arange(0, 501, 50))
ax2.set_xticks(num_extensions, minor=True)   # set minor ticks on x-axis
ax2.set_yticks(np.arange(0, 1.1, 0.1), minor=True)   # set minor ticks on y-axis
ax2.grid()
ax2.grid(which='minor', alpha=0.3)
ax2.set_xlabel("Number of vertex expansions", fontsize=18)
ax2.set_ylabel("Planning Success rate", fontsize=18)

plt.savefig(osp.join(CUR_DIR, "planning_res_ext_optimal_v4.png"), bbox_inches="tight")
