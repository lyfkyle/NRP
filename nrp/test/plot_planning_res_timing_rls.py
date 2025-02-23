import os
import os.path as osp
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import json

from nrp import ROOT_DIR
from nrp.env.fetch_11d import utils as utils_11d

CUR_DIR = osp.dirname(osp.abspath(__file__))

planners = ["nrp_d", "nrp_g", "rrt", "rrt_is", "decomposed"]
legends = ["NRP-d", "NRP-g", "RRT", "RRT-IS", "Decomposed"]

fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))  # unpack

planning_time = np.arange(0, 10.1, 0.5)
for i, planner in enumerate(planners):
    planner_success = []
    for repeat in range(10):
        with open(osp.join(ROOT_DIR, "results/rls/test_time/{}/result_{}.json".format(planner, repeat)), "r") as f:
            planner_success_tmp = json.load(f)
            planner_success_cnt = planner_success_tmp["success_list"]
            planner_success.append([0] + planner_success_cnt)

    planner_success_np = np.array(planner_success)
    planner_success_mean = np.mean(planner_success_np, axis=0)
    planner_success_std = np.std(planner_success_np, axis=0)
    print(planner_success_mean, planner_success_std)

    p = ax1.plot(planning_time, planner_success_mean, "o-", label=legends[i], color=utils_11d.c_map[legends[i]])
    ax1.fill_between(
        planning_time,
        (planner_success_mean - planner_success_std),
        (planner_success_mean + planner_success_std),
        color=p[0].get_color(),
        alpha=0.1,
    )


handles, labels = ax1.get_legend_handles_labels()
# order = [0, 1, 2, 3]
# ax2.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='center left', bbox_to_anchor=(1.04, 0.5), fancybox=True, shadow=True)
# ax1.legend(loc="lower center", ncols=len(planners), bbox_to_anchor=(0.5, -0.22), fancybox=True, shadow=True)
ax1.legend(loc="lower right", ncols=2, fancybox=True, shadow=True, fontsize=9)
ax1.set_title("Real world 11D-Fetch", fontsize=18)
ax1.set_ylim([0, 10.5])
ax1.set_yticks(np.arange(0, 10.1, 2))
ax1.set_yticks(np.arange(0, 10, 1), minor=True)  # set minor ticks on y-axis
ax1.set_xticks(np.arange(11))
ax1.set_xticks(planning_time, minor=True)  # set minor ticks on x-axis
ax1.grid()
ax1.grid(which="minor", alpha=0.3)
ax1.set_xlabel("Planning time (s)", fontsize=18)
ax1.set_ylabel("Planning Success Count", fontsize=18)

plt.savefig(osp.join(CUR_DIR, "planning_res_timing_rls.png"), bbox_inches="tight")
