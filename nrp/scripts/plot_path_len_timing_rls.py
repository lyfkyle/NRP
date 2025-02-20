import os.path as osp
import json
import numpy as np
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use("Agg")


from nrp import ROOT_DIR
from nrp.env.rls import utils


CUR_DIR = osp.dirname(osp.abspath(__file__))

planners = [
    "nrp_d_star",
    "nrp_g_star",
    "rrt_star",
    "bit_star",
    "decomposed_star",
]
legends = ["NRP*-d", "NRP*-g", "IRRT*", "BIT*", "Decomposed"]


# test_time
fig, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))  # unpack
orig_planning_time = np.arange(0.5, 10.1, 0.5)
planning_time = np.arange(0, 10.1, 0.5)
for i, planner in enumerate(planners):
    path_cost = []
    for repeat in range(10):
        with open(
            osp.join(
                ROOT_DIR,
                "test/eval_res/rls/path_len/test_time/{}/res_{}.json".format(
                    planner, repeat
                ),
            ),
            "r",
        ) as f:
            path_cost_tmp = json.load(f)

        # path_cost.append([1.0 / (path_cost_tmp[str(j)] + 1e-3) for j in range(len(orig_planning_time))])
        path_cost.append(
            [0] + [path_cost_tmp[str(j)] for j in range(len(orig_planning_time))]
        )

    path_cost_np = np.array(path_cost)
    path_cost_mean = np.mean(path_cost_np, axis=0)
    path_cost_std = np.std(path_cost_np, axis=0)

    # only get the mean and std for successful paths
    # path_cost_np = path_cost_np.transpose()
    # path_cost_mean = []
    # path_cost_std = []
    # for col in path_cost_np:
    #     col_non_zero = col[col!=0]
    #     if len(col_non_zero) == 0:
    #         path_cost_mean.append(0)
    #         path_cost_std.append(0)
    #     else:
    #         path_cost_mean.append(np.mean(col_non_zero))
    #         path_cost_std.append(np.std(col_non_zero))
    # path_cost_mean = np.array(path_cost_mean)
    # path_cost_std = np.array(path_cost_std)
    print(path_cost_mean, path_cost_std)

    p = ax2.plot(
        planning_time,
        path_cost_mean,
        "o-",
        label=legends[i],
        color=utils.c_map[legends[i]],
    )
    ax2.fill_between(
        planning_time,
        (path_cost_mean - path_cost_std),
        (path_cost_mean + path_cost_std),
        color=p[0].get_color(),
        alpha=0.1,
    )

handles, labels = ax2.get_legend_handles_labels()
# order = [0, 1, 2, 3]
# ax2.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='center left', bbox_to_anchor=(1.04, 0.5), fancybox=True, shadow=True)
# ax1.legend(loc="lower center", ncols=len(planners), bbox_to_anchor=(0.5, -0.22), fancybox=True, shadow=True)
# ax2.legend(loc="lower center", ncols=len(time_planners), bbox_to_anchor=(-0.1, -0.3), fancybox=True, shadow=True, fontsize=14)
ax2.legend(loc="lower right", ncols=2, fancybox=True, shadow=True, fontsize=9)
# ax2.set_title("Relative Path Optimality vs. Time", fontsize=15)
ax2.set_title("Real world 11D-Fetch", fontsize=18)
ax2.set_yticks(np.arange(0, 1.1, 0.2))
ax2.set_yticks(np.arange(0, 1.1, 0.1), minor=True)  # set minor ticks on y-axis
ax2.set_xticks(np.arange(11))
ax2.set_xticks(np.arange(0, 10.1, 0.5), minor=True)  # set minor ticks on x-axis
ax2.grid()
ax2.grid(which="minor", alpha=0.3)
ax2.set_xlabel("Planning time (s)", fontsize=18)
ax2.set_ylabel("Relative path optimality", fontsize=18)

# fig.suptitle("Real world 11D-Fetch", fontsize=22)
# plt.savefig(osp.join(CUR_DIR, "path_cost_optimal_success_weighted.png"), bbox_inches="tight")
plt.savefig(osp.join(CUR_DIR, "path_cost_optimal_timing_rls.png"), bbox_inches="tight")
