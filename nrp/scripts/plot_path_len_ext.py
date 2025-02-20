import os.path as osp
import json
import numpy as np
import argparse
import matplotlib.pyplot as plt

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

orig_num_extensions = np.arange(25, 501, 25)
num_extensions = np.arange(0, 501, 25)

for i, planner in enumerate(planners):
    path_cost = []
    for repeat in range(10):
        with open(osp.join(CUR_DIR, "../test/eval_res/snake_8d/path_len/test_ext_500/{}/res_{}.json".format(planner, repeat)), "r") as f:
            path_cost_tmp = json.load(f)

        # path_cost.append([1.0 / (path_cost_tmp[str(j)] + 1e-3) for j in range(len(num_extensions))])
        path_cost.append([0] + [path_cost_tmp[str(j)] for j in range(len(orig_num_extensions))])

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

    p = ax1.plot(num_extensions, path_cost_mean, 'o-', label=legends[i], ms=4, color=c_map[legends[i]])
    ax1.fill_between(num_extensions, (path_cost_mean-path_cost_std), (path_cost_mean+path_cost_std), color=p[0].get_color(), alpha=.1)

# planners = ["neural_d_star", "neural_g_star", "cvae_star", "next", "fire_star", "rrt_star", "bit_star", "vqmpt_rrt_star"]
# legends = ["NRP*-d", "NRP*-g", "CVAE-IRRT*", "NEXT", "FIRE*", "IRRT*", "BIT*", "VQMPT*"]
planners = ["neural_d_star_no_col", "neural_g_star", "cvae_star", "next", "fire_star", "rrt_star", "bit_star_100"]
legends = ["NRP*-d", "NRP*-g", "CVAE-IRRT*", "NEXT", "FIRE*", "IRRT*", "BIT*"]

ax1.set_title("8D-Snake", fontsize=18)
ax1.set_ylim([-0.05, 1.0])
ax1.set_yticks(np.arange(0, 1.1, 0.2))
ax1.set_yticks(np.arange(0, 1.1, 0.1), minor=True)   # set minor ticks on y-axis
# ax1.set_yscale('log')
ax1.set_xticks(np.arange(0, 501, 50))
ax1.set_xticks(num_extensions, minor=True)   # set minor ticks on x-axis
ax1.grid()
ax1.grid(which='minor', alpha=0.3)
ax1.set_xlabel("Number of vertex expansions", fontsize=18)
ax1.set_ylabel("Relative path optimality", fontsize=18)

for i, planner in enumerate(planners):
    path_cost = []
    for repeat in range(10):
        with open(osp.join(CUR_DIR, "../test/eval_res/fetch_11d/path_len/test_ext_v3/{}/res_{}.json".format(planner, repeat)), "r") as f:
            path_cost_tmp = json.load(f)

        # path_cost.append([1.0 / (path_cost_tmp[str(j)] + 1e-3) for j in range(len(orig_planning_time))])
        path_cost.append([0] + [path_cost_tmp[str(j)] for j in range(len(orig_num_extensions))])

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

    p = ax2.plot(num_extensions, path_cost_mean, 'o-', label=legends[i], ms=4, color=c_map[legends[i]])
    ax2.fill_between(num_extensions, (path_cost_mean-path_cost_std), (path_cost_mean+path_cost_std), color=p[0].get_color(), alpha=.1)

handles, labels = ax2.get_legend_handles_labels()
# order = [0, 1, 2, 3]
# ax2.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='center left', bbox_to_anchor=(1.04, 0.5), fancybox=True, shadow=True)
ax2.legend(loc="lower center", ncols=(len(planners)+1)//2, bbox_to_anchor=(-0.18, -0.35), fancybox=True, shadow=True, fontsize=14)
# ax2.legend(loc="lower center", ncols=len(planners), bbox_to_anchor=(-0.1, -0.3), fancybox=True, shadow=True, fontsize=14)
ax2.set_title("11D-Fetch", fontsize=18)
ax2.set_ylim([-0.05, 1.0])
ax2.set_yticks(np.arange(0, 1.1, 0.2))
ax2.set_yticks(np.arange(0, 1.1, 0.1), minor=True)   # set minor ticks on y-axis
# ax1.set_yscale('log')
ax2.set_xticks(np.arange(0, 501, 50))
ax2.set_xticks(num_extensions, minor=True)   # set minor ticks on x-axis
ax2.grid()
ax2.grid(which='minor', alpha=0.3)
ax2.set_xlabel("Number of vertex expansions", fontsize=18)
ax2.set_ylabel("Relative path optimality", fontsize=18)

plt.savefig(osp.join(CUR_DIR, "path_cost_optimal_ext_v4.png"), bbox_inches="tight")
