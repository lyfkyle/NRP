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

# ext_planners = ["neural_d_star", "neural_g_star", "rrt_star", "bit_star"]

# fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols=2, figsize=(12, 5)) # unpack

# num_extensions = np.arange(0, 501, 25)

# for i, planner in enumerate(ext_planners):
#     path_cost = []
#     for repeat in range(10):
#         with open(osp.join(CUR_DIR, "eval_res/path_len_max/test_ext/{}/res_{}.json".format(planner, repeat)), "r") as f:
#             path_cost_tmp = json.load(f)

#         # path_cost.append([1.0 / (path_cost_tmp[str(j)] + 1e-3) for j in range(len(num_extensions))])
#         path_cost.append([0] + [path_cost_tmp[str(j)] for j in range(len(num_extensions) - 1)])

#     path_cost_np = np.array(path_cost)
#     path_cost_mean = np.mean(path_cost_np, axis=0)
#     path_cost_std = np.std(path_cost_np, axis=0)

#     # only get the mean and std for successful paths
#     # path_cost_np = path_cost_np.transpose()
#     # path_cost_mean = []
#     # path_cost_std = []
#     # for col in path_cost_np:
#     #     col_non_zero = col[col!=0]
#     #     if len(col_non_zero) == 0:
#     #         path_cost_mean.append(0)
#     #         path_cost_std.append(0)
#     #     else:
#     #         path_cost_mean.append(np.mean(col_non_zero))
#     #         path_cost_std.append(np.std(col_non_zero))
#     # path_cost_mean = np.array(path_cost_mean)
#     # path_cost_std = np.array(path_cost_std)
#     print(planner, path_cost_mean, path_cost_std)

#     p = ax1.plot(num_extensions, path_cost_mean, 'o-', label=legends[i], color=c_map[legends[i]])
#     ax1.fill_between(num_extensions, (path_cost_mean-path_cost_std), (path_cost_mean+path_cost_std), color=p[0].get_color(), alpha=.1)

# ax1.set_title("Relative Path Optimality (max) vs. Expansions", fontsize=15)
# ax1.set_yticks(np.arange(0, 1.1, 0.2))
# ax1.set_yticks(np.arange(0, 1.1, 0.1), minor=True)   # set minor ticks on y-axis
# # ax1.set_yscale('log')
# ax1.set_xticks(np.arange(0, 501, 50))
# ax1.set_xticks(np.arange(0, 501, 25), minor=True)   # set minor ticks on x-axis
# # ax1.set_yticks(np.arange(0, 1.1, 0.1), minor=True)   # set minor ticks on y-axis
# ax1.grid()
# ax1.grid(which='minor', alpha=0.3)
# ax1.set_xlabel("Number of vertex expansions", fontsize=18)
# ax1.set_ylabel("Relative path optimality", fontsize=18)