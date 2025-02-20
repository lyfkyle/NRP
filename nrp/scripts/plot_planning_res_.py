import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math
import json
import os.path as osp

CUR_DIR = osp.dirname(osp.abspath(__file__))

# fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols=2, figsize=(12, 5)) # unpack
# # plt.suptitle("Success rate comparison")

# num_extensions = np.arange(0, 301, 25)
# col_only = np.array([0, 83, 114, 125, 136, 143, 151, 158, 158, 162, 164, 164, 165])
# sel_only = np.array([0, 81, 115, 136, 146, 154, 162, 167, 169, 177, 183, 185, 189])
# rrt_ne_d_success = np.array([0, 109, 139, 154, 166, 175, 182, 184, 186, 187, 189, 193, 195])
# rrt_ne_g_success = np.array([0, 95, 128, 147, 164, 167, 172, 180, 186, 188, 192, 192, 192])
# rrt_ne_d_global_success = np.array([0, 35, 58, 75, 89, 96, 103, 109, 112, 119, 124, 125, 127])
# rrt_ne_g_global_success = np.array([0, 60, 84, 106, 123, 134, 144, 151, 157, 160, 167, 169, 170])
# ax1.plot(num_extensions, np.array(col_only) / 250, 'o-', label='RRT-NE-d-Col')
# ax1.plot(num_extensions, np.array(sel_only) / 250, 'o-', label='RRT-NE-d-Sel')
# ax1.plot(num_extensions, np.array(rrt_ne_d_success) / 250, 'o-', label='RRT-NE-d')
# # ax1.plot(num_extensions, np.array(rrt_ne_g_success) / 250, 'o-', label='RRT-NE-g')
# # ax1.plot(num_extensions, np.array(rrt_ne_d_global_success) / 250, 'o-', label='RRT-NE-d-Global')
# # ax1.plot(num_extensions, np.array(rrt_ne_g_global_success) / 250, 'o-', label='RRT-NE-g-Global')
# ax1.set_title("Planning Success Rate Against Extension Number")
# ax1.set_ylim([0, 1.0])
# ax1.set_xticks(num_extensions)
# ax1.set_xlabel("Number of extensions")
# ax1.set_ylabel("Success rate")

# planning_time = np.arange(0, 3.1, 0.2)
# col_only = np.array([0, 131, 151, 162, 173, 177, 183, 187, 189, 190, 191, 193, 193, 194, 195, 196])
# sel_only = np.array([0, 138, 168, 181, 185, 187, 191, 196, 198, 200, 200, 200, 201, 201, 201, 202])
# rrt_ne_d_success = np.array([0, 146, 160, 173, 182, 185, 190, 190, 192, 194, 196, 197, 203, 206, 210, 210])
# rrt_ne_g_success = np.array([0, 142, 167, 183, 190, 196, 198, 199, 199, 204, 206, 208, 208, 210, 211, 212])
# rrt_ne_d_global_success = np.array([0, 50, 74, 94, 113, 121, 127, 135, 137, 142, 150, 151, 154, 157, 158, 161])
# rrt_ne_g_global_success = np.array([0, 81, 129, 145, 163, 169, 174, 178, 183, 188, 190, 191, 192, 196, 200, 200])
# ax2.plot(planning_time, np.array(col_only) / 250, 'o-', label='RRT-NE-d-Col')
# ax2.plot(planning_time, np.array(sel_only) / 250, 'o-', label='RRT-NE-d-Sel')
# ax2.plot(planning_time, np.array(rrt_ne_d_success) / 250, 'o-', label='RRT-NE-d')
# # ax2.plot(planning_time, np.array(rrt_ne_g_success) / 250, 'o-', label='RRT-NE-g')
# # ax2.plot(planning_time, np.array(rrt_ne_d_global_success) / 250, 'o-', label='RRT-NE-d-Global')
# # ax2.plot(planning_time, np.array(rrt_ne_g_global_success) / 250, 'o-', label='RRT-NE-g-Global')
# ax2.legend(loc='center left', bbox_to_anchor=(1.04, 0.5), fancybox=True, shadow=True)
# ax2.set_title("Planning Success Rate Against Time")
# ax2.set_ylim([0, 1.0])
# ax2.set_xlabel("Planning time (s)")
# ax2.set_ylabel("Success rate")

# plt.savefig("planning_res_ablation.png", bbox_inches="tight")

fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols=2, figsize=(12, 5)) # unpack
# plt.suptitle("Success rate comparison")

planners = ["neural_d_global", "neural_g_global", "neural_d", "neural_g"]
legends = ["RRT-NE-d-Global", "RRT-NE-g-Global", "RRT-NE-d", "RRT-NE-g"]

num_extensions = np.arange(0, 301, 25)
for i, planner in enumerate(planners):
    planner_success = []
    for repeat in range(10):
        with open(osp.join(CUR_DIR, "planner/eval_res/test_ext/{}/result_{}.json".format(planner, repeat)), "r") as f:
            planner_success_tmp = json.load(f)
            planner_success_cnt = planner_success_tmp["success_list"]
            planner_success.append([0] + planner_success_cnt)

    planner_success_np = np.array(planner_success)
    planner_success_mean = np.mean(planner_success_np, axis=0)
    planner_success_std = np.std(planner_success_np, axis=0)
    print(planner_success_mean, planner_success_std)

    p = ax1.plot(num_extensions, planner_success_mean / 250, 'o-', label=legends[i])
    ax1.fill_between(num_extensions, (planner_success_mean-planner_success_std) / 250, (planner_success_mean+planner_success_std) / 250, color=p[0].get_color(), alpha=.1)

ax1.set_title("Planning Success Rate Against Extension Number")
ax1.set_ylim([0, 1.0])
ax1.set_xticks(num_extensions)
ax1.set_xlabel("Number of extensions")
ax1.set_ylabel("Success rate")

planning_time = np.arange(0, 3.1, 0.2)
for i, planner in enumerate(planners):
    planner_success = []
    for repeat in range(10):
        with open(osp.join(CUR_DIR, "planner/eval_res/test_time/{}/result_{}.json".format(planner, repeat)), "r") as f:
            planner_success_tmp = json.load(f)
            planner_success_cnt = planner_success_tmp["success_list"]
            planner_success.append([0] + planner_success_cnt)

    planner_success_np = np.array(planner_success)
    planner_success_mean = np.mean(planner_success_np, axis=0)
    planner_success_std = np.std(planner_success_np, axis=0)
    print(planner_success_mean, planner_success_std)

    p = ax2.plot(planning_time, planner_success_mean / 250, 'o-', label=legends[i])
    ax2.fill_between(planning_time, (planner_success_mean-planner_success_std) / 250, (planner_success_mean+planner_success_std) / 250, color=p[0].get_color(), alpha=.1)

ax2.legend(loc='center left', bbox_to_anchor=(1.04, 0.5), fancybox=True, shadow=True)
ax2.set_title("Planning Success Rate Against Planning Time")
ax2.set_ylim([0, 1.0])
ax2.set_xticks(planning_time)
ax2.set_xlabel("Planning time (s)")
ax2.set_ylabel("Success rate")

# ax2.legend(loc='center left', bbox_to_anchor=(1.04, 0.5), fancybox=True, shadow=True)
# ax2.set_title("Planning Success Rate Against Planning Time")
# ax2.set_ylim([0, 1.0])
# ax2.set_xticks(planning_time)
# ax2.set_xlabel("Planning time (s)")
# ax2.set_ylabel("Success rate")
# col_only = np.array([0, 83, 114, 125, 136, 143, 151, 158, 158, 162, 164, 164, 165])
# sel_only = np.array([0, 81, 115, 136, 146, 154, 162, 167, 169, 177, 183, 185, 189])
# rrt_ne_d_success = np.array([0, 109, 139, 154, 166, 175, 182, 184, 186, 187, 189, 193, 195])
# rrt_ne_g_success = np.array([0, 95, 128, 147, 164, 167, 172, 180, 186, 188, 192, 192, 192])
# rrt_ne_d_global_success = np.array([0, 35, 58, 75, 89, 96, 103, 109, 112, 119, 124, 125, 127])
# rrt_ne_g_global_success = np.array([0, 60, 84, 106, 123, 134, 144, 151, 157, 160, 167, 169, 170])
# ax1.plot(num_extensions, np.array(rrt_ne_d_success) / 250, 'o-', label='RRT-NE-d')
# ax1.plot(num_extensions, np.array(rrt_ne_g_success) / 250, 'o-', label='RRT-NE-g')
# ax1.plot(num_extensions, np.array(rrt_ne_d_global_success) / 250, 'o-', label='RRT-NE-d-Global')
# ax1.plot(num_extensions, np.array(rrt_ne_g_global_success) / 250, 'o-', label='RRT-NE-g-Global')
# ax1.set_title("Planning Success Rate Against Extension Number")
# ax1.set_ylim([0, 1.0])
# ax1.set_xticks(num_extensions)
# ax1.set_xlabel("Number of extensions")
# ax1.set_ylabel("Success rate")

# col_only = np.array([0, 131, 151, 162, 173, 177, 183, 187, 189, 190, 191, 193, 193, 194, 195, 196])
# sel_only = np.array([0, 138, 168, 181, 185, 187, 191, 196, 198, 200, 200, 200, 201, 201, 201, 202])
# rrt_ne_d_success = np.array([0, 146, 160, 173, 182, 185, 190, 190, 192, 194, 196, 197, 203, 206, 210, 210])
# rrt_ne_g_success = np.array([0, 142, 167, 183, 190, 196, 198, 199, 199, 204, 206, 208, 208, 210, 211, 212])
# rrt_ne_d_global_success = np.array([0, 50, 74, 94, 113, 121, 127, 135, 137, 142, 150, 151, 154, 157, 158, 161])
# rrt_ne_g_global_success = np.array([0, 81, 129, 145, 163, 169, 174, 178, 183, 188, 190, 191, 192, 196, 200, 200])
# ax2.plot(planning_time, np.array(rrt_ne_d_success) / 250, 'o-', label='RRT-NE-d')
# ax2.plot(planning_time, np.array(rrt_ne_g_success) / 250, 'o-', label='RRT-NE-g')
# ax2.plot(planning_time, np.array(rrt_ne_d_global_success) / 250, 'o-', label='RRT-NE-d-Global')
# ax2.plot(planning_time, np.array(rrt_ne_g_global_success) / 250, 'o-', label='RRT-NE-g-Global')
# ax2.legend(loc='center left', bbox_to_anchor=(1.04, 0.5), fancybox=True, shadow=True)
# ax2.set_title("Planning Success Rate Against Time")
# ax2.set_ylim([0, 1.0])
# ax2.set_xlabel("Planning time (s)")
# ax2.set_ylabel("Success rate")

plt.savefig("planning_res_local_vs_global.png", bbox_inches="tight")

