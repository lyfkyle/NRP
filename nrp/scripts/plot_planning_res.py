import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math
import torch

# print(math.ceil(4.0 / 0.2))

fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols=2, figsize=(12, 5)) # unpack
# plt.suptitle("Success rate comparison")

num_extensions = np.arange(0, 301, 25)
rrt_success = np.array([0, 8, 13, 16, 19, 22, 23, 25, 28, 31, 35, 35, 39])
rrt_is_success = np.array([0, 44, 54, 67, 81, 85, 88, 95, 105, 110, 112, 115, 118])
cvae_success = np.array([0, 32, 38, 41, 48, 51, 58, 64, 68, 70, 74, 75, 77])
next_success = np.array([0, 28, 53, 62, 69, 75, 81, 86, 89, 92, 96, 97, 100])
rrt_ne_d_success = np.array([0, 109, 139, 154, 166, 175, 182, 184, 186, 187, 189, 193, 195])
rrt_ne_g_success = np.array([0, 95, 128, 147, 164, 167, 172, 180, 186, 188, 192, 192, 192])
ax1.plot(num_extensions, np.array(rrt_success) / 250, 'o-', label='RRT')
ax1.plot(num_extensions, np.array(rrt_is_success) / 250, 'o-', label='RRT-IS')
ax1.plot(num_extensions, np.array(cvae_success) / 250, 'o-', label='CVAE')
ax1.plot(num_extensions, np.array(next_success) / 250, 'o-', label='NEXT')
ax1.plot(num_extensions, np.array(rrt_ne_d_success) / 250, 'o-', label='RRT-NE-d')
ax1.plot(num_extensions, np.array(rrt_ne_g_success) / 250, 'o-', label='RRT-NE-g')
ax1.set_title("Planning Success Rate Against Extension Number")
ax1.set_ylim([0, 1.0])
ax1.set_xticks(num_extensions)
ax1.set_xlabel("Number of extensions")
ax1.set_ylabel("Success rate")

planning_time = np.arange(0, 3.1, 0.2)
rrt_success = np.array([0, 34, 53, 59, 63, 65, 69, 72, 72, 73, 73, 74, 75, 78, 78, 78])
rrt_is_success = np.array([0, 103, 123, 136, 143, 147, 154, 158, 160, 165, 167, 167, 171, 173, 174, 175])
cvae_success = np.array([0, 58, 72, 80, 87, 88, 90, 95, 99, 100, 103, 104, 106, 108, 109, 112])
next_success = np.array([0, 60, 79, 88, 98, 108, 109, 118, 120, 123, 124, 125, 127, 128, 131, 135])
rrt_ne_d_success = np.array([0, 146, 160, 173, 182, 185, 190, 190, 192, 194, 196, 197, 203, 206, 210, 210])
rrt_ne_g_success = np.array([0, 130, 159, 174, 186, 190, 196, 200, 203, 204, 206, 207, 210, 210, 210, 211])
ax2.plot(planning_time, np.array(rrt_success) / 250, 'o-', label='RRT')
ax2.plot(planning_time, np.array(rrt_is_success) / 250, 'o-', label='RRT-IS')
ax2.plot(planning_time, np.array(cvae_success) / 250, 'o-', label='CVAE')
ax2.plot(planning_time, np.array(next_success) / 250, 'o-', label='NEXT')
# ax2.plot(planning_time, np.array(rrt_ne_g_global_success) / 250, 'o-', label='RRT-NE-global-g')
ax2.plot(planning_time, np.array(rrt_ne_d_success) / 250, 'o-', label='RRT-NE-d')
ax2.plot(planning_time, np.array(rrt_ne_g_success) / 250, 'o-', label='RRT-NE-g')
ax2.legend(loc='center left', bbox_to_anchor=(1.04, 0.5), fancybox=True, shadow=True)
ax2.set_title("Planning Success Rate Against Planning Time")
ax2.set_ylim([0, 1.0])
ax2.set_xticks(planning_time)
ax2.set_xlabel("Planning time (s)")
ax2.set_ylabel("Success rate")

# # # plt.show()
# plt.savefig("res.png")

# Extension success rate:
# rrt = 0.005
# neural_N1 = 0.7
# neural_N3 = 0.47
# neural_linkpos_N1 = 0.85
# neural_linkpos_N3 = 0.69
# ext_success_rate = [rrt, neural_N1, neural_N3, neural_linkpos_N1, neural_linkpos_N3]

# fig = plt.subplots()
# p1 = plt.bar(["rrt", "neural N=1", "neural N=3", "n_lp N=1", "n_lp N=3"], ext_success_rate)
# plt.title('Extension success rate')
# plt.ylabel('Success rate')
# # plt.show()
# plt.savefig("ext_success_rate.png")
# plt.legend((p1[0], p2[0]), ('collision check', 'neural network'))

# # time plot
# N = 3
# colision_time = np.array([0.000926, 0.00262, 0.00228])
# neural_time = np.array([0, 0.00215, 0.00206])
# vertex_selection_time = np.array([0.0016, 0.00069, 0.00076])
# # neural_time_N3 = (0, 0.008)
# # neural_time_linkpos_N1 = (0, 0.008)
# # neural_time_linkpos_N3 = (0, 0.008)
# ind = np.arange(N)

# # fig = plt.subplots()
# p1 = ax3.bar(ind, colision_time)
# p2 = ax3.bar(ind, neural_time, bottom = colision_time)
# p3 = ax3.bar(ind, vertex_selection_time, bottom = neural_time + colision_time)
# # p4 = plt.bar(ind, neural_time_N1, bottom = colision_time)

# ax3.set_title('Time analysis per extension')
# ax3.set_ylabel('Time spent (s)')
# ax3.set_xticks(ind, ('RRT', 'RRT-NE', "RRT-NE-CVAE"))
# ax3.legend((p1[0], p2[0], p3[0]), ('collision check', 'neural network', 'vertex selection'))

plt.savefig("planning_res.png", bbox_inches="tight")

