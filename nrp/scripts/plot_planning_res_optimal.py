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
informed_rrtstar_success = np.array([0, 8, 10, 14, 15, 17, 20, 22, 22, 23, 24, 25, 28])
bit_success = np.array([0, 62, 89, 103, 109, 115, 123, 127, 131, 133, 136, 140, 144])
rrt_ne_d_star_success = np.array([0, 92, 124, 141, 149, 155, 164, 168, 173, 178, 183, 187, 189])
rrt_ne_g_star_success = np.array([0, 78, 113, 135, 146, 154, 160, 169, 173, 176, 180, 182, 186])
ax1.plot(num_extensions, np.array(informed_rrtstar_success) / 250, 'o-', label='IRRT*')
ax1.plot(num_extensions, np.array(bit_success) / 250, 'o-', label='BIT*')
ax1.plot(num_extensions, np.array(rrt_ne_d_star_success) / 250, 'o-', label='IRRT*-NE-d')
ax1.plot(num_extensions, np.array(rrt_ne_g_star_success) / 250, 'o-', label='IRRT*-NE-g')
ax1.set_title("Planning Success Rate Against Expansions")
ax1.set_ylim([0, 1.0])
ax1.set_xticks(num_extensions)
ax1.set_xlabel("Number of vertex expansions")
ax1.set_ylabel("Success rate")

planning_time = np.arange(0.0, 3.1, 0.2)
informed_rrtstar_success = np.array([0, 29, 39, 48, 53, 56, 59, 60, 62, 65, 66, 67, 68, 68, 68, 69])
bit_success = np.array([0, 108, 131, 138, 149, 157, 164, 168, 169, 173, 176, 181, 182, 182, 188, 188])
rrt_ne_d_star_success = np.array([0, 90, 132, 150, 162, 169, 177, 181, 184, 187, 189, 190, 193, 194, 195, 195])
rrt_ne_g_star_success = np.array([0, 87, 129, 154, 169, 177, 185, 190, 193, 198, 200, 201, 203, 203, 204, 205])
ax2.plot(planning_time, np.array(informed_rrtstar_success) / 250, 'o-', label='IRRT*')
ax2.plot(planning_time, np.array(bit_success) / 250, 'o-', label='BIT*')
ax2.plot(planning_time, np.array(rrt_ne_d_star_success) / 250, 'o-', label='IRRT*-NE-d')
ax2.plot(planning_time, np.array(rrt_ne_g_star_success) / 250, 'o-', label='IRRT*-NE-g')
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

plt.savefig("planning_res_optimal.png", bbox_inches="tight")

