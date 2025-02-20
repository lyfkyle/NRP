import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math
import torch

# print(math.ceil(4.0 / 0.2))

fig, (ax1, ax2, ax3) = plt.subplots(nrows = 1, ncols=3, figsize=(18, 5)) # unpack
# plt.suptitle("Success rate comparison")

num_sel_res_time = [
    [118, 139, 155, 165, 172, 177, 180, 183, 183, 184],
    [115, 138, 153, 155, 160, 167, 172, 176, 177, 178],
    [103, 128, 139, 148, 156, 162, 166, 171, 175, 177],
    [106, 129, 140, 147, 153, 155, 163, 168, 172, 173],
    [104, 129, 141, 146, 152, 159, 167, 171, 173, 175],
]
num_sel_res_ext = [
    [113, 139, 158, 169, 175, 182, 184, 191, 197, 200, 202, 204],
    [119, 147, 162, 170, 179, 189, 192, 193, 196, 201, 203, 204],
    [131, 152, 164, 172, 178, 188, 194, 195, 199, 200, 203, 204],
    [125, 147, 164, 170, 181, 185, 192, 193, 195, 201, 201, 203],
    [121, 146, 166, 175, 184, 187, 192, 196, 198, 200, 205, 206],
]
num_sels = np.arange(1, 6)
num_extensions = np.arange(25, 301, 25)
time_allowed = np.arange(1, 11)

for i, num_sel in enumerate(num_sels):
    ax1.plot(num_extensions, np.array(num_sel_res_ext[i]) / 250, 'o-', label=f'NE-{num_sel}')

ax1.legend()
ax1.set_title("RRT-NE: Planning success rate against number of extensions")
ax1.set_ylim([0.4, 1.0])
ax1.set_xlabel("Number of extensions")
ax1.set_ylabel("Success rate")

for i, num_sel in enumerate(num_sels):
    ax2.plot(time_allowed, np.array(num_sel_res_time[i]) / 250, 'o-', label=f'NE-{num_sel}')

ax2.legend()
ax2.set_title("RRT-NE: Planning success rate against number of time")
ax2.set_ylim([0.4, 1.0])
ax2.set_xlabel("Allowed planning time(s)")
ax2.set_ylabel("Success rate")

N = 2
timing_res = [
    [0.0241, 0.0081, 0.0160],
    [0.0373, 0.0157, 0.0216],
    [0.0483, 0.0233, 0.0250],
    [0.0600, 0.0306, 0.0295],
    [0.0710, 0.0378, 0.0331],
]

collision_time = [t[2] for t in timing_res]
neural_time = [t[1] for t in timing_res]

ind = ["RRT-NE-{}".format(i) for i in range(1, 6)]

# fig = plt.subplots()
p1 = ax3.bar(ind, collision_time)
p2 = ax3.bar(ind, neural_time, bottom = collision_time)

ax3.set_title('Time analysis per extension')
ax3.set_ylabel('Time spent (s)')
ax3.legend((p1[0], p2[0]), ('collision check', 'neural network'))

plt.savefig("num_ne.png")