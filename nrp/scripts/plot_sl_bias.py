import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math
import torch

# print(math.ceil(4.0 / 0.2))

fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols=2, figsize=(12, 5)) # unpack
# plt.suptitle("Success rate comparison")

rrt_ne_sl_bias_res = [
    [93, 121, 144, 152, 159, 168, 173, 176, 181, 186, 191, 193, 0, 0, 0],
    [93, 128, 143, 156, 166, 170, 177, 182, 184, 186, 188, 190, 0, 0, 0],
    [97, 131, 154, 162, 168, 174, 177, 183, 187, 190, 194, 199, 0, 0, 0],
    [96, 129, 140, 152, 162, 170, 180, 185, 188, 189, 190, 192, 0, 0, 0],
    [87, 114, 135, 146, 153, 159, 164, 170, 178, 182, 184, 184, 0, 0, 0],
    [88, 118, 141, 151, 159, 166, 173, 180, 182, 184, 191, 195, 0, 0, 0],
    [83, 107, 126, 143, 153, 158, 163, 167, 171, 178, 182, 186, 0, 0, 0],
    [77, 103, 126, 137, 145, 156, 162, 166, 168, 174, 177, 179, 0, 0, 0],
    [73, 103, 120, 127, 131, 137, 146, 153, 154, 156, 160, 161, 0, 0, 0],
    [52, 74, 91, 107, 117, 126, 129, 138, 142, 145, 147, 155, 0, 0, 0],
    [43, 58, 67, 76, 82, 88, 93, 100, 103, 107, 107, 109, 0, 0, 0],
]
cvae_sl_bias_res = [
    [95, 128, 147, 164, 167, 172, 180, 186, 188, 192, 192, 192, 0, 0, 0],
    [98, 130, 149, 161, 166, 174, 182, 185, 191, 192, 195, 197, 0, 0, 0],
    [94, 128, 154, 165, 169, 176, 179, 184, 186, 191, 194, 195, 0, 0, 0],
    [100, 125, 144, 154, 160, 167, 173, 175, 181, 184, 186, 189, 0, 0, 0],
    [89, 117, 136, 148, 157, 160, 163, 170, 172, 177, 181, 185, 0, 0, 0],
    [87, 121, 133, 150, 159, 165, 173, 175, 178, 182, 185, 186, 0, 0, 0],
    [71, 108, 132, 143, 148, 152, 162, 169, 170, 172, 176, 183, 0, 0, 0],
    [72, 105, 127, 140, 145, 154, 157, 160, 164, 167, 169, 171, 0, 0, 0],
    [60, 98, 119, 130, 141, 150, 157, 162, 164, 171, 173, 176, 0, 0, 0],
    [52, 80, 100, 113, 123, 131, 137, 139, 144, 147, 149, 154, 0, 0, 0],
    [42, 57, 66, 72, 81, 89, 93, 95, 99, 106, 110, 113, 0, 0, 0],
]
goal_biases = np.arange(0.1, 0.9, 0.1)
num_extensions = np.arange(25, 301, 25)

for i, goal_bias in enumerate(goal_biases):
    ax1.plot(num_extensions, np.array(rrt_ne_sl_bias_res[i])[:len(num_extensions)] / 250, 'o-', label=f'SL-extension-bias-{goal_bias:.1f}')
ax1.legend()
ax1.set_title("RRT-NE: Straight-line extension bias test")
# ax1.set_ylim([0, 1.0])
ax1.set_xlabel("Number of extensions")
ax1.set_ylabel("Success rate")

for i, goal_bias in enumerate(goal_biases):
    ax2.plot(num_extensions, np.array(cvae_sl_bias_res[i])[:len(num_extensions)] / 250, 'o-', label=f'SL-extension-bias-{goal_bias:.1f}')
ax2.legend()
ax2.set_title("RRT-NE-CVAE: Straight-line extension bias test")
# ax2.set_ylim([0, 1.0])
ax2.set_xlabel("Number of extensions")
ax2.set_ylabel("Success rate")

plt.savefig("sl_bias_ext.png")

fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols=2, figsize=(12, 5)) # unpack
# plt.suptitle("Success rate comparison")

rrt_ne_sl_bias_res = [
    [137, 154, 169, 174, 182, 185, 191, 193, 197, 200, 204, 207, 208, 208, 209],
    [131, 152, 166, 179, 187, 191, 193, 197, 199, 200, 202, 202, 203, 203, 204],
    [140, 159, 174, 180, 192, 198, 201, 204, 204, 206, 208, 208, 210, 211, 212],
    [136, 161, 174, 183, 192, 197, 201, 204, 205, 206, 206, 208, 209, 210, 211],
    [142, 165, 175, 182, 191, 198, 203, 205, 208, 211, 213, 216, 217, 217, 218],
    [144, 167, 178, 186, 191, 195, 197, 200, 203, 203, 204, 206, 207, 208, 212],
    [148, 169, 180, 187, 193, 194, 201, 203, 203, 204, 205, 205, 209, 211, 211],
    [142, 177, 190, 196, 199, 203, 205, 206, 207, 207, 207, 208, 208, 208, 209],
    [143, 166, 181, 187, 191, 196, 202, 204, 204, 204, 205, 205, 209, 213, 213],
    [143, 167, 176, 186, 190, 196, 198, 200, 203, 205, 207, 208, 208, 209, 212],
    [110, 142, 149, 152, 157, 159, 163, 169, 170, 172, 173, 174, 174, 174, 174],
]
cvae_sl_bias_res = [
    [122, 158, 173, 178, 181, 184, 188, 192, 193, 198, 199, 201, 201, 203, 205],
    [131, 155, 166, 172, 181, 186, 188, 189, 191, 195, 195, 199, 200, 203, 204],
    [130, 156, 168, 175, 178, 185, 188, 193, 196, 199, 200, 204, 206, 207, 207],
    [133, 159, 171, 178, 184, 191, 195, 196, 203, 205, 205, 205, 207, 209, 210],
    [127, 150, 162, 171, 174, 181, 187, 190, 193, 194, 194, 196, 197, 198, 204],
    [142, 165, 176, 183, 188, 191, 198, 198, 202, 203, 203, 204, 204, 206, 206],
    [131, 155, 167, 177, 186, 190, 193, 196, 199, 202, 202, 204, 205, 206, 206],
    [146, 173, 180, 184, 190, 193, 196, 200, 202, 204, 204, 205, 205, 207, 208],
    [141, 157, 171, 173, 183, 187, 191, 192, 197, 201, 203, 203, 203, 203, 207],
    [133, 157, 175, 178, 183, 185, 190, 191, 195, 195, 196, 200, 202, 203, 204],
    [113, 135, 145, 151, 156, 160, 160, 164, 164, 164, 168, 168, 168, 168, 169],
]
goal_biases = np.arange(0.1, 0.9, 0.1)
time_allowed = np.arange(0.2, 3.1, 0.2)

for i, goal_bias in enumerate(goal_biases):
    ax1.plot(time_allowed, np.array(rrt_ne_sl_bias_res[i])[:len(time_allowed)] / 250, 'o-', label=f'SL-extension-bias-{goal_bias:.1f}')
ax1.legend()
ax1.set_title("RRT-NE: Straight-line extension bias test")
# ax1.set_ylim([0, 1.0])
ax1.set_xlabel("Allowed planning time(s)")
ax1.set_ylabel("Success rate")

for i, goal_bias in enumerate(goal_biases):
    ax2.plot(time_allowed, np.array(cvae_sl_bias_res[i])[:len(time_allowed)] / 250, 'o-', label=f'SL-extension-bias-{goal_bias:.1f}')
ax2.legend()
ax2.set_title("RRT-NE-CVAE: Straight-line extension bias test")
# ax2.set_ylim([0, 1.0])
ax2.set_xlabel("Allowed planning time(s)")
ax2.set_ylabel("Success rate")

plt.savefig("sl_bias_time.png")

