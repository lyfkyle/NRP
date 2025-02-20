import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math
import torch

# print(math.ceil(4.0 / 0.2))

fig, (ax1, ax2, ax3) = plt.subplots(nrows = 1, ncols=3, figsize=(18, 5)) # unpack
# plt.suptitle("Success rate comparison")

base_goal_bias_res = [
    [25, 55, 66, 72, 83, 91, 97, 103, 107, 110, 117, 122, 0, 0, 0],
    [37, 55, 66, 78, 85, 92, 99, 106, 113, 118, 122, 126, 0, 0, 0],
    [45, 65, 78, 86, 94, 100, 107, 116, 119, 120, 121, 125, 0, 0, 0],
    [37, 58, 69, 81, 86, 91, 99, 105, 109, 110, 112, 117, 0, 0, 0],
    [39, 51, 60, 66, 70, 74, 78, 82, 89, 94, 97, 100, 0, 0, 0],
    [32, 43, 52, 61, 64, 69, 75, 79, 84, 88, 89, 90, 0, 0, 0],
    [28, 43, 51, 57, 65, 67, 73, 76, 81, 84, 87, 88, 0, 0, 0],
    [36, 44, 49, 57, 60, 62, 64, 67, 72, 73, 77, 78, 0, 0, 0],
    [21, 27, 32, 41, 43, 43, 46, 49, 57, 58, 59, 59, 0, 0, 0],
]
neural_goal_bias_res = [
    [67, 102, 122, 136, 147, 159, 164, 173, 179, 187, 190, 191, 0, 0, 0],
    [86, 124, 144, 163, 168, 173, 178, 181, 184, 187, 191, 191, 0, 0, 0],
    [98, 132, 150, 165, 169, 179, 184, 190, 192, 195, 196, 198, 0, 0, 0],
    [107, 133, 149, 161, 170, 174, 178, 181, 184, 189, 192, 196, 0, 0, 0],
    [88, 123, 152, 159, 168, 172, 177, 182, 185, 186, 190, 193, 0, 0, 0],
    [100, 125, 144, 155, 167, 170, 173, 175, 177, 179, 184, 186, 0, 0, 0],
    [106, 126, 145, 153, 160, 166, 168, 175, 179, 185, 187, 190, 0, 0, 0],
    [106, 125, 139, 149, 158, 161, 163, 168, 168, 172, 175, 176, 0, 0, 0],
    [111, 131, 139, 146, 150, 156, 158, 161, 162, 165, 166, 168, 0, 0, 0],
]
cvae_goal_bias_res = [
    [51, 97, 125, 140, 156, 161, 165, 170, 176, 183, 188, 191, 0, 0, 0],
    [65, 118, 144, 161, 171, 178, 184, 190, 193, 195, 195, 199, 0, 0, 0],
    [84, 124, 143, 152, 158, 164, 170, 178, 185, 188, 195, 197, 0, 0, 0],
    [95, 130, 146, 162, 172, 181, 185, 187, 189, 190, 196, 199, 0, 0, 0],
    [96, 133, 150, 159, 167, 175, 180, 182, 184, 186, 190, 191, 0, 0, 0],
    [98, 136, 156, 167, 171, 176, 180, 182, 183, 183, 183, 186, 0, 0, 0],
    [102, 139, 150, 161, 171, 177, 178, 180, 180, 181, 184, 185, 0, 0, 0],
    [104, 124, 142, 150, 154, 164, 167, 170, 173, 175, 177, 178, 0, 0, 0],
    [108, 128, 143, 151, 159, 163, 169, 169, 169, 170, 170, 172, 0, 0, 0],
]
goal_biases = np.arange(0.1, 0.9, 0.1)
num_extensions = np.arange(25, 301, 25)

for i, goal_bias in enumerate(goal_biases):
    ax1.plot(num_extensions, np.array(base_goal_bias_res[i])[:len(num_extensions)] / 250, 'o-', label=f'Goal-bias-{goal_bias:.1f}')
ax1.legend()
ax1.set_title("RRT: Goal bias test")
# ax1.set_ylim([0, 1.0])
ax1.set_xlabel("Number of extensions")
ax1.set_ylabel("Success rate")

for i, goal_bias in enumerate(goal_biases):
    ax2.plot(num_extensions, np.array(neural_goal_bias_res[i])[:len(num_extensions)] / 250, 'o-', label=f'Goal-bias-{goal_bias:.1f}')
ax2.legend()
ax2.set_title("RRT-NE: Goal bias test")
# ax2.set_ylim([0, 1.0])
ax2.set_xlabel("Number of extensions")
ax2.set_ylabel("Success rate")

for i, goal_bias in enumerate(goal_biases):
    ax3.plot(num_extensions, np.array(cvae_goal_bias_res[i])[:len(num_extensions)] / 250, 'o-', label=f'Goal-bias-{goal_bias:.1f}')
ax3.legend()
ax3.set_title("RRT-NE-CVAE: Goal bias test")
# ax3.set_ylim([0, 1.0])
ax3.set_xlabel("Number of extensions")
ax3.set_ylabel("Success rate")

plt.savefig("goal_bias_ext.png")

fig, (ax1, ax2, ax3) = plt.subplots(nrows = 1, ncols=3, figsize=(18, 5)) # unpack
# plt.suptitle("Success rate comparison")

base_goal_bias_res = [
    [125, 145, 156, 162, 168, 170, 173, 175, 178, 180, 184, 184, 185, 188, 0],
    [110, 133, 145, 150, 156, 163, 166, 166, 169, 171, 172, 175, 177, 177, 0],
    [114, 134, 148, 160, 168, 171, 175, 177, 179, 181, 182, 185, 186, 187, 0],
    [119, 141, 155, 158, 162, 165, 167, 168, 172, 175, 177, 178, 181, 183, 0],
    [101, 122, 135, 147, 151, 154, 159, 161, 162, 163, 163, 164, 166, 167, 0],
    [100, 116, 127, 141, 147, 154, 160, 164, 169, 170, 171, 173, 174, 175, 0],
    [90, 113, 125, 134, 139, 147, 148, 152, 153, 156, 158, 160, 162, 164, 0],
    [86, 109, 118, 126, 129, 136, 139, 143, 145, 146, 148, 152, 153, 154, 0],
    [69, 89, 100, 105, 113, 123, 124, 125, 128, 131, 134, 140, 142, 143, 0],
]
neural_goal_bias_res = [
    [118, 152, 170, 182, 187, 192, 192, 194, 197, 201, 204, 204, 205, 205, 0],
    [125, 158, 174, 189, 195, 202, 205, 208, 209, 209, 210, 212, 212, 214, 0],
    [138, 163, 176, 187, 187, 192, 198, 200, 203, 203, 203, 203, 203, 203, 0],
    [142, 165, 174, 184, 186, 189, 192, 196, 198, 199, 202, 204, 205, 207, 0],
    [142, 165, 178, 184, 188, 189, 192, 192, 195, 195, 198, 199, 200, 201, 0],
    [139, 157, 175, 182, 183, 186, 190, 194, 194, 196, 198, 199, 202, 204, 0],
    [141, 161, 172, 174, 177, 181, 185, 187, 190, 191, 196, 197, 200, 201, 0],
    [129, 151, 160, 168, 170, 174, 177, 179, 182, 183, 185, 185, 186, 188, 0],
    [131, 150, 163, 167, 171, 175, 180, 182, 183, 183, 184, 188, 190, 192, 0],
]
cvae_goal_bias_res = [
    [89, 139, 168, 183, 188, 189, 197, 198, 201, 201, 202, 203, 205, 206, 0],
    [115, 148, 170, 181, 188, 194, 199, 201, 203, 204, 209, 210, 210, 211, 0],
    [134, 171, 179, 185, 188, 192, 196, 198, 201, 202, 203, 204, 205, 205, 0],
    [139, 163, 183, 190, 195, 195, 198, 201, 205, 205, 207, 207, 209, 209, 0],
    [126, 155, 173, 181, 185, 190, 194, 199, 200, 200, 202, 202, 202, 203, 0],
    [127, 154, 172, 184, 190, 192, 197, 198, 198, 199, 200, 201, 204, 205, 0],
    [132, 159, 172, 180, 184, 190, 190, 191, 193, 194, 197, 197, 198, 199, 0],
    [140, 161, 170, 176, 180, 187, 188, 190, 192, 194, 194, 195, 195, 197, 0],
    [141, 161, 167, 173, 177, 180, 181, 182, 183, 185, 185, 186, 189, 189, 0],
]
goal_biases = np.arange(0.1, 0.9, 0.1)
time_allowed = np.arange(0.2, 2.9, 0.2)

for i, goal_bias in enumerate(goal_biases):
    ax1.plot(time_allowed, np.array(base_goal_bias_res[i])[:len(time_allowed)] / 250, 'o-', label=f'Goal-bias-{goal_bias:.1f}')
ax1.legend()
ax1.set_title("RRT: Goal bias test")
# ax1.set_ylim([0, 1.0])
ax1.set_xlabel("Allowed planning time(s)")
ax1.set_ylabel("Success rate")

for i, goal_bias in enumerate(goal_biases):
    ax2.plot(time_allowed, np.array(neural_goal_bias_res[i])[:len(time_allowed)] / 250, 'o-', label=f'Goal-bias-{goal_bias:.1f}')
ax2.legend()
ax2.set_title("RRT-NE: Goal bias test")
# ax2.set_ylim([0, 1.0])
ax2.set_xlabel("Allowed planning time(s)")
ax2.set_ylabel("Success rate")

for i, goal_bias in enumerate(goal_biases):
    ax3.plot(time_allowed, np.array(cvae_goal_bias_res[i])[:len(time_allowed)] / 250, 'o-', label=f'Goal-bias-{goal_bias:.1f}')
ax3.legend()
ax3.set_title("RRT-NE-CVAE: Goal bias test")
# ax3.set_ylim([0, 1.0])
ax3.set_xlabel("Number of extensions")
ax3.set_ylabel("Success rate")

plt.savefig("goal_bias_time.png")

