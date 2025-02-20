import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math
import torch

# print(math.ceil(4.0 / 0.2))

fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols=2, figsize=(13, 5)) # unpack
# plt.suptitle("Success rate comparison")

# planning_time = np.arange(1, 11)
planning_time = np.arange(0.2, 3.1, 0.2)
col_only = np.array([131, 151, 162, 173, 177, 183, 187, 189, 190, 191, 193, 193, 194, 195, 196])
sel_only = np.array([138, 168, 181, 185, 187, 191, 196, 198, 200, 200, 200, 201, 201, 201, 202])
rrt_ne_success = np.array([138, 164, 180, 184, 189, 191, 194, 197, 199, 202, 204, 206, 207, 209, 210])
# learnt_success_linkinfo_N3 = np.array([83,111,130,139,150,158,161,166,170,174])
ax1.plot(planning_time, np.array(col_only) / 250, 'o-', label='RRT-NE-ColOnly')
ax1.plot(planning_time, np.array(sel_only) / 250, 'o-', label='RRT-NE-SelOnly')
ax1.plot(planning_time, np.array(rrt_ne_success) / 250, 'o-', label='RRT-NE')
# ax1.plot(planning_time, np.array(learnt_success_linkinfo_N3) / 250, 'o-', label='RRT-NE-3')
ax1.legend()
ax1.set_title("Planning success rate against time")
ax1.set_xlabel("Planning time (s)")
ax1.set_ylabel("Success rate")

num_extensions = np.arange(25, 301, 25)
col_only = np.array([83, 114, 125, 136, 143, 151, 158, 158, 162, 164, 164, 165])
sel_only = np.array([81, 115, 136, 146, 154, 162, 167, 169, 177, 183, 185, 189])
rrt_ne_success = np.array([100, 121, 144, 155, 160, 170, 176, 177, 182, 185, 187, 191])
ax2.plot(num_extensions, np.array(col_only) / 250, 'o-', label='RRT-NE-ColOnly')
ax2.plot(num_extensions, np.array(sel_only) / 250, 'o-', label='RRT-NE-SelOnly')
ax2.plot(num_extensions, np.array(rrt_ne_success) / 250, 'o-', label='RRT-NE')
ax2.legend()
ax2.set_title("Planning success rate against number of extension")
ax2.set_xlabel("Number of extensions")
ax2.set_ylabel("Success rate")

plt.savefig("ablation.png")

