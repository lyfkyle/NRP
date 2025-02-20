import matplotlib as mpl
mpl.use('Agg')
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
import math
import torch
import json
import pandas as pd
import os

CUR_DIR = osp.dirname(osp.abspath(__file__))

# time plot
N = 3

fig, axs = plt.subplots(1, 2, sharey=True)
fig.set_figwidth(15)
plt.subplots_adjust(right=0.75, top=0.85)
planner_dirs = [["rrt", "neural_d_rrt", "neural_g_rrt"], ["rrtstar", "neural_d_rrtstar", "neural_g_rrtstar"]]

base_dir = os.path.join(CUR_DIR, "planner/eval_res/timing_analysis_new")

for (ax, planner_dir) in zip(axs, planner_dirs):
    all_data = dict()

    for planner in planner_dir:
        data_path = os.path.join(base_dir, planner, "result.json")
        with open(data_path, "r") as f:
            json_dict = json.loads(f.read())
        all_data[planner] = json_dict

    df = pd.DataFrame.from_dict(all_data)
    # print(np.array(df.loc["total_extend_time"]))

    vertex_selection_success_time = np.array(df.loc["total_vertex_selection_success_time"])
    vertex_selection_fail_time = np.array(df.loc["total_vertex_selection_fail_time"])
    neural_success_time = np.array(df.loc["total_neural_extend_success_time"])
    neural_fail_time = np.array(df.loc["total_neural_extend_fail_time"])
    col_success_time = np.array(df.loc["total_col_check_success_time"])
    col_fail_time = np.array(df.loc["total_col_check_fail_time"])
    other_time = np.array(df.loc["total_other_time"])

    total_fail_time = vertex_selection_fail_time + neural_fail_time + col_fail_time

    # colision_time = np.array([0.000926, 0.0012008027904323343, 0.0046142079987176385, 0.004025924261263457])
    # neural_time = np.array([0, 0, 0.002094152528761667, 0.0014677102749400105])
    # vertex_selection_time = np.array([0.001988768799237879, 0.0025419718023270556, 0.0029491010995869563, 0.0029881566159833716])
    # other_time = np.array([0.00, 0, 0.00012193419778798887, 0.00012193419778798887])
    # neural_time_N3 = (0, 0.008)
    # neural_time_linkpos_N1 = (0, 0.008)
    # neural_time_linkpos_N3 = (0, 0.008)
    ind = np.arange(N)

    # fig = plt.subplots()
    p1 = ax.bar(ind, neural_success_time)
    p2 = ax.bar(ind, col_success_time, bottom=neural_success_time)
    p3 = ax.bar(ind, vertex_selection_success_time, bottom=col_success_time + neural_success_time)
    # p4 = ax.bar(ind, neural_fail_time, bottom=col_success_time + neural_success_time + vertex_selection_success_time)
    # p5 = ax.bar(ind, col_fail_time, bottom=col_success_time + neural_success_time + vertex_selection_success_time + neural_fail_time)
    # p6 = ax.bar(ind, vertex_selection_fail_time, bottom=col_success_time + neural_success_time + vertex_selection_success_time + neural_fail_time + col_fail_time)
    # p7 = ax.bar(ind, other_time, bottom=col_success_time + neural_success_time + vertex_selection_success_time + neural_fail_time + col_fail_time + vertex_selection_fail_time)
    p4 = ax.bar(ind, other_time, bottom=col_success_time + neural_success_time + vertex_selection_success_time)
    p5 = ax.bar(ind, total_fail_time, bottom=col_success_time + neural_success_time + vertex_selection_success_time + other_time)

plt.suptitle('Time Analysis', fontsize=15)
plt.sca(axs[0])
plt.title('RRT based')
plt.ylabel('Total time spent (s)')
plt.xticks(ind, ('RRT', 'RRT-NE-d', 'RRT-NE-g'))
plt.sca(axs[1])
plt.title('IRRT* based')
plt.xticks(ind, ('IRRT*', 'IRRT*-NE-d', 'IRRT*-NE-g'))
# plt.legend((p1[0], p2[0], p3[0], p4[0], p5[0], p6[0], p7[0]),
#            ('successful neural network', 'successful collision check', 'successful vertex selection', 'failed neural network', 'failed collision check', 'failed vertex selection', 'other'),
#            loc="center left", bbox_to_anchor =(1, 0.5))
plt.legend((p1[0], p2[0], p3[0], p4[0], p5[0]),
           ('successful iteration: neural network', 'successful iteration: collision check', 'successful iteration: vertex selection', 'successful iteration: other', 'unseccessful iteration'),
           loc="center left", bbox_to_anchor =(1, 0.5))

plt.savefig("timing_analysis_new.png")


