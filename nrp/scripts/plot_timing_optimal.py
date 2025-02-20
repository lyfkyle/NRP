import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math
import torch

# time plot
N = 4
colision_time = np.array([0.000926, 0.0012008027904323343, 0.0046142079987176385, 0.004025924261263457])
neural_time = np.array([0, 0, 0.002094152528761667, 0.0014677102749400105])
vertex_selection_time = np.array([0.001988768799237879, 0.0025419718023270556, 0.0029491010995869563, 0.0029881566159833716])
# other_time = np.array([0.00, 0, 0.00012193419778798887, 0.00012193419778798887])
# neural_time_N3 = (0, 0.008)
# neural_time_linkpos_N1 = (0, 0.008)
# neural_time_linkpos_N3 = (0, 0.008)
ind = np.arange(N)

# fig = plt.subplots()
p1 = plt.bar(ind, colision_time)
p2 = plt.bar(ind, neural_time, bottom = colision_time)
p3 = plt.bar(ind, vertex_selection_time, bottom = neural_time + colision_time)
# p4 = plt.bar(ind, neural_time_N1, bottom = colision_time)

plt.title('Time analysis per extension')
plt.ylabel('Time spent (s)')
plt.xticks(ind, ('IRRT*', 'BIT*', 'IRRT*-NE-d', "IRRT*-NE-g"))
plt.legend((p1[0], p2[0], p3[0]), ('collision check', 'neural network', 'vertex selection'))

plt.savefig("timing_analysis.png")

