# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 21:09:43 2020

@author: Clym Stock-Williams
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns; sns.set(color_codes=True)

plt.style.use('ggplot')
rcParams['font.sans-serif'] = "Segoe UI"
rcParams['font.family'] = "sans-serif"

plot_size = 10.0

# %% Get function properties
FUNCTION_NAME = "ZDT6"
NUM_INPUT_DIMS = 10
NUM_OBJECTIVES = 2

NUM_TOTAL_EVALUATIONS = 300
NUM_SAMPLES = NUM_INPUT_DIMS * 4

# d2F1F2_PF = fn.get_M2_pareto_front(FUNCTION_NAME)
# d1Reference = [max(d2F1F2_PF[:,0]) * 1.1, max(d2F1F2_PF[:,1]) * 1.1]
# max_hypervolume = pf.calculateHypervolume(d2F1F2_PF, d1Reference)

# domain, fitnessfunc, _, NUM_INPUT_DIMS, NUM_OBJECTIVES = fn.get_function_definition(
#     FUNCTION_NAME, NUM_INPUT_DIMS, NUM_OBJECTIVES)

# %% Load data 
def load_data(folder: str):
    evaluation_idxs = []
    d2Times = []
    
    files = os.scandir(folder)
    for file in files:
        if not file.name.endswith('csv'):
            continue
        if not 'timings' in file.name:
            continue
        all_data = np.genfromtxt(file.path,
                                 delimiter=',', skip_header=1)
        evaluation_idxs = all_data[:,0]
        
        d2Times.append(all_data[:,1])
        
    d2Times = np.array(d2Times)
    
    return evaluation_idxs, d2Times

# %% xHVI
ZETA = 0.0

folder_xhvi = os.path.join("Results_Detailed_Timed",
                           FUNCTION_NAME + "_D{1}_M{2}_Z{0:.2f}_Rnorm".format(ZETA, NUM_INPUT_DIMS, NUM_OBJECTIVES))
# ev_idxs_xhvi, d2Timings_xhvi = load_data(folder_xhvi)
all_data = np.genfromtxt(os.path.join(folder_xhvi, 
                                      "{0}_{1}D_{2}M_timings_1.csv".format(FUNCTION_NAME, NUM_INPUT_DIMS, NUM_OBJECTIVES)),
                         delimiter=',', skip_header=1)
ev_idxs_xhvi, d2Timings_xhvi = all_data[:,0], all_data[:,1]

# %% HypI
ZETA = 0.0

folder_hypi = os.path.join("Results_Detailed_HypI_Timed",
                           FUNCTION_NAME + "_D{1}_M{2}_Z{0:.2f}_Rnorm_fortiming".format(ZETA, NUM_INPUT_DIMS, NUM_OBJECTIVES))
# ev_idxs_xhvi, d2Timings_xhvi = load_data(folder_xhvi)
all_data = np.genfromtxt(os.path.join(folder_hypi, 
                                      "{0}_{1}D_{2}M_timings_1.csv".format(FUNCTION_NAME, NUM_INPUT_DIMS, NUM_OBJECTIVES)),
                         delimiter=',', skip_header=1)
ev_idxs_hypi, d2Timings_hypi = all_data[:,0], all_data[:,1]

# %% EHVI
folder_ehvi = os.path.join("Results_Detailed_EHVI_Timed",
                           FUNCTION_NAME + "_D{0}_norm_M{1}".format(NUM_INPUT_DIMS, NUM_OBJECTIVES))
# ev_idxs_ehvi, d2Timings_ehvi = load_data(folder_ehvi)
all_data = np.genfromtxt(os.path.join(folder_ehvi, 
                                      "{0}_{1}D_{2}M_timings_1.csv".format(FUNCTION_NAME, NUM_INPUT_DIMS, NUM_OBJECTIVES)),
                         delimiter=',', skip_header=1)
ev_idxs_ehvi, d2Timings_ehvi = all_data[:,0], all_data[:,1]

# %% Plot graphs
fig = plt.figure(figsize=[plot_size*1.62, plot_size])
# plt.plot(ev_idxs_xhvi,
#          np.percentile(d2Timings_xhvi, 75, axis=0),
#          linewidth=1, linestyle='--', color='g', label='xHVI upper quartile')
# plt.plot(ev_idxs_xhvi,
#          np.median(d2Timings_xhvi, axis=0),
#          linewidth=2, color='g', label='xHVI median')
# plt.plot(ev_idxs_xhvi,
#          np.percentile(d2Timings_xhvi, 25, axis=0),
#          linewidth=1, linestyle='--', color='g', label='xHVI lower quartile')

plt.plot(ev_idxs_xhvi, d2Timings_xhvi/60.,
         linewidth=2, linestyle='-', color='g', label='xHVI')

plt.plot(ev_idxs_hypi, d2Timings_hypi/60.,
         linewidth=2, linestyle='-.', color='r', label='HypI')

# plt.plot(ev_idxs_ehvi,
#          np.percentile(d2Timings_ehvi, 75, axis=0),
#          linewidth=1, linestyle='--', color='b', label='EHVI upper quartile')
# plt.plot(ev_idxs_ehvi,
#          np.median(d2Timings_ehvi, axis=0),
#          linewidth=2, color='b', label='EHVI median')
# plt.plot(ev_idxs_ehvi,
#          np.percentile(d2Timings_ehvi, 25, axis=0),
#          linewidth=1, linestyle='--', color='b', label='EHVI lower quartile')
plt.plot(ev_idxs_ehvi, d2Timings_ehvi/60.,
         linewidth=2, linestyle='--', color='b', label='EHVI')

plt.legend(loc='upper left', fontsize=plot_size*2.5, labelspacing=0.25)
plt.xlabel("Number of evaluations", fontsize=plot_size*3.0)
plt.ylabel('Time elapsed (min)', fontsize=plot_size*3.0)
plt.yticks(np.arange(0,211,30))
for tick in fig.get_axes()[0].get_xticklabels():
    tick.set_fontsize(plot_size*2.5)
for tick in fig.get_axes()[0].get_yticklabels():
    tick.set_fontsize(plot_size*2.5)
plt.xlim([0, 300])
plt.ylim([0, 210])

plt.savefig(os.path.join("img","timings_" + FUNCTION_NAME + "_1.svg"), facecolor=None, edgecolor=None)
