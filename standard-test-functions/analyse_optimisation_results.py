# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 13:17:40 2020

@author: Clym Stock-Williams

Analyses the results of MO Bayesian Optimisation run multiple times
on standard deterministic test problems.
"""
import numpy as np
import os
import matplotlib.pyplot as plt
import ParetoFrontND as pf
import StandardTestFunctions as fn

# %% Setup
FUNCTION_NAME = "ZDT3"

NUM_INPUT_DIMS = 10
NUM_OBJECTIVES = 2

XHVI_USED = True

FOLDER = ""
if XHVI_USED:
    ZETA = 0.0
    # REFERENCE = 1.5
    REFERENCE_START = 1.8
    REFERENCE_END = 1.2
    # FOLDER = "Results_Detailed\\" + FUNCTION_NAME + "_D{2}_M{3}_Z{0:.2f}_R{1:.2f}".format(ZETA, REFERENCE, NUM_INPUT_DIMS, NUM_OBJECTIVES)
    FOLDER = "Results_Detailed\\" + FUNCTION_NAME + "_D{2}_M{3}_Z{0:.2f}_Rstart{1:.2f}_Rend{4:.2f}".format(ZETA, REFERENCE_START, NUM_INPUT_DIMS, NUM_OBJECTIVES, REFERENCE_END)
else:
    FOLDER = "Results_Detailed_EHVI\\" + FUNCTION_NAME + "_D{0}_M{1}".format(NUM_INPUT_DIMS, NUM_OBJECTIVES)

# d1Reference = np.repeat(1000.0, NUM_OBJECTIVES).tolist()
lower_lim_plot, upper_lim_plot = 1e-2, 1e2
d1Reference = [1.0, 1000.0]

# %% Get function properties
domain, fitnessfunc, d1x_opt, NUM_INPUT_DIMS, NUM_OBJECTIVES = fn.get_function_definition(
    FUNCTION_NAME, NUM_INPUT_DIMS, NUM_OBJECTIVES)

def evaluate_fitness(ind):
    assert len(ind) == NUM_INPUT_DIMS
    return fitnessfunc(ind)

d1F1F2 = np.array(list( map(evaluate_fitness, d1x_opt) ))
d1F1F2_PF, _ = pf.getNonDominatedFront(d1F1F2)

max_hypervolume = pf.calculateHypervolume(d1F1F2_PF, d1Reference)

# %% Load files
files = os.scandir(FOLDER)
d2HyperVolumeProgress = np.zeros((0,1))
d1NonDominatedSets = []
for file in files:
    if not file.name.endswith('csv'):
        continue
    # name_sections = file.name.split('_')
    # iteration = int(name_sections[-1].split('.')[0])
    all_data = np.genfromtxt(file.path,
                             delimiter=',', skip_header=1)
    d2NonDominatedSet, _ = pf.getNonDominatedFront(all_data[:, 1:NUM_OBJECTIVES + 1])
    d1NonDominatedSets.append(d2NonDominatedSet)
    # d2SolutionInput = all_data[:, NUM_OBJECTIVES+1:NUM_OBJECTIVES+NUM_INPUT_DIMS+1]
    
    d1HyperVolume = all_data[:, 0]
    for i in range(all_data.shape[0]):
        if all_data[i,0] >= 0.0:
            d2Nds, _ = pf.getNonDominatedFront(all_data[0:i+1, 1:NUM_OBJECTIVES + 1])
            d1HyperVolume[i] = pf.calculateHypervolume(d2Nds, d1Reference)
    
    if len(d2HyperVolumeProgress) > 0:
        d2HyperVolumeProgress = np.hstack((d2HyperVolumeProgress, d1HyperVolume.reshape((-1,1))))
    else:
        d2HyperVolumeProgress = d1HyperVolume.reshape((-1,1))

d2HypervolumeLoss = max_hypervolume - d2HyperVolumeProgress

# %% Optimisation progress
plt.plot(range(d2HypervolumeLoss.shape[0]), 
         np.median(d2HypervolumeLoss, axis=1),
         linewidth=2, color='k', label='Median')
plt.plot(range(d2HypervolumeLoss.shape[0]), 
         np.percentile(d2HypervolumeLoss, 75, axis=1),
         linewidth=1, color='r', label='Upper quartile')
plt.plot(range(d2HypervolumeLoss.shape[0]), 
         np.percentile(d2HypervolumeLoss, 25, axis=1),
         linewidth=1, color='g', label='Lower quartile')
iWorst = np.argmax(d2HypervolumeLoss[-1,:])
plt.plot(range(d2HypervolumeLoss.shape[0]), 
         d2HypervolumeLoss[:,iWorst],
         linewidth=1, color='r', linestyle='--', label='Worst')
# plt.plot(range(d2HypervolumeLoss.shape[0]), 
#          np.max(d2HypervolumeLoss, axis=1),
#          linewidth=1, color='r', linestyle='--', label='Worst')
iBest = np.argmin(d2HypervolumeLoss[-1,:])
# plt.plot(range(d2HypervolumeLoss.shape[0]), 
#          np.min(d2HypervolumeLoss, axis=1),
#          linewidth=1, color='g', linestyle='--', label='Best')
plt.plot(range(d2HypervolumeLoss.shape[0]), 
         d2HypervolumeLoss[:,iBest],
         linewidth=1, color='g', linestyle='--', label='Best')
plt.yscale('log')
plt.ylim(lower_lim_plot, upper_lim_plot)
plt.xlim(0, d2HyperVolumeProgress.shape[0])
plt.grid()
plt.legend(loc='lower left')
plt.ylabel("Hypervolume loss")
plt.xlabel("Number of evaluations")
subtitle = ""
if XHVI_USED:
    # subtitle = "xHVI, Exploration = {0:.2f}, Reference = {1:.2f}".format(ZETA, REFERENCE)
    subtitle = "xHVI, Exploration = {0:.2f}, Reference $i\leq100$ = {1:.2f} i>100 = {2:.2f}".format(ZETA, REFERENCE_START, REFERENCE_END)
else:
    subtitle = "Using EHVI acquisition function"
plt.title("{0} ($D={1}, M={2}$) \n {3}".format(
    FUNCTION_NAME, NUM_INPUT_DIMS, NUM_OBJECTIVES, subtitle))

plt.savefig(os.path.join(FOLDER, "SummaryOptimisationProgress.png"), dpi=400)

# %% Analysis
print("--- Function: {} ---".format(FUNCTION_NAME))
if XHVI_USED:
    # print("Acquisition: xHVI, zeta = {0:.2f}, ref = {1:.2f}".format(ZETA, REFERENCE))
    print("Acquisition: xHVI, zeta = {0:.2f}, ref[i<=100] = {1:.2f} ref[i>100] = {2:.2f}".format(ZETA, REFERENCE_START, REFERENCE_END))
else:
    print("Acquistion: EHVI")

print("--- Hypervolume Losses ---")
print("Worst = {:.3f}".format(
    np.max(d2HypervolumeLoss[-1,:])))
print("Upper Quartile = {:.3f}".format(
    np.percentile(d2HypervolumeLoss[-1,:], 75)))
print("Median = {:.3f}".format(
    np.median(d2HypervolumeLoss[-1,:])))
print("Lower Quartile = {:.3f}".format(
    np.percentile(d2HypervolumeLoss[-1,:], 25)))
print("Best = {:.3f}".format(
    np.min(d2HypervolumeLoss[-1,:])))

print("--- Final Non-Dominated Set Size ---")
set_size = []
for front in d1NonDominatedSets:
    set_size.append(front.shape[0])
print("Smallest = {}".format(np.min(set_size)))
print("Lower quartile = {}".format(np.percentile(set_size, 25)))
print("Median = {}".format(np.median(set_size)))
print("Upper quartile = {}".format(np.percentile(set_size, 75)))
print("Largest = {}".format(np.max(set_size)))

# %% 
plt.figure()
plt.scatter(set_size, d2HypervolumeLoss[-1,:])
plt.ylabel("Hypervolume loss")
plt.xlabel("Final non-dominated set size")
plt.title("{0} ($D={1}, M={2}$) \n {3}".format(
    FUNCTION_NAME, NUM_INPUT_DIMS, NUM_OBJECTIVES, subtitle))
plt.yscale('log')
plt.grid()
plt.xlim(0,30)
plt.xticks(range(0,31,2))
plt.savefig(os.path.join(FOLDER, "LossVsSetSize.png"), dpi=400)
