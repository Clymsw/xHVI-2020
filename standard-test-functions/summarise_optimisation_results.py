# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 13:17:40 2020

@author: Clym Stock-Williams

Summarises the results of MO Bayesian Optimisation run multiple times
on standard deterministic test problems.
"""
import numpy as np
import os
import ParetoFrontND as pf
import StandardTestFunctions as fn

# %% Setup
FUNCTION_NAME = "DTLZ3"

NUM_INPUT_DIMS = 10
NUM_OBJECTIVES = 2

NUM_TOTAL_EVALUATIONS = 300
NUM_SAMPLES = NUM_INPUT_DIMS * 4

XHVI_USED = True 

FOLDER = ""
if XHVI_USED:
    ZETA = 0.0
    # FOLDER = os.path.join("Results_Detailed_Timed", FUNCTION_NAME + "_D{1}_M{2}_Z{0:.2f}_Rnorm".format(ZETA, NUM_INPUT_DIMS, NUM_OBJECTIVES))
    FOLDER = os.path.join("Results_Detailed_HypI_Timed", FUNCTION_NAME + "_D{1}_M{2}_Z{0:.2f}_Rnorm".format(ZETA, NUM_INPUT_DIMS, NUM_OBJECTIVES))
else:
    FOLDER = os.path.join("Results_Detailed_EHVI_Timed", FUNCTION_NAME + "_D{0}_norm_M{1}".format(NUM_INPUT_DIMS, NUM_OBJECTIVES))

# %% Get function properties
d2F1F2_PF = fn.get_M2_pareto_front(FUNCTION_NAME)
d1Reference = [max(d2F1F2_PF[:,0]) * 1.1, max(d2F1F2_PF[:,1]) * 1.1]
max_hypervolume = pf.calculateHypervolume(d2F1F2_PF, d1Reference)

domain, fitnessfunc, _, NUM_INPUT_DIMS, NUM_OBJECTIVES = fn.get_function_definition(
    FUNCTION_NAME, NUM_INPUT_DIMS, NUM_OBJECTIVES)

# %% Load files
filename_out_hv = 'summary_hv_ref_{0:.2f}_{1:.2f}.csv'.format(d1Reference[0], d1Reference[1])
filename_out_igd = 'summary_igd.csv'
# filename_out_timing = 'summary_timing.csv'

files = os.scandir(FOLDER)
for file in files:
    if not file.name.endswith('csv'):
        continue
    if not 'results' in file.name:
        continue
    all_data = np.genfromtxt(file.path,
                             delimiter=',', skip_header=1)
    
    all_data = all_data[:NUM_TOTAL_EVALUATIONS,:]
    
    print("Analysing optimisation {}".format(file.name.split('_')[-1].split('.')[0]))
    
    d2FinalNonDominatedSet, _ = pf.getNonDominatedFront(all_data[:, 1:NUM_OBJECTIVES + 1])
    d2FinalNonDominatedSet = np.unique(d2FinalNonDominatedSet, axis=0)
    for m in range(d2FinalNonDominatedSet.shape[1]):
        with open(os.path.join(FOLDER,'summary_finalset_f{}.csv'.format(m+1)), "a+") as fid:
            line = np.array2string(d2FinalNonDominatedSet[:,m], 
                                   separator=',', 
                                   formatter={'float_kind':lambda x: "%.4f" % x}, 
                                   suppress_small=True,
                                   max_line_width=1e9)
            fid.write(line[1:-1] + "\n")
    
    d1HyperVolume = np.zeros((all_data.shape[0], 1))
    d1IGD = np.zeros((all_data.shape[0], 1))
    print("Analysing evaluation", end =" ")
    for i in range(all_data.shape[0]):
        print("{}".format(i+1), end =", ")
        d2Nds, _ = pf.getNonDominatedFront(all_data[:i+1, 1:NUM_OBJECTIVES + 1])
        d2Nds = np.unique(d2Nds, axis=0)
        d1HyperVolume[i] = pf.calculateHypervolume(d2Nds, d1Reference)
        d1IGD[i] = pf.calculateIGD(d2F1F2_PF, d2Nds)
    
    with open(os.path.join(FOLDER, filename_out_hv), "a+") as fid:
        line = np.array2string((max_hypervolume - d1HyperVolume[:,0]).T,
                                separator=',',
                                formatter={'float_kind':lambda x: "%.6f" % x},
                                suppress_small=True,
                                max_line_width=1e9)
        fid.write(line[1:-1] + "\n")
    with open(os.path.join(FOLDER, filename_out_igd), "a+") as fid:
        line = np.array2string(d1IGD[:,0].T,
                                separator=',',
                                formatter={'float_kind':lambda x: "%.6f" % x},
                                suppress_small=True,
                                max_line_width=1e9)
        fid.write(line[1:-1] + "\n")
        
    print("")
