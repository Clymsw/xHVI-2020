# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 13:17:40 2020

@author: Clym Stock-Williams

Summarises the results of MO Bayesian Optimisation run multiple times
on standard deterministic test problems.
"""
import numpy as np
import os

# %% Setup
FUNCTION_NAME = "BBOB suite problem f20 instance 1 in 10D"
NUM_INPUT_DIMS = 10

NUM_TOTAL_EVALUATIONS = 300
NUM_SAMPLES = NUM_INPUT_DIMS * 4

FOLDER = os.path.join("Results_Detailed_M1", f"{FUNCTION_NAME}")

# %% Load files
filename_out = f"{FUNCTION_NAME}_summary.csv"
filename_out_timing = "summary_timing.csv"

files = os.scandir(FOLDER)
for file in files:
    if not file.name.endswith('csv'):
        continue
    if not 'results' in file.name:
        continue
    all_data = np.genfromtxt(file.path,
                             delimiter=',', skip_header=1)
        
    # print("Analysing optimisation {}".format(file.name.split('_')[-1].split('.')[0]))
    
    name_parts = file.name[:-4].split('_')
    
    zeta = float(name_parts[2][:-1])
    jitter = float(name_parts[3][:-1])
    iteration = int(name_parts[5])

    best_found = np.min(all_data[:,0])

    with open(os.path.join(FOLDER,filename_out), "a+") as fid:
        line = f"{zeta:.2f}, {jitter:.2f}, {iteration}, {best_found:.6f}"
        # np.array2string(d2FinalNonDominatedSet[:,m], 
        #                         separator=',', 
        #                         formatter={'float_kind':lambda x: "%.4f" % x}, 
        #                         suppress_small=True,
        #                         max_line_width=1e9)
        fid.write(line + "\n")
        
    print(".", end='')
