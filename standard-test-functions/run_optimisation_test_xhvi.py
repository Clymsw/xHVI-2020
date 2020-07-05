# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 16:16:15 2020

@author: Clym Stock-Williams

Runs MO Bayesian Optimisation on standard deterministic test problems,
applying the xHVI infill criterion

N.B. Line 91 of GPyOpt/models/gpmodel.py needs to have the argument 'robust=True' added
"""
import time
import pyDOE as doe
import numpy as np
import os
import multiprocessing as mp
import matplotlib.pyplot as plt
import ParetoFrontND as pf
import StandardTestFunctions as fn
import GPyOpt
import gc

def normalise_f(f, exploration_param):
    avg = np.mean(f)
    low = np.min(f)
    high = np.max(f)
    offset = (1 - exploration_param) * avg + exploration_param * low
    return (f - offset) / (high - low + 1e-6)

# %% Setup problem 
RUN_OPTIMISATION = True
MIN_ITERATION = 1
MAX_ITERATION = 21

ZETA = 0.0

NUM_TOTAL_EVALUATIONS = 300

FUNCTION_NAME = "ZDT6"
#FUNCTION_NAME = "DTLZ4"

NUM_INPUT_DIMS = 10
NUM_OBJECTIVES = 2

d1Reference = [1.1, 100.0]
# d1Reference = np.repeat(100.0, NUM_OBJECTIVES).tolist()

# Define input domain in GPyOpt format and fitness evaluation function
domain, fitnessfunc, d1x_opt, NUM_INPUT_DIMS, NUM_OBJECTIVES = fn.get_function_definition(
    FUNCTION_NAME, NUM_INPUT_DIMS, NUM_OBJECTIVES)
    
def evaluate_fitness(ind):
    assert len(ind) == NUM_INPUT_DIMS
    return fitnessfunc(ind)

d1F1F2 = np.array(list( map(evaluate_fitness, d1x_opt) ))
d1F1F2_PF, _ = pf.getNonDominatedFront(d1F1F2)

# OUTPUT_FOLDER = "Results_Detailed\\" + FUNCTION_NAME + "_D{2}_M{3}_Z{0:.2f}_R{1:.2f}".format(ZETA, REFERENCE, NUM_INPUT_DIMS, NUM_OBJECTIVES)
OUTPUT_FOLDER = os.path.join("Results_Detailed_Timed", 
                             FUNCTION_NAME + "_D{1}_M{2}_Z{0:.2f}_Rnorm".format(ZETA, NUM_INPUT_DIMS, NUM_OBJECTIVES))
if RUN_OPTIMISATION:
    try:
        os.mkdir(OUTPUT_FOLDER)
    except FileExistsError:
        pass

# %% Run optimisation
NUM_SAMPLES = NUM_INPUT_DIMS * 4

def run(ITERATION):
    print("Made it into iteration {}".format(ITERATION))
    np.random.seed(1234 + ITERATION)
    
    def make_file_name(description: str, ending: str):
        return FUNCTION_NAME + "_{}D".format(NUM_INPUT_DIMS) + "_{}M".format(NUM_OBJECTIVES) + "_{0}_{1}.{2}".format(description, ITERATION, ending)
    
    if not RUN_OPTIMISATION:
        # %% Load in previous results
        all_data = np.genfromtxt(os.path.join(OUTPUT_FOLDER, make_file_name("results", "csv")),
                                 delimiter=',', skip_header=1)
        d2SolutionOutput = all_data[:, 1:NUM_OBJECTIVES + 1]
        d2SolutionInput = all_data[:, NUM_OBJECTIVES+1:NUM_OBJECTIVES+NUM_INPUT_DIMS+1]
        d1HyperVolume = all_data[~np.isnan(all_data[:, 0]), 0]
    
    else:
        d1HyperVolume = np.empty((NUM_TOTAL_EVALUATIONS - NUM_SAMPLES + 1))
        d2SolutionInput = np.empty((NUM_TOTAL_EVALUATIONS, NUM_INPUT_DIMS))
        d2SolutionOutput = np.empty((NUM_TOTAL_EVALUATIONS, NUM_OBJECTIVES))
        times = np.empty((NUM_TOTAL_EVALUATIONS - NUM_SAMPLES + 1))
        start_time = time.perf_counter()
        
        # %% Generate initial experimental design
        d2SolutionInput[:NUM_SAMPLES,:] = doe.lhs(NUM_INPUT_DIMS, samples=NUM_SAMPLES, criterion='center')
        # Evaluate our design
        d2SolutionOutput[:NUM_SAMPLES,:] = np.array(list( map(evaluate_fitness, d2SolutionInput[:NUM_SAMPLES,:]) ))

        # %% Run optimisation loop
        i = NUM_SAMPLES
        k = 0
        des_space = GPyOpt.core.task.space.Design_space(domain)
        acq_opt = GPyOpt.optimization.AcquisitionOptimizer(des_space, optimizer='lbfgs')
        while i < NUM_TOTAL_EVALUATIONS:
            incr = time.perf_counter() - start_time
            times[k] = incr
            # Calculate current hypervolume
            d1HyperVolume[k] = pf.calculateHypervolume(d2SolutionOutput[:i,:], d1Reference)
            print("{0} evaluations done, current HVI: {1:0.3f}".format(i, d1HyperVolume[k]))
            
            # Normalise solution space
            y_norm = normalise_f(np.array([(d2SolutionOutput[:i, 0])]).transpose(), 0.0)
            for m in range(1, NUM_OBJECTIVES):
                y_norm = np.hstack((y_norm, normalise_f(np.array([(d2SolutionOutput[:i, m])]).transpose(), 0.0)))
            
            # Calculate infill criterion
            d1HVI = pf.calculateHypervolumeContributions(y_norm, np.repeat(1.0, NUM_OBJECTIVES).tolist())
            d1HVIn = pf.calculateNegativeHypervolumeContributions(y_norm)
            d1xHVI = (d1HVI - d1HVIn)
            d1xHVI_norm = normalise_f(d1xHVI, ZETA)
            
            # Fit a GP model to xHVI
            model = GPyOpt.models.GPModel(exact_feval=True, ARD=True)
            model.updateModel(d2SolutionInput[:i,:], -d1xHVI_norm, [], [])
            
            # Run acquisition function
            acq = GPyOpt.acquisitions.AcquisitionEI(model, des_space, jitter = 0.0, optimizer = acq_opt)
            next_point, y_next = acq.optimize()
            
            # Evaluate fitness and archive
            d2SolutionOutput[i,:] = evaluate_fitness(next_point[0])
            d2SolutionInput[i,:] = next_point[0]
            i += 1
            k += 1
        
        d1HyperVolume[-1] = pf.calculateHypervolume(d2SolutionOutput, d1Reference)
        times[-1] = time.perf_counter() - start_time
            
        # %% Save out results
        headerline = "Hypervolume,"
        for m in range(NUM_OBJECTIVES):
            headerline += "Objective {},".format(m+1)
        for d in range(NUM_INPUT_DIMS):
            headerline += "Input {},".format(d+1)
        headerline = headerline[:-1]
        
        with open(os.path.join(OUTPUT_FOLDER, make_file_name("results", "csv")), "w") as fid:
            fid.write(headerline + "\n")
            for l in range(NUM_TOTAL_EVALUATIONS):
                line = ""
                if l < NUM_SAMPLES - 1:     
                    line += ","
                else:
                    line += "{:.3f},".format(d1HyperVolume[l - NUM_SAMPLES + 1])
                for m in range(NUM_OBJECTIVES):
                    line += "{:.4f},".format(d2SolutionOutput[l,m])
                for d in range(NUM_INPUT_DIMS):
                    line += "{:.4f},".format(d2SolutionInput[l,d])
                fid.write(line[:-1] + "\n")
        
        with open(os.path.join(OUTPUT_FOLDER, make_file_name("timings", "csv")), "w") as fid:
            fid.write("Evaluation," + "Time taken" + "\n")
            for l in range(len(times)):
                line = "{},".format(l + NUM_SAMPLES)
                line += "{:.3f}".format(times[l])
                fid.write(line + "\n")
            
    # %% Plot optimisation progress
    plt.figure()
    plt.plot(np.arange(NUM_SAMPLES, NUM_TOTAL_EVALUATIONS + 1), d1HyperVolume)
    plt.xlabel('Number of evaluations'), plt.ylabel('Hypervolume')
    plt.grid()
    
    plt.savefig(os.path.join(OUTPUT_FOLDER, make_file_name("OptProgress", "png")), dpi=400)
    
    # %% Plot optimisation results with true PF
    d2Front, _ = pf.getNonDominatedFront(d2SolutionOutput)
    
    for d in range(1, NUM_OBJECTIVES):
        plt.figure()
        
        plt.plot(d1F1F2_PF[:,0], d1F1F2_PF[:,d], 
                 color = 'green', marker='.', markersize=2,
                 linestyle='', label = 'True Pareto Front')
        
        plt.plot(d2SolutionOutput[:,0], d2SolutionOutput[:,d], 
                 linestyle='', marker = '.', markersize=2, 
                 label = 'All evaluated solutions')
        
        plt.plot(d2Front[:,0], d2Front[:,d], 
                 linestyle = '', marker = '.', color = 'red', markersize = 4, 
                 label = 'Final non-dominated set')
        
        plt.xlabel('$f_1$'), plt.ylabel("$f_{}$".format(d+1))
        plt.grid()
        plt.legend(loc='upper right')
        plt.title("Total hypervolume: {0:4g}".format(d1HyperVolume[-1]))
        
        plt.savefig(os.path.join(OUTPUT_FOLDER, make_file_name("OptPF_F{0}v{1}".format(1,d+1), "png")), dpi=400)
    
    # %% Garbage collection
    del d2SolutionOutput, d2SolutionInput, d1HyperVolume, times
    gc.collect()

# %% RUN!
if __name__ == '__main__':
    print("{} processors available".format(mp.cpu_count()))
    pool = mp.Pool(mp.cpu_count() - 2)
    print("Pool opened")
    pool.map(run, range(MIN_ITERATION, MAX_ITERATION + 1))
    pool.close()

# for it in range(MIN_ITERATION - 1, MAX_ITERATION):
#     run(it+1)