# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 21:00:46 2020

@author: Clym Stock-Williams

Runs MO Bayesian Optimisation on standard deterministic test problems,
applying the EHVI acquisition function

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

def fit_model(d2X, d1Y):
    model = GPyOpt.models.GPModel(exact_feval=True, ARD=True)
    model.updateModel(d2X, d1Y, [], [])
    return model

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

FUNCTION_NAME = "ZDT6"
# FUNCTION_NAME = "DTLZ3"

NUM_INPUT_DIMS = 10
NUM_OBJECTIVES = 2

ZETA = 0.0

NUM_TOTAL_EVALUATIONS = 300

# d1Reference = np.repeat(1000.0, NUM_OBJECTIVES).tolist()
d1Reference = [1.1, 100.0]

# Define input domain in GPyOpt format and fitness evaluation function
domain, fitnessfunc, d1x_opt, NUM_INPUT_DIMS, NUM_OBJECTIVES = fn.get_function_definition(
    FUNCTION_NAME, NUM_INPUT_DIMS, NUM_OBJECTIVES)

def evaluate_fitness(ind):
    assert len(ind) == NUM_INPUT_DIMS
    return fitnessfunc(ind)

bounds = []
for q in range(len(domain)):
    bounds.append(domain[q]['domain'])
bounds = np.array(bounds)

d1F1F2 = np.array(list( map(evaluate_fitness, d1x_opt) ))
d1F1F2_PF, _ = pf.getNonDominatedFront(d1F1F2)

OUTPUT_FOLDER = "." + os.sep + "Results_Detailed_EHVI_Timed2" + os.sep + FUNCTION_NAME + "_D{}".format(NUM_INPUT_DIMS) + "_M{}".format(NUM_OBJECTIVES) 
if RUN_OPTIMISATION:
    try:
        os.mkdir(OUTPUT_FOLDER)
    except FileExistsError:
        pass

# %% Create and plot true HV
if NUM_INPUT_DIMS == 2:
    plt.figure()
    d1Test = np.linspace(0.0, 1.0, 50)
    d2X, d2Y = np.meshgrid(d1Test, d1Test)
    d2TestPoints = np.hstack((d2X.reshape((-1,1)), d2Y.reshape((-1,1))))
    
    d2SolutionGrid = np.array(list( map(evaluate_fitness, d2TestPoints)))
    plt.plot(d2SolutionGrid[:,0], d2SolutionGrid[:,1], 
            linestyle='', marker = '.', markersize=2, 
            label = 'Grid in input domain')
    d2FrontGrid, _ = pf.getNonDominatedFront(d2SolutionGrid)
    plt.plot(d2FrontGrid[:,0], d2FrontGrid[:,1], 
            linestyle = '', marker = '.', color = 'red', markersize = 4, 
            label = 'Pareto Front')
    hvol = pf.calculateHypervolume(d2FrontGrid, d1Reference)
    plt.xlabel('$f_1$'), plt.ylabel('$f_2$')
    plt.grid()
    plt.legend()
    plt.title("Total hypervolume: {0:4g}".format(hvol))
    
    plt.savefig(
        os.path.join(OUTPUT_FOLDER, 
                     FUNCTION_NAME + "_{}D".format(NUM_INPUT_DIMS) + "_{}M".format(NUM_OBJECTIVES) + "_TruePF.png"), 
        dpi=400)

# %% Run/get optimisation
NUM_SAMPLES = NUM_INPUT_DIMS * 4

def run(ITERATION):    
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
        # TODO: Load in timings
        times = np.ones((NUM_TOTAL_EVALUATIONS - NUM_SAMPLES + 1)) * -1
        
    else:
        d1HyperVolume = np.ones((NUM_TOTAL_EVALUATIONS - NUM_SAMPLES + 1)) * -1
        d2SolutionInput = np.ones((NUM_TOTAL_EVALUATIONS, NUM_INPUT_DIMS)) * -1
        d2SolutionOutput = np.ones((NUM_TOTAL_EVALUATIONS, NUM_OBJECTIVES)) * -1
        times = np.ones((NUM_TOTAL_EVALUATIONS - NUM_SAMPLES + 1)) * -1
        start_time = time.perf_counter()
        
        # %% Generate initial experimental design
        d2SolutionInput[:NUM_SAMPLES,:] = doe.lhs(NUM_INPUT_DIMS, samples=NUM_SAMPLES, criterion='center')
        # Evaluate our design
        d2SolutionOutput[:NUM_SAMPLES,:] = np.array(list( map(evaluate_fitness, d2SolutionInput[:NUM_SAMPLES,:]) ))
        
        # %% Run optimisation loop
        i = NUM_SAMPLES
        k = 0
        
        optimizer = GPyOpt.optimization.optimizer.OptLbfgs(bounds)
        while i < NUM_TOTAL_EVALUATIONS:
            incr = time.perf_counter() - start_time
            times[k] = incr
            # Calculate current hypervolume
            d1HyperVolume[k] = pf.calculateHypervolume(d2SolutionOutput[:i,:], d1Reference)
            print("{0} evaluations done, current HVI: {1:0.3f}".format(i, d1HyperVolume[k]))
            
            # Fit GPs
            f_norm = normalise_f(np.array([(d2SolutionOutput[:i, 0])]).transpose(), ZETA)
            for m in range(1, NUM_OBJECTIVES):
                f_norm = np.hstack((f_norm, normalise_f(np.array([(d2SolutionOutput[:i, m])]).transpose(), ZETA)))
            models = []
            for m in range(NUM_OBJECTIVES):
                models.append(fit_model(d2SolutionInput[:i,:], f_norm[:,m].reshape((-1,1))))
            
            d2CurrentFrontNorm, _ = pf.getNonDominatedFront(f_norm)
            
            def ehvi_evaluate(d1X):
                mu = []
                s = []
                for m in range(len(models)):
                    mu_new, s_new = models[m].predict(d1X)
                    mu.append(mu_new[0])
                    s.append(s_new[0])
                ehvi = pf.calculateExpectedHypervolumeContributionMC(
                    np.array(mu),
                    np.array(s),
                    d2CurrentFrontNorm, 
                    np.repeat(1.0, NUM_OBJECTIVES).tolist(),
                    1000)
                return -ehvi # Maximise
        
            # Run EHVI acquisition
            ehvi_max = 0.0
            x_next = d2SolutionInput[0]
            for n in range(10):
                # Multi-restart
                x_test = pf.getExcitingNewLocation(d2SolutionOutput[:i,:], d2SolutionInput[:i,:], bounds[:,0], bounds[:,1], jitter=0.2)
                print("EHVI optimisation, iteration {0}/10]".format(n+1))
                x_opt, f_opt = optimizer.optimize(np.array(x_test), f=ehvi_evaluate)
                #print("Reached [{0:0.3f}, {1:0.3f}], value {2:0.4f}".format(x_opt[0][0], x_opt[0][1], f_opt[0][0]))
                if f_opt[0][0] < ehvi_max:
                    ehvi_max = f_opt[0][0]
                    x_next = x_opt[0]
                    #print("New best.")
            
            # Evaluate fitness and archive
            d2SolutionOutput[i,:] = evaluate_fitness(x_next)
            d2SolutionInput[i,:] = x_next
            i += 1
            k += 1
            print("")
        
        d1HyperVolume[k] = pf.calculateHypervolume(d2SolutionOutput, d1Reference)
        times[k] = time.perf_counter() - start_time
        
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
