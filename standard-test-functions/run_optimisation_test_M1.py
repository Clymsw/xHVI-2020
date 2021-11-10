# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 21:00:46 2020

@author: Clym Stock-Williams

Runs SO Bayesian Optimisation on standard deterministic test problems,
applying various acquisition function exploration/exploitation trade-offs

N.B. Line 91 of GPyOpt/models/gpmodel.py needs to have the argument 'robust=True' added
"""
import time
import pyDOE as doe
import numpy as np
import cocoex
import os
import multiprocessing as mp
import matplotlib.pyplot as plt
import GPyOpt
import gc

#def fit_model(d2X, d1Y):
#    model = GPyOpt.models.GPModel(exact_feval=True, ARD=True)
#    model.updateModel(d2X, d1Y, [], [])
#    return model

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

# Define input domain in GPyOpt format and fitness evaluation function
f_id = 71 + 20

suite = cocoex.Suite("bbob", "instances: 1", "")
#for p in suite:
#    print(f"{p.name}")
fitnessfunc = suite.get_problem(f_id)
FUNCTION_NAME = fitnessfunc.name
print(f"Optimising {FUNCTION_NAME}")
NUM_INPUT_DIMS = fitnessfunc.dimension
NUM_TOTAL_EVALUATIONS = 30 * NUM_INPUT_DIMS

tests = []
for xi in [-1.0, -0.5, 0.0, 0.5, 1.0]:
    for zeta in [-0.5, -0.1, -0.01, 0.0, 0.01, 0.1]:
        for it in range(MIN_ITERATION, MAX_ITERATION + 1):
            tests.append((it, xi, zeta))

domain = []
for i in range(NUM_INPUT_DIMS):
    domain.append({'name': "x_{}]".format(i+1), 'type': 'continuous', 'domain': (fitnessfunc.lower_bounds[i], fitnessfunc.upper_bounds[i])})

def evaluate_fitness(ind):
    assert len(ind) == NUM_INPUT_DIMS
    return fitnessfunc(ind)

bounds = []
for q in range(len(domain)):
    bounds.append(domain[q]['domain'])
bounds = np.array(bounds)

OUTPUT_FOLDER = os.path.join("Results_Detailed_M1", f"{FUNCTION_NAME}")

if RUN_OPTIMISATION:
    try:
        os.mkdir(OUTPUT_FOLDER)
    except FileExistsError:
        pass

# %% Run/get optimisation
NUM_SAMPLES = NUM_INPUT_DIMS * 10

def run(ITERATION, XI, JITTER):    
    np.random.seed(1234 + ITERATION)
    
    def make_file_name(description: str, ending: str):
        return f"{FUNCTION_NAME}_{NUM_INPUT_DIMS}D_{XI:0.2f}X_{JITTER:0.2f}Z_{description}_{ITERATION}.{ending}"

    if not RUN_OPTIMISATION:
        # %% Load in previous results
        all_data = np.genfromtxt(os.path.join(OUTPUT_FOLDER, make_file_name("results", "csv")),
                                 delimiter=',', skip_header=1)
        d1SolutionOutput = all_data[:, 0]
        d2SolutionInput = all_data[:, 1:]
        # TODO: Load in timings
        times = np.ones((NUM_TOTAL_EVALUATIONS - NUM_SAMPLES + 1)) * -1
        
    else:
        d2SolutionInput = np.ones((NUM_TOTAL_EVALUATIONS, NUM_INPUT_DIMS)) * -1
        d1SolutionOutput = np.ones(NUM_TOTAL_EVALUATIONS) * -1
        times = np.ones((NUM_TOTAL_EVALUATIONS - NUM_SAMPLES + 1)) * -1
        start_time = time.perf_counter()
        
        # %% Generate initial experimental design
        init_des = doe.lhs(NUM_INPUT_DIMS, samples=NUM_SAMPLES, criterion='center')
        for d in range(len(bounds)):
            d2SolutionInput[:NUM_SAMPLES,d] = init_des[:,d]*(bounds[d][1] - bounds[d][0]) + bounds[d][0]
        # Evaluate our design
        d1SolutionOutput[:NUM_SAMPLES] = np.array(list( map(evaluate_fitness, d2SolutionInput[:NUM_SAMPLES,:]) ))
        
        # %% Run optimisation loop
        i = NUM_SAMPLES
        k = 0
        des_space = GPyOpt.core.task.space.Design_space(domain)
        acq_opt = GPyOpt.optimization.AcquisitionOptimizer(des_space, optimizer='lbfgs')
        while i < NUM_TOTAL_EVALUATIONS:
            incr = time.perf_counter() - start_time
            times[k] = incr
            print(f"{i} evaluations done, current best: {np.min(d1SolutionOutput[:i]):0.3f}")
            
            # Fit a GP model
            f_norm = normalise_f(np.array([(d1SolutionOutput[:i])]).transpose(), XI)
            model = GPyOpt.models.GPModel(exact_feval=True, ARD=True, verbose=False)
            model.updateModel(d2SolutionInput[:i,:], f_norm.reshape((-1,1)), [], [])
            
            # Run acquisition function
            acq = GPyOpt.acquisitions.AcquisitionEI(model, des_space, jitter = JITTER, optimizer = acq_opt)
            next_point, y_next = acq.optimize()
            
            # Evaluate fitness and archive
            f_new = evaluate_fitness(next_point[0])
            print(f"This evaluation: {f_new:0.3f}")
            d1SolutionOutput[i] = f_new
            d2SolutionInput[i,:] = next_point
            i += 1
            k += 1
            #if (np.abs(d1SolutionOutput[i-2] - d1SolutionOutput[i-1]) / d1SolutionOutput[i-1]) < 1e-9:
            #    break
            print("")
        
        times[k] = time.perf_counter() - start_time

        #times = times[:k+1]
        #d1SolutionOutput = d1SolutionOutput[:i]
        #d2SolutionInput = d2SolutionInput[:i,:]

        # %% Save out results
        headerline = "Objective,"
        for d in range(NUM_INPUT_DIMS):
            headerline += "Input {},".format(d+1)
        headerline = headerline[:-1]
        
        with open(os.path.join(OUTPUT_FOLDER, make_file_name("results", "csv")), "w") as fid:
            fid.write(headerline + "\n")
            for l in range(len(d1SolutionOutput)):
                line = "{:.4f},".format(d1SolutionOutput[l])
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
    plt.plot(np.arange(NUM_SAMPLES, len(d1SolutionOutput)), d1SolutionOutput[NUM_SAMPLES:], 'bo')
    plt.plot(np.arange(0, len(d1SolutionOutput)), np.minimum.accumulate(d1SolutionOutput), 'k--')
    plt.plot(np.arange(0, len(d1SolutionOutput)), np.maximum.accumulate(d1SolutionOutput), 'k--')
    #plt.plot([0., len(d1SolutionOutput)], [f_opt, f_opt], 'g--')
    plt.xlabel('Number of evaluations'), plt.ylabel('Solution')
    if ((np.max(d1SolutionOutput) - np.min(d1SolutionOutput)) > 500):
        if np.min(d1SolutionOutput) <= 0:
            plt.yscale("symlog")
        else:
            plt.yscale("log")
    plt.grid()
    
    plt.savefig(os.path.join(OUTPUT_FOLDER, make_file_name("OptProgress", "png")), dpi=400)
    
    # %% Garbage collection
    del d1SolutionOutput, d2SolutionInput, times
    gc.collect()

# %% RUN!
if __name__ == '__main__':
    print("{} processors available".format(mp.cpu_count()))
    pool = mp.Pool(mp.cpu_count() - 2)
    print("Pool opened")
    pool.starmap(run, tests)
    pool.close()

#for test in tests:
#        run(test[0], test[1], test[2])
