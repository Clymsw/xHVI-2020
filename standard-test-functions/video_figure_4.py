# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 13:09:15 2020

@author: Clym Stock-Williams

Creates a video of optimisation on a simple N=2, M=2 test problem.
"""
import pyDOE as doe
import numpy as np
import os
from itertools import repeat
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import plotFunctions as figs
import ParetoFrontND as pf
import StandardTestFunctions as fn
import GPyOpt

def fit_model(d2X, d1Y):
    model = GPyOpt.models.GPModel(exact_feval=True, ARD=True)
    model.updateModel(d2X, d1Y, [], [])
    return model

def expected_improvement(mu, sigma, y_star):
    s = (y_star - mu) / sigma
    return sigma * (s * stats.norm.cdf(s) + stats.norm.pdf(s))

def normalise_f(f, exploration_param, ref_data=None):
    if ref_data is None:
        avg = np.mean(f)
        low = np.min(f)
        high = np.max(f)
    else:
        avg = np.mean(ref_data)
        low = np.min(ref_data)
        high = np.max(ref_data)
    offset = (1 - exploration_param) * avg + exploration_param * low
    return (f - offset) / (high - low + 1e-6)

np.random.seed(1234)

# %% Setup problem
FUNCTION_NAME = "DTLZ2"

INFILL = "HypI"
# INFILL = "xHVI"

NUM_INPUT_DIMS = 2
NUM_OBJECTIVES = 2

NUM_TOTAL_EVALUATIONS = 30

ZETA = 0.0

# d1Reference = np.repeat(1000.0, NUM_OBJECTIVES).tolist()
d1Reference = [1.1, 1000.0]

# Define input domain in GPyOpt format and fitness evaluation function
domain, fitnessfunc, d1x_opt, NUM_INPUT_DIMS, NUM_OBJECTIVES = fn.get_function_definition(
    FUNCTION_NAME, NUM_INPUT_DIMS, NUM_OBJECTIVES)

def evaluate_fitness(ind):
    assert len(ind) == NUM_INPUT_DIMS
    return fitnessfunc(ind)

d1F1F2 = np.array(list( map(evaluate_fitness, d1x_opt) ))
d1F1F2_PF, _ = pf.getNonDominatedFront(d1F1F2)
d1MaxHyperVolume = pf.calculateHypervolume(d1F1F2_PF, d1Reference)

# %% Generate initial experimental design
NUM_SAMPLES = NUM_INPUT_DIMS * 4
d2SolutionInput = doe.lhs(NUM_INPUT_DIMS, samples=NUM_SAMPLES, criterion='center')
d2SolutionOutput = np.array(list( map(evaluate_fitness, d2SolutionInput) ))

# Generate map across input space
d1Test = np.linspace(0.0, 1.0, 50)
d2X1, d2X2 = np.meshgrid(d1Test, d1Test)
d2TestPoints = np.hstack((d2X1.reshape((-1,1)), d2X2.reshape((-1,1))))
d2TestResults = np.array(list( map(evaluate_fitness, d2TestPoints)))
d2Sol1 = d2TestResults[:,0].reshape(d2X1.shape)
d2Sol2 = d2TestResults[:,1].reshape(d2X1.shape)

# %% xHVI example
des_space = GPyOpt.core.task.space.Design_space(domain)
acq_opt = GPyOpt.optimization.AcquisitionOptimizer(des_space, optimizer='lbfgs')
reference = np.repeat(1.0, NUM_OBJECTIVES).tolist()

i = NUM_SAMPLES
d1Hypervolume = []
while i < NUM_TOTAL_EVALUATIONS:
    # Plot archive
    fig = plt.figure(figsize=[figs.plot_size * 1.62, figs.plot_size * 2.0], tight_layout=True)
    gs = gridspec.GridSpec(3, 2)
    
    ax = fig.add_subplot(gs[0, 0])
    figs.plot_map_with_points(d2X1, d2X2, d2Sol1, [0.25,0.5,0.75], 'RdYlGn_r', [0., 1.],
                              d2SolutionInput[:,0], d2SolutionInput[:,1], 
                              sYLabel='$x_2$',sTitle='$f_1$')
    plt.xlim((0.,1.))
    plt.ylim((0.,1.))
    
    ax = fig.add_subplot(gs[0, 1])
    figs.plot_map_with_points(d2X1, d2X2, d2Sol2, [1,3,5,7,9], 'RdYlGn_r', [0., 10.],
                              d2SolutionInput[:,0], d2SolutionInput[:,1],
                              sTitle='$f_2$')
    plt.xlim((0.,1.))
    plt.ylim((0.,1.))
    
    # plt.savefig(os.path.join("video","xHVI_{0}_{1}_archive.png".format(FUNCTION_NAME,i)), facecolor=None, edgecolor=None)
    
    # Normalise everything
    y_norm = normalise_f(np.array([(d2SolutionOutput[:, 0])]).transpose(), 0.0)
    for m in range(1, NUM_OBJECTIVES):
        y_norm = np.hstack((y_norm, normalise_f(np.array([(d2SolutionOutput[:, m])]).transpose(), 0.0)))
    
    test_f_norm = normalise_f(np.array([(d2TestResults[:, 0])]).transpose(), 0.0,
                              np.array([(d2SolutionOutput[:, 0])]).transpose())
    for m in range(1, NUM_OBJECTIVES):
        test_f_norm = np.hstack((test_f_norm, normalise_f(np.array([(d2TestResults[:, m])]).transpose(), 0.0,
                                                          np.array([(d2SolutionOutput[:, m])]).transpose())))
    
    d1TrueHVI = pf.calculatePotentialHvcsGivenFront(test_f_norm, y_norm, reference)
    d2TrueHVI = d1TrueHVI.reshape(d2X1.shape)
    
    # Plot HV progress
    d1Hypervolume.append(pf.calculateHypervolume(d2SolutionOutput, d1Reference))
    
    # plt.subplot(ax[1,0])
    # plt.plot(np.arange(NUM_SAMPLES,i+1), (1. - np.array(d1Hypervolume)/d1MaxHyperVolume)*100.)
    
    # ax = fig.add_subplot(gs[1, :])
    # figs.plot_sample_with_points(test_f_norm[:,0], test_f_norm[:,1], d1TrueHVI[:,0], 'RdYlGn', [0., 0.2], 'True potential HVI',
    #                              y_norm[:,0], y_norm[:,1], 'k', 'Current archive',
    #                              sXLabel='$f_{1,norm}$', sYLabel='$f_{2,norm}$', sTitle='Potential HVI')
    
    # plt.savefig(os.path.join("video","xHVI_{0}_{1}_f1f2.png".format(FUNCTION_NAME,i)), facecolor=None, edgecolor=None)

    # %% Calculate infill criterion
    d1Infill=[]
    if INFILL == "HypI":
        d1Infill = pf.calculateHypIs(y_norm, reference)
    elif INFILL == "xHVI":
        d1HVI = pf.calculateHypervolumeContributions(y_norm, reference)
        d1HVIn = pf.calculateNegativeHypervolumeContributions(y_norm)
        d1Infill = (d1HVI - d1HVIn)
    d1Infill_norm = normalise_f(d1Infill, ZETA)    

    # %% Fit a GP model to infill
    model = GPyOpt.models.GPModel(exact_feval=True, ARD=True)
    model.updateModel(d2SolutionInput[:,:], -d1Infill_norm, [], [])
    
    mu, stdev = model.predict(d2TestPoints)
    ei_xhvi = np.array(list( map(expected_improvement, mu, stdev, repeat(min(-d1Infill_norm)))))
    d2xhvc_pred = mu.reshape(d2X1.shape)
    d2xhvc_std_pred = stdev.reshape(d2X1.shape)
    d2xhvi_ei = ei_xhvi.reshape(d2X1.shape)
    
    ax = fig.add_subplot(gs[1, 0])
    figs.plot_map_with_points(d2X1, d2X2, d2xhvc_pred, np.linspace(-0.5, 0.5, 5), 'RdYlGn_r', [-0.5, 0.5],
                              d2SolutionInput[:,0], d2SolutionInput[:,1], d1Infill_norm[:,0],'RdYlGn', 
                              sYLabel='$x_2$', sTitle='Mean function ($-${}$_{{norm}}$)'.format(INFILL))
    plt.xlim((0.,1.))
    plt.ylim((0.,1.))
    
    ax = fig.add_subplot(gs[1, 1])
    figs.plot_map_with_points(d2X1, d2X2, d2xhvc_std_pred, np.linspace(0.05, 0.2, 4), 'RdYlGn_r', [0., 0.2],
                              d2SolutionInput[:,0], d2SolutionInput[:,1], 'green',
                              sTitle="Standard deviation ($-${}$_{{norm}}$)".format(INFILL))
    plt.xlim((0.,1.))
    plt.ylim((0.,1.))
    
    # plt.savefig(os.path.join("video","xHVI_{0}_{1}_gp.png".format(FUNCTION_NAME,i)), facecolor=None, edgecolor=None)
    
    # Run acquisition function
    acq = GPyOpt.acquisitions.AcquisitionEI(model, des_space, jitter = 0.0, optimizer = acq_opt)
    next_point, y_next_est = acq.optimize()
    
    ax = fig.add_subplot(gs[2, 0])
    figs.plot_map_with_points(d2X1, d2X2, d2TrueHVI/np.max(d2TrueHVI)*100., np.linspace(10, 90, 5), 'RdYlGn', [0., 100.],
                              d2SolutionInput[:,0], d2SolutionInput[:,1], d1Infill[:,0]/np.max(d1Infill[:,0])*100., 'RdYlGn', 
                              sXLabel='$x_1$', sYLabel='$x_2$', sTitle='Potential HVI (%)')
    plt.xlim((0.,1.))
    plt.ylim((0.,1.))
    
    ax = fig.add_subplot(gs[2, 1])
    figs.plot_map_with_points(d2X1, d2X2, d2xhvi_ei/np.max(d2xhvi_ei)*100., np.linspace(10., 90., 5), 'RdYlGn', [0., 100.],
                              d2SolutionInput[:,0], d2SolutionInput[:,1], d1Infill[:,0]/np.max(d1Infill[:,0])*100., 'RdYlGn', 
                              next_point[0][0], next_point[0][1],
                              sXLabel='$x_1$', sTitle='Expected Improvement in {} (%)'.format(INFILL))
    plt.xlim((0.,1.))
    plt.ylim((0.,1.))
    
    plt.savefig(os.path.join("video","{2}_{0}_{1}.png".format(FUNCTION_NAME,i,INFILL)), facecolor=None, edgecolor=None)
    
    # %% Archive and increment
    d2SolutionInput = np.vstack((d2SolutionInput, next_point[0]))
    d2SolutionOutput = np.vstack((d2SolutionOutput, evaluate_fitness(next_point[0])))
    i += 1
    
    