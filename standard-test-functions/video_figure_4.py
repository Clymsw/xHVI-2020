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
FUNCTION_NAME = "ZDT4"
# FUNCTION_NAME = "DTLZ2"

NUM_INPUT_DIMS = 2
NUM_OBJECTIVES = 2

NUM_TOTAL_EVALUATIONS = 50

ZETA = 0.0

# Define input domain in GPyOpt format and fitness evaluation function
domain, fitnessfunc, _, NUM_INPUT_DIMS, NUM_OBJECTIVES = fn.get_function_definition(
    FUNCTION_NAME, NUM_INPUT_DIMS, NUM_OBJECTIVES)

bounds = []
for q in range(len(domain)):
    bounds.append(domain[q]['domain'])
bounds = np.array(bounds)

def evaluate_fitness(ind):
    assert len(ind) == NUM_INPUT_DIMS
    return fitnessfunc(ind)

d2F1F2_PF = fn.get_M2_pareto_front(FUNCTION_NAME)
d2PFRange = np.vstack([np.min(d2F1F2_PF, axis=0), np.max(d2F1F2_PF, axis=0)])
# d1Reference = 1.1 * np.max(d2F1F2_PF, axis=0)
# dMaxHyperVolume = pf.calculateHypervolume(d2F1F2_PF, d1Reference)

d2InputPlotRange = bounds.copy()
bound_range = np.max(bounds, axis=1) - np.min(bounds, axis=1)
d2InputPlotRange[0][0] = d2InputPlotRange[0][0] - bound_range[0]/10
d2InputPlotRange[0][1] = d2InputPlotRange[0][1] + bound_range[0]/10
d2InputPlotRange[1][0] = d2InputPlotRange[1][0] - bound_range[1]/10
d2InputPlotRange[1][1] = d2InputPlotRange[1][1] + bound_range[1]/10

# %% Generate initial experimental design
NUM_SAMPLES = NUM_INPUT_DIMS * 4
d2SolutionInputHypi = doe.lhs(NUM_INPUT_DIMS, samples=NUM_SAMPLES, criterion='center')
d2SolutionInputHypi[:,0] = (d2SolutionInputHypi[:,0] * bound_range[0]) + bounds[0][0]
d2SolutionInputHypi[:,1] = (d2SolutionInputHypi[:,1] * bound_range[1]) + bounds[1][0]
d2SolutionInputXhvi = d2SolutionInputHypi.copy()
d2SolutionOutputHypi = np.array(list( map(evaluate_fitness, d2SolutionInputHypi) ))
d2SolutionOutputXhvi = np.array(list( map(evaluate_fitness, d2SolutionInputXhvi) ))

# Generate map across input space
d2X1, d2X2 = np.meshgrid(np.linspace(bounds[0][0], bounds[0][1], 50), np.linspace(bounds[1][0], bounds[1][1], 50))
d2TestPoints = np.hstack((d2X1.reshape((-1,1)), d2X2.reshape((-1,1))))
d2TestResults = np.array(list( map(evaluate_fitness, d2TestPoints)))
d2OutputRange = d2PFRange
d2OutputRange[1,:] = np.max(d2TestResults, axis = 0)
d2Sol1 = d2TestResults[:,0].reshape(d2X1.shape)
d2Sol2 = d2TestResults[:,1].reshape(d2X1.shape)

d1Reference = 1.1 * np.max(d2TestResults, axis=0)
dMaxHyperVolume = pf.calculateHypervolume(d2F1F2_PF, d1Reference)

d1F1Contours = np.arange(0.25, 0.8, 0.25) * (d2OutputRange[1,0] - d2OutputRange[0,0]) + d2OutputRange[0,0]
d1F2Contours = np.arange(0.25, 0.8, 0.25) * (d2OutputRange[1,1] - d2OutputRange[0,1]) + d2OutputRange[0,1]

# %% xHVI example
des_space = GPyOpt.core.task.space.Design_space(domain)
acq_opt = GPyOpt.optimization.AcquisitionOptimizer(des_space, optimizer='lbfgs')

optimizer = GPyOpt.optimization.optimizer.OptLbfgs(bounds)

reference = np.repeat(1.0, NUM_OBJECTIVES).tolist()

i = NUM_SAMPLES
q = 1
d1HypervolumeHypi = []
d1HypervolumeXhvi = []
while i < NUM_TOTAL_EVALUATIONS:
    # Plot archive
    fig = plt.figure(figsize=[figs.plot_size * 2.0, figs.plot_size * 2.], tight_layout=True)
    gs = gridspec.GridSpec(ncols=5, nrows=4, figure=fig, height_ratios=[1,1,1,1], width_ratios=[1,1,1,1,1], hspace=0.5, wspace=0.2)
    
    # %% Plot F1
    ax = fig.add_subplot(gs[1, 0])
    if i > NUM_SAMPLES:
        figs.plot_map_with_points(d2X1, d2X2, d2Sol1, d1F1Contours, 'RdYlGn_r', d2OutputRange[:,0], 
                                  d2SolutionInputHypi[:,0], d2SolutionInputHypi[:,1],
                                  d1XScat2=d2SolutionInputHypi[-1,0], d1YScat2=d2SolutionInputHypi[-1,1],
                                  sYLabel='$x_2$', sTitle='$f_1$')
    else:
        figs.plot_map_with_points(d2X1, d2X2, d2Sol1, d1F1Contours, 'RdYlGn_r', d2OutputRange[:,0],
                                  d2SolutionInputHypi[:,0], d2SolutionInputHypi[:,1],
                                  sYLabel='$x_2$', sTitle='$f_1$')
    plt.xlim(d2InputPlotRange[0])
    plt.ylim(d2InputPlotRange[1])
    
    ax = fig.add_subplot(gs[1, -2])
    if i > NUM_SAMPLES:
        figs.plot_map_with_points(d2X1, d2X2, d2Sol1, d1F1Contours, 'RdYlGn_r', d2OutputRange[:,0], 
                                  d2SolutionInputXhvi[:,0], d2SolutionInputXhvi[:,1],
                                  d1XScat2=d2SolutionInputXhvi[-1,0], d1YScat2=d2SolutionInputXhvi[-1,1],
                                  sTitle='$f_1$')
    else:
        figs.plot_map_with_points(d2X1, d2X2, d2Sol1, d1F1Contours, 'RdYlGn_r', d2OutputRange[:,0],
                                  d2SolutionInputXhvi[:,0], d2SolutionInputXhvi[:,1],
                                  sTitle='$f_1$')
    plt.xlim(d2InputPlotRange[0])
    plt.ylim(d2InputPlotRange[1])
    
    # %% Plot F2
    ax = fig.add_subplot(gs[1, 1])
    if i > NUM_SAMPLES:
        figs.plot_map_with_points(d2X1, d2X2, d2Sol2, d1F2Contours, 'RdYlGn_r', d2OutputRange[:,1],
                              d2SolutionInputHypi[:,0], d2SolutionInputHypi[:,1],
                              d1XScat2=d2SolutionInputHypi[-1,0], d1YScat2=d2SolutionInputHypi[-1,1],
                              sTitle='$f_2$')
    else:
        figs.plot_map_with_points(d2X1, d2X2, d2Sol2, d1F2Contours, 'RdYlGn_r', d2OutputRange[:,1],
                              d2SolutionInputHypi[:,0], d2SolutionInputHypi[:,1],
                              sTitle='$f_2$')
    plt.xlim(d2InputPlotRange[0])
    plt.ylim(d2InputPlotRange[1])
    
    ax = fig.add_subplot(gs[1, -1])
    if i > NUM_SAMPLES:
        figs.plot_map_with_points(d2X1, d2X2, d2Sol2, d1F2Contours, 'RdYlGn_r', d2OutputRange[:,1],
                              d2SolutionInputXhvi[:,0], d2SolutionInputXhvi[:,1],
                              d1XScat2=d2SolutionInputXhvi[-1,0], d1YScat2=d2SolutionInputXhvi[-1,1],
                              sTitle='$f_2$')
    else:
        figs.plot_map_with_points(d2X1, d2X2, d2Sol2, d1F2Contours, 'RdYlGn_r', d2OutputRange[:,1],
                              d2SolutionInputXhvi[:,0], d2SolutionInputXhvi[:,1],
                              sTitle='$f_2$')
    plt.xlim(d2InputPlotRange[0])
    plt.ylim(d2InputPlotRange[1])
    
    # %% Plot F1 v F2
    d1TestHvcsHypi = pf.calculatePotentialHvcsGivenFront(d2TestResults, d2SolutionOutputHypi, d1Reference)
    d1TestHvcsRangeHypi = [np.min(d1TestHvcsHypi), np.max(d1TestHvcsHypi)]
    dCurrentHypervolumeHypi = pf.calculateHypervolume(d2SolutionOutputHypi, d1Reference)
    
    d1TestHvcsXhvi = pf.calculatePotentialHvcsGivenFront(d2TestResults, d2SolutionOutputXhvi, d1Reference)
    d1TestHvcsRangeXhvi = [np.min(d1TestHvcsHypi), np.max(d1TestHvcsXhvi)]
    dCurrentHypervolumeXhvi = pf.calculateHypervolume(d2SolutionOutputXhvi, d1Reference)
    
    ax = fig.add_subplot(gs[0, :2])
    if i > NUM_SAMPLES:
        figs.plot_sample_with_points(d2TestResults[:,0], d2TestResults[:,1], d1TestHvcsHypi, 'RdYlGn', [0., (dMaxHyperVolume - dCurrentHypervolumeHypi)/2], '',
                              d2SolutionOutputHypi[:,0], d2SolutionOutputHypi[:,1], 'k', '',
                              d1X2=d2SolutionOutputHypi[-1,0], d1Y2=d2SolutionOutputHypi[-1,1], sCol2='k',
                              sXLabel='$f_1$', sYLabel='$f_2$', sTitle='HypI \n Hypervolume Loss: {:.1f}%'.format((dMaxHyperVolume - dCurrentHypervolumeHypi)/dMaxHyperVolume * 100.))
    else:
        figs.plot_sample_with_points(d2TestResults[:,0], d2TestResults[:,1], d1TestHvcsHypi, 'RdYlGn', [0., (dMaxHyperVolume - dCurrentHypervolumeHypi)/2], '',
                              d2SolutionOutputHypi[:,0], d2SolutionOutputHypi[:,1], 'k', '',
                              sXLabel='$f_1$', sYLabel='$f_2$', sTitle='HypI \n Hypervolume Loss: {:.1f}%'.format((dMaxHyperVolume - dCurrentHypervolumeHypi)/dMaxHyperVolume * 100.))
    plt.plot(d2F1F2_PF[:,0], d2F1F2_PF[:,1], 'g.')
    # ax.set_aspect(0.2)
    
    ax = fig.add_subplot(gs[0, -2:])
    if i > NUM_SAMPLES:
        figs.plot_sample_with_points(d2TestResults[:,0], d2TestResults[:,1], d1TestHvcsXhvi, 'RdYlGn', [0., (dMaxHyperVolume - dCurrentHypervolumeXhvi)/2], '',
                              d2SolutionOutputXhvi[:,0], d2SolutionOutputXhvi[:,1], 'k', '',
                              d1X2=d2SolutionOutputXhvi[-1,0], d1Y2=d2SolutionOutputXhvi[-1,1], sCol2='k',
                              sXLabel='$f_1$', sTitle='xHVI \n Hypervolume Loss: {:.1f}%'.format((dMaxHyperVolume - dCurrentHypervolumeXhvi)/dMaxHyperVolume * 100.))
    else:
        figs.plot_sample_with_points(d2TestResults[:,0], d2TestResults[:,1], d1TestHvcsXhvi, 'RdYlGn', [0., (dMaxHyperVolume - dCurrentHypervolumeXhvi)/2], '',
                              d2SolutionOutputXhvi[:,0], d2SolutionOutputXhvi[:,1], 'k', '',
                              sXLabel='$f_1$', sTitle='xHVI \n Hypervolume Loss: {:.1f}%'.format((dMaxHyperVolume - dCurrentHypervolumeXhvi)/dMaxHyperVolume * 100.))
    plt.plot(d2F1F2_PF[:,0], d2F1F2_PF[:,1], 'g.')
    # ax.set_aspect(0.2)
    
    # %% Plot True Hypervolume Improvement
    ax = fig.add_subplot(gs[3, 1])
    figs.plot_map_with_points(d2X1, d2X2, d1TestHvcsHypi.reshape(d2X1.shape), [0.5 * (dMaxHyperVolume - dCurrentHypervolumeHypi)/2], 
                              'RdYlGn', [0., (dMaxHyperVolume - dCurrentHypervolumeHypi)/2],
                              d2SolutionInputHypi[:,0], d2SolutionInputHypi[:,1], 
                              sXLabel='$x_1$', sTitle='Potential \n Hypervolume \n Improvement')
    plt.xlim(d2InputPlotRange[0])
    plt.ylim(d2InputPlotRange[1])
    
    ax = fig.add_subplot(gs[3, -2])
    figs.plot_map_with_points(d2X1, d2X2, d1TestHvcsXhvi.reshape(d2X1.shape), [0.5 * (dMaxHyperVolume - dCurrentHypervolumeXhvi)/2], 
                              'RdYlGn', [0., (dMaxHyperVolume - dCurrentHypervolumeHypi)/2],
                              d2SolutionInputXhvi[:,0], d2SolutionInputXhvi[:,1], 
                              sXLabel='$x_1$', sTitle='Potential \n Hypervolume \n Improvement')
    plt.xlim(d2InputPlotRange[0])
    plt.ylim(d2InputPlotRange[1])
    
    # d1Offset = np.diff(d2OutputRange, axis=0) * 0.1
    # plt.xlim([d2OutputRange[0,0]-d1Offset[0], d2OutputRange[0,1]+d1Offset[0]])
    # plt.ylim([d2OutputRange[1,0]-d1Offset[1], d2OutputRange[1,1]+d1Offset[1]])
    
    if q == 1:
        plt.savefig(os.path.join("video","{0}_{1:03d}.png".format(FUNCTION_NAME,q)), facecolor=None, edgecolor=None)
        q = q+1
    
    # %% Normalise everything
    y_norm_hypi = normalise_f(np.array([(d2SolutionOutputHypi[:, 0])]).transpose(), 0.0)
    for m in range(1, NUM_OBJECTIVES):
        y_norm_hypi = np.hstack((y_norm_hypi, normalise_f(np.array([(d2SolutionOutputHypi[:, m])]).transpose(), 0.0)))
        
    y_norm_xhvi = normalise_f(np.array([(d2SolutionOutputXhvi[:, 0])]).transpose(), 0.0)
    for m in range(1, NUM_OBJECTIVES):
        y_norm_xhvi = np.hstack((y_norm_xhvi, normalise_f(np.array([(d2SolutionOutputXhvi[:, m])]).transpose(), 0.0)))
    
    # test_f_norm = normalise_f(np.array([(d2TestResults[:, 0])]).transpose(), 0.0,
    #                           np.array([(d2SolutionOutput[:, 0])]).transpose())
    # for m in range(1, NUM_OBJECTIVES):
    #     test_f_norm = np.hstack((test_f_norm, normalise_f(np.array([(d2TestResults[:, m])]).transpose(), 0.0,
    #                                                       np.array([(d2SolutionOutput[:, m])]).transpose())))
    
    # d1TrueHVI = pf.calculatePotentialHvcsGivenFront(test_f_norm, y_norm, reference)
    # d2TrueHVI = d1TrueHVI.reshape(d2X1.shape)
    
    # Plot HV progress
    d1HypervolumeHypi.append(pf.calculateHypervolume(d2SolutionOutputHypi, d1Reference))
    d1HypervolumeXhvi.append(pf.calculateHypervolume(d2SolutionOutputXhvi, d1Reference))
    
    # plt.subplot(ax[1,0])
    # plt.plot(np.arange(NUM_SAMPLES,i+1), (1. - np.array(d1Hypervolume)/d1MaxHyperVolume)*100.)
    
    # ax = fig.add_subplot(gs[1, :])
    # figs.plot_sample_with_points(test_f_norm[:,0], test_f_norm[:,1], d1TrueHVI[:,0], 'RdYlGn', [0., 0.2], 'True potential HVI',
    #                               y_norm[:,0], y_norm[:,1], 'k', 'Current archive',
    #                               sXLabel='$f_{1,norm}$', sYLabel='$f_{2,norm}$', sTitle='Potential HVI')
    
    # plt.savefig(os.path.join("video","xHVI_{0}_{1}_f1f2.png".format(FUNCTION_NAME,i)), facecolor=None, edgecolor=None)

    # %% Calculate infill criteria
    d1InfillHypi = pf.calculateHypIs(y_norm_hypi, reference)
    d1InfillHypi_norm = normalise_f(d1InfillHypi, ZETA)
    
    d1HVI = pf.calculateHypervolumeContributions(y_norm_xhvi, reference)
    d1HVIn = pf.calculateNegativeHypervolumeContributions(y_norm_xhvi)
    d1InfillXhvi = (d1HVI - d1HVIn)
    d1InfillXhvi_norm = normalise_f(d1InfillXhvi, ZETA)  
    
    # %% Fit GP models to infill criteria
    modelHypi = GPyOpt.models.GPModel(exact_feval=True, ARD=True)
    modelHypi.updateModel(d2SolutionInputHypi[:,:], -d1InfillHypi_norm, [], [])
    
    muHypi, stdevHypi = modelHypi.predict(d2TestPoints)
    ei_infill_hypi = np.array(list( map(expected_improvement, muHypi, stdevHypi, repeat(min(-d1InfillHypi_norm)))))
    d2InfillHypiNormMean = muHypi.reshape(d2X1.shape)
    d2InfillHypiNormStd = stdevHypi.reshape(d2X1.shape)
    d2InfillHypiEI = ei_infill_hypi.reshape(d2X1.shape)
    
    modelXhvi = GPyOpt.models.GPModel(exact_feval=True, ARD=True)
    modelXhvi.updateModel(d2SolutionInputXhvi[:,:], -d1InfillXhvi_norm, [], [])
    
    muXhvi, stdevXhvi = modelXhvi.predict(d2TestPoints)
    ei_infill_xhvi = np.array(list( map(expected_improvement, muXhvi, stdevXhvi, repeat(min(-d1InfillXhvi_norm)))))
    d2InfillXhviNormMean = muXhvi.reshape(d2X1.shape)
    d2InfillXhviNormStd = stdevXhvi.reshape(d2X1.shape)
    d2InfillXhviEI = ei_infill_xhvi.reshape(d2X1.shape)
    
    # %% Plot GP models
    ax = fig.add_subplot(gs[2, 1])
    figs.plot_map_with_points(d2X1, d2X2, d2InfillHypiNormMean, np.linspace(-0.5, 0.5, 5), 'RdYlGn_r', [-0.5, 0.5],
                              d2SolutionInputHypi[:,0], d2SolutionInputHypi[:,1], d1InfillHypi_norm[:,0],'RdYlGn', 
                              sTitle='GP Mean \n $\mu(-HypI_{{norm}}$)')
    plt.xlim(d2InputPlotRange[0])
    plt.ylim(d2InputPlotRange[1])
    
    ax = fig.add_subplot(gs[2, 0])
    figs.plot_map_with_points(d2X1, d2X2, d2InfillHypiNormStd, np.linspace(0.05, 0.2, 4), 'RdYlGn_r', [0., 0.2],
                              d2SolutionInputHypi[:,0], d2SolutionInputHypi[:,1], 'green',
                              sYLabel='$x_2$',sTitle="GP Uncertainty \n $\sigma(-HypI_{{norm}}$)")
    plt.xlim(d2InputPlotRange[0])
    plt.ylim(d2InputPlotRange[1])
    
    ax = fig.add_subplot(gs[2, -2])
    figs.plot_map_with_points(d2X1, d2X2, d2InfillXhviNormMean, np.linspace(-0.5, 0.5, 5), 'RdYlGn_r', [-0.5, 0.5],
                              d2SolutionInputXhvi[:,0], d2SolutionInputXhvi[:,1], d1InfillXhvi_norm[:,0],'RdYlGn', 
                              sTitle='GP Mean \n $\mu(-xHVI_{{norm}}$)')
    plt.xlim(d2InputPlotRange[0])
    plt.ylim(d2InputPlotRange[1])
    
    ax = fig.add_subplot(gs[2, -1])
    figs.plot_map_with_points(d2X1, d2X2, d2InfillXhviNormStd, np.linspace(0.05, 0.2, 4), 'RdYlGn_r', [0., 0.2],
                              d2SolutionInputXhvi[:,0], d2SolutionInputXhvi[:,1], 'green',
                              sTitle="GP Uncertainty \n $\sigma(-xHVI_{{norm}}$)")
    plt.xlim(d2InputPlotRange[0])
    plt.ylim(d2InputPlotRange[1])
    
    if q == 2:
        plt.savefig(os.path.join("video","{0}_{1:03d}.png".format(FUNCTION_NAME,q)), facecolor=None, edgecolor=None)
        q = q+1
    
    # Run acquisition function
    acqHypi = GPyOpt.acquisitions.AcquisitionEI(modelHypi, des_space, jitter = 0.0, optimizer = acq_opt)
    next_point_hypi, y_next_est_hypi = acqHypi.optimize()
    
    acqXhvi = GPyOpt.acquisitions.AcquisitionEI(modelXhvi, des_space, jitter = 0.0, optimizer = acq_opt)
    next_point_xhvi, y_next_est_xhvi = acqXhvi.optimize()
    
    # ax = fig.add_subplot(gs[2, 0])
    # figs.plot_map_with_points(d2X1, d2X2, d2TrueHVI/np.max(d2TrueHVI)*100., np.linspace(10, 90, 5), 'RdYlGn', [0., 100.],
    #                           d2SolutionInput[:,0], d2SolutionInput[:,1], d1Infill[:,0]/np.max(d1Infill[:,0])*100., 'RdYlGn', 
    #                           sXLabel='$x_1$', sYLabel='$x_2$', sTitle='Potential HVI (%)')
    # plt.xlim((0.,1.))
    # plt.ylim((0.,1.))
    
    ax = fig.add_subplot(gs[3, 0])
    
    infill_ei_rat = d2InfillHypiEI/np.max(d2InfillHypiEI)
    true_imp_rat = d1TestHvcsHypi.reshape(d2X1.shape) / np.max(d1TestHvcsHypi)
    
    figs.plot_map_with_points(d2X1, d2X2, infill_ei_rat, np.linspace(0.1, 0.9, 3), 'RdYlGn', [0., 1.],
                              d2SolutionInputHypi[:,0], d2SolutionInputHypi[:,1], 
                              (d1InfillHypi-np.min(d1InfillHypi))/(np.max(d1InfillHypi)-np.min(d1InfillHypi)), 'RdYlGn', 
                              d1XScat2=next_point_hypi[0][0], d1YScat2=next_point_hypi[0][1],
                              sXLabel='$x_1$', sYLabel='$x_2$',sTitle='Expected Improvement\n $EI_{{norm}}$(HypI)\n Next: [{0:.2f}, {1:.2f}]'.format(
                                  next_point_hypi[0][0], next_point_hypi[0][1]))
    plt.xlim(d2InputPlotRange[0])
    plt.ylim(d2InputPlotRange[1])
    
    ax = fig.add_subplot(gs[2, 2])
    figs.plot_map_with_points(d2X1, d2X2, true_imp_rat - infill_ei_rat, np.arange(-0.5, 0.6, 0.5), 'bwr', [-1., 1.],
                              d2SolutionInputHypi[:,0], d2SolutionInputHypi[:,1], 
                              sTitle='HypI\n Acquisition Error')
    plt.xlim(d2InputPlotRange[0])
    plt.ylim(d2InputPlotRange[1])
    
    
    ax = fig.add_subplot(gs[3, -1])
    
    infill_ei_rat = d2InfillXhviEI/np.max(d2InfillXhviEI)
    true_imp_rat = d1TestHvcsXhvi.reshape(d2X1.shape) / np.max(d1TestHvcsXhvi)
    
    figs.plot_map_with_points(d2X1, d2X2, infill_ei_rat, np.linspace(0.1, 0.9, 3), 'RdYlGn', [0., 1.],
                              d2SolutionInputXhvi[:,0], d2SolutionInputXhvi[:,1], 
                              (d1InfillXhvi-np.min(d1InfillXhvi))/(np.max(d1InfillXhvi)-np.min(d1InfillXhvi)), 'RdYlGn', 
                              d1XScat2=next_point_xhvi[0][0], d1YScat2=next_point_xhvi[0][1],
                              sXLabel='$x_1$', sTitle='Expected Improvement\n $EI_{{norm}}$(xHVI)\n Next: [{0:.2f}, {1:.2f}]'.format(
                                  next_point_xhvi[0][0], next_point_xhvi[0][1]))
    plt.xlim(d2InputPlotRange[0])
    plt.ylim(d2InputPlotRange[1])
    
    ax = fig.add_subplot(gs[3, -3])
    figs.plot_map_with_points(d2X1, d2X2, true_imp_rat - infill_ei_rat, np.arange(-0.5, 0.6, 0.5), 'bwr', [-1., 1.],
                              d2SolutionInputXhvi[:,0], d2SolutionInputXhvi[:,1], 
                              sXLabel='$x_1$', sTitle='xHVI\n Acquisition Error')
    plt.xlim(d2InputPlotRange[0])
    plt.ylim(d2InputPlotRange[1])
    
    plt.savefig(os.path.join("video","{0}_{1:03d}.png".format(FUNCTION_NAME,q)), facecolor=None, edgecolor=None)
    q = q+1
    
    # %% Archive and increment
    d2SolutionInputHypi = np.vstack((d2SolutionInputHypi, next_point_hypi[0]))
    d2SolutionOutputHypi = np.vstack((d2SolutionOutputHypi, evaluate_fitness(next_point_hypi[0])))
    d2SolutionInputXhvi = np.vstack((d2SolutionInputXhvi, next_point_xhvi[0]))
    d2SolutionOutputXhvi = np.vstack((d2SolutionOutputXhvi, evaluate_fitness(next_point_xhvi[0])))
    i += 1
    
    