# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 13:09:15 2020

@author: Clym Stock-Williams

Creates elements of Figure 4 from the PPSN 2020 paper.
"""
import pyDOE as doe
import numpy as np
import os
from itertools import repeat
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import rcParams
import ParetoFrontND as pf
import StandardTestFunctions as fn
import GPyOpt

plt.style.use('ggplot')
rcParams['font.sans-serif'] = "Segoe UI"
rcParams['font.family'] = "sans-serif"

plot_size = 10.0

def fit_model(d2X, d1Y):
    model = GPyOpt.models.GPModel(exact_feval=True, ARD=True)
    model.updateModel(d2X, d1Y, [], [])
    return model

def expected_improvement(mu, sigma, y_star):
    s = (y_star - mu) / sigma
    return sigma * (s * stats.norm.cdf(s) + stats.norm.pdf(s))

def normalise_f(f, exploration_param):
    avg = np.mean(f)
    low = np.min(f)
    high = np.max(f)
    offset = (1 - exploration_param) * avg + exploration_param * low
    return (f - offset) / (high - low + 1e-6)

np.random.seed(1234)

# %% Setup problem 
FUNCTION_NAME = "ZDT3"

NUM_INPUT_DIMS = 2
NUM_OBJECTIVES = 2

ZETA = 0.0
#REFERENCE = 1.2
# REFERENCE_START = 1.8
# REFERENCE_END = 1.2
#ABSOLUTE_REFERENCE = [REFERENCE, 10.]
# ABSOLUTE_REFERENCE = [100., 100.]

# d1Reference = np.repeat(1000.0, NUM_OBJECTIVES).tolist()
d1Reference = [1.1, 1000.0]

# Define input domain in GPyOpt format and fitness evaluation function
domain, fitnessfunc, d1x_opt, NUM_INPUT_DIMS, NUM_OBJECTIVES = fn.get_function_definition(
    FUNCTION_NAME, NUM_INPUT_DIMS, NUM_OBJECTIVES)

def evaluate_fitness(ind):
    assert len(ind) == NUM_INPUT_DIMS
    return fitnessfunc(ind)

# d1F1F2 = np.array(list( map(evaluate_fitness, d1x_opt) ))
# d1F1F2_PF, _ = pf.getNonDominatedFront(d1F1F2)

# %% Generate initial experimental design
NUM_SAMPLES = NUM_INPUT_DIMS * 4
d2SolutionInput = doe.lhs(NUM_INPUT_DIMS, samples=NUM_SAMPLES, criterion='center')
d2SolutionOutput = np.array(list( map(evaluate_fitness, d2SolutionInput) ))

# Generate map across input space
d1Test = np.linspace(0.0, 1.0, 50)
d2X, d2Y = np.meshgrid(d1Test, d1Test)
d2TestPoints = np.hstack((d2X.reshape((-1,1)), d2Y.reshape((-1,1))))
d2TestResults = np.array(list( map(evaluate_fitness, d2TestPoints)))
d2Sol1 = d2TestResults[:,0].reshape(d2X.shape)
d2Sol2 = d2TestResults[:,1].reshape(d2X.shape)

fig, ax = plt.subplots(1, 2)

plt.subplot(ax[0])
contours = plt.contour(d2X, d2Y, d2Sol1, [0.25,0.5,0.75], colors='black')
plt.clabel(contours, inline=True, fontsize=7)
plt.imshow(d2Sol1, extent=[0, 1.0, 0, 1.0], origin='lower',
           cmap='RdYlGn_r', alpha=0.5, vmin=0, vmax=1)
plt.scatter(d2SolutionInput[:,0], d2SolutionInput[:,1], c = 'black', marker = 'x', label = 'Initial Design')
plt.xlabel('$x_1$', fontsize=10)
plt.ylabel('$x_2$', fontsize=10)
plt.title('$f_1$', fontsize=10)
plt.tick_params(
    axis='both', 
    left=True,
    labelleft=True, 
    bottom=True,
    labelbottom=True)
for tick in ax[0].get_xticklabels():
    tick.set_fontsize(9)
for tick in ax[0].get_yticklabels():
    tick.set_fontsize(9)

plt.subplot(ax[1])
contours = plt.contour(d2X, d2Y, d2Sol2, [1,3,5,7,9], colors='black')
plt.clabel(contours, inline=True, fontsize=8)
plt.imshow(d2Sol2, extent=[0, 1.0, 0, 1.0], origin='lower',
           cmap='RdYlGn_r', alpha=0.5, vmin=0, vmax=10)
plt.scatter(d2SolutionInput[:,0], d2SolutionInput[:,1], c = 'black', marker = 'x', label = 'Initial Design')
plt.xlabel('$x_1$', fontsize=10)
plt.title('$f_2$', fontsize=10)
plt.tick_params(
    axis='both', 
    left=False,
    labelleft=False, 
    bottom=True,
    labelbottom=True)
for tick in ax[1].get_xticklabels():
    tick.set_fontsize(9)
for tick in ax[1].get_yticklabels():
    tick.set_fontsize(9)

plt.savefig(os.path.join("img","figure_4_a1.svg"), facecolor=None, edgecolor=None)

plt.figure(figsize=[plot_size * 1.62, plot_size])
plot1 = plt.plot(d2TestResults[:,0], d2TestResults[:,1], 
                 linestyle='', marker = '.', markersize=plot_size, color = 'lightblue', 
                 label = 'Grid in input domain')
plt.plot(d2SolutionOutput[:,0], d2SolutionOutput[:,1], 
         c = 'black', linestyle='', marker = 'x', markersize=plot_size*1.5, 
         label = 'Initial Design')
d2TestFront, _ = pf.getNonDominatedFront(d2TestResults)
plt.plot(d2TestFront[:,0], d2TestFront[:,1], 
         linestyle = '', marker = '.', color = 'g', markersize = plot_size*1.5, 
         label = 'Pareto Front')
plt.xlabel('$f_1$', fontsize=plot_size*3.0)
plt.ylabel('$f_2$', fontsize=plot_size*3.0)
plt.tick_params(
    axis='both', 
    left=True,
    labelleft=True, 
    bottom=True,
    labelbottom=True)
for tick in plot1[0].axes.get_xticklabels():
    tick.set_fontsize(plot_size*2.0)
for tick in plot1[0].axes.get_yticklabels():
    tick.set_fontsize(plot_size*2.0)
plt.legend(loc='upper right', labelspacing=0.25, fontsize=plot_size*2.0)

plt.savefig(os.path.join("img","figure_4_a2.svg"), facecolor=None, edgecolor=None)

# %% xHVI example
des_space = GPyOpt.core.task.space.Design_space(domain)
acq_opt = GPyOpt.optimization.AcquisitionOptimizer(des_space, optimizer='lbfgs')
y_norm = normalise_f(np.array([(d2SolutionOutput[:, 0])]).transpose(), 0.0)
for m in range(1, NUM_OBJECTIVES):
    y_norm = np.hstack((y_norm, normalise_f(np.array([(d2SolutionOutput[:, m])]).transpose(), 0.0)))
#Calculate xHVI
reference = np.repeat(1.0, NUM_OBJECTIVES).tolist()
d1HVI = pf.calculateHypervolumeContributions(y_norm, reference)
d1HVIn = pf.calculateNegativeHypervolumeContributions(y_norm)
d1xHVI = (d1HVI - d1HVIn)
d1xHVI_norm = normalise_f(d1xHVI, ZETA)
# Fit a GP model to xHVC
model = GPyOpt.models.GPModel(exact_feval=True, ARD=True)
model.updateModel(d2SolutionInput[:,:], -d1xHVI_norm, [], [])
# Run acquisition function
acq = GPyOpt.acquisitions.AcquisitionEI(model, des_space, jitter = 0.0, optimizer = acq_opt)
next_point, y_next_est = acq.optimize()
# Evaluate fitness and archive
y_next = evaluate_fitness(next_point[0])

# Figure: calculated xHVI
plt.figure(figsize=[plot_size * 1.62, plot_size])
scat1 = plt.scatter(y_norm[:,0], y_norm[:,1], c = d1xHVI[:,0], s=plot_size*25.0,
           cmap='RdYlGn', vmin=min(d1xHVI), vmax=-min(d1xHVI),
           linewidths=1, edgecolors='k')
plt.axis('equal')
plt.plot(reference[0], reference[1], marker = 'x', markersize = plot_size*2.0, color = 'k')
plt.text(0.51, 0.92, 'reference point $r$', fontsize=plot_size*2.0)
plt.xlabel('$f_1$ (normalised)', fontsize=plot_size*3.0)
plt.ylabel('$f_2$ (normalised)', fontsize=plot_size*3.0)
cb = plt.colorbar()
for t in cb.ax.get_yticklabels():
    t.set_fontsize(plot_size*2.0)
cb.set_label(label = 'xHVI', fontsize=plot_size*3.0)
for tick in scat1.axes.get_xticklabels():
    tick.set_fontsize(plot_size*2.0)
for tick in scat1.axes.get_yticklabels():
    tick.set_fontsize(plot_size*2.0)
    
plt.savefig(os.path.join("img","figure_4_b1.svg"), facecolor=None, edgecolor=None)

# Figures: surrogate for normalised xHVI
mu, stdev = model.predict(d2TestPoints)
ei_xhvi = np.array(list( map(expected_improvement, mu, stdev, repeat(min(-d1xHVI_norm)))))
d2xhvc_pred = mu.reshape(d2X.shape)
d2xhvc_std_pred = stdev.reshape(d2X.shape)
d2xhvi_ei = ei_xhvi.reshape(d2X.shape)

plt.figure(figsize=[plot_size, plot_size])
ax = plt.axes([0, 0.05, 0.9, 0.9 ])
contours = plt.contour(d2X, d2Y, d2xhvc_pred, np.linspace(-0.5, 0.5, 5), colors='black')
plt.clabel(contours, inline=True, fontsize=plot_size*1.5)
im1 = plt.imshow(d2xhvc_pred, extent=[0, 1.0, 0, 1.0], origin='lower',
           cmap='RdYlGn_r', alpha=0.5, vmin=-0.5, vmax=0.5)
plt.scatter(d2SolutionInput[:,0], d2SolutionInput[:,1], c=d1xHVI_norm[:,0], s=plot_size*25.0,
            cmap='RdYlGn', vmin=-0.5, vmax=0.5,
            linewidths=1, edgecolors='k')
plt.xlabel('$x_1$', fontsize=plot_size*3.0)
plt.ylabel('$x_2$', fontsize=plot_size*3.0)
plt.title('Mean function from $-xHVI_{norm}$ surrogate', fontsize=plot_size*3.0)
cax = plt.axes([0.95, 0.05, 0.05, 0.9])
cb = plt.colorbar(mappable=im1, cax=cax)
for t in cb.ax.get_yticklabels():
    t.set_fontsize(plot_size*2.0)
# cb.set_label(label = '$-xHVI_{norm}$', fontsize=plot_size*3.0)
for tick in im1.axes.get_xticklabels():
    tick.set_fontsize(plot_size*2.0)
for tick in im1.axes.get_yticklabels():
    tick.set_fontsize(plot_size*2.0)

plt.savefig(os.path.join("img","figure_4_b2a.svg"), facecolor=None, edgecolor=None)

plt.figure(figsize=[plot_size, plot_size])
ax = plt.axes([0, 0.05, 0.9, 0.9])
contours = plt.contour(d2X, d2Y, d2xhvc_std_pred, np.linspace(0.05, 0.2, 4), colors='black')
plt.clabel(contours, inline=True, fontsize=plot_size*1.5)
im1 = plt.imshow(d2xhvc_std_pred, extent=[0, 1.0, 0, 1.0], origin='lower',
           cmap='RdYlGn_r', alpha=0.5, vmin=0.0, vmax=0.2)
plt.scatter(d2SolutionInput[:,0], d2SolutionInput[:,1], c ='g', s=plot_size*25.0,
            linewidths=1, edgecolors='k')
plt.xlabel('$x_1$', fontsize=plot_size*3.0)
plt.ylabel('$x_2$', fontsize=plot_size*3.0)
plt.title('Standard deviation from $-xHVI_{norm}$ surrogate', fontsize=plot_size*3.0);
cax = plt.axes([0.95, 0.05, 0.05, 0.9])
cb = plt.colorbar(mappable=im1, cax=cax)
for t in cb.ax.get_yticklabels():
    t.set_fontsize(plot_size*2.0)
# cb.set_label(label = '$xHVC_{norm}$', fontsize=10)
for tick in im1.axes.get_xticklabels():
    tick.set_fontsize(plot_size*2.0)
for tick in im1.axes.get_yticklabels():
    tick.set_fontsize(plot_size*2.0)

plt.savefig(os.path.join("img","figure_4_b2b.svg"), facecolor=None, edgecolor=None)

# Figure: EI(xHVI) acquisition
plt.figure(figsize=[plot_size, plot_size])
ax = plt.axes([0, 0.05, 0.9, 0.9 ])
contours = plt.contour(d2X, d2Y, d2xhvi_ei, np.linspace(0.05, 0.2, 4), colors='black')
plt.clabel(contours, inline=True, fontsize=plot_size*1.5)
im1 = plt.imshow(d2xhvi_ei, extent=[0, 1.0, 0, 1.0], origin='lower',
           cmap='RdYlGn', alpha=0.5, vmin=0.0, vmax=0.2)
plt.scatter(next_point[0][0], next_point[0][1], c='k', marker = 'x', 
            s=plot_size*25.0, linewidth=5)
plt.xlabel('$x_1$', fontsize=plot_size*3.0)
plt.ylabel('$x_2$', fontsize=plot_size*3.0)
plt.title('Expected Improvement Acquisition with xHVI', fontsize=plot_size*3.0);
cax = plt.axes([0.95, 0.05, 0.05, 0.9])
cb = plt.colorbar(mappable=im1, cax=cax)
for t in cb.ax.get_yticklabels():
    t.set_fontsize(plot_size*2.0)
cb.set_label(label = 'Expected Improvement', fontsize=plot_size*3.0)
for tick in im1.axes.get_xticklabels():
    tick.set_fontsize(plot_size*2.0)
for tick in im1.axes.get_yticklabels():
    tick.set_fontsize(plot_size*2.0)

plt.savefig(os.path.join("img","figure_4_b3.svg"), facecolor=None, edgecolor=None)

# %% HypI example
#Calculate HypI
d1HypI = pf.calculateHypIs(y_norm, reference)
d1HypI_norm = normalise_f(d1HypI, ZETA)
# Fit a GP model to xHVC
model2 = GPyOpt.models.GPModel(exact_feval=True, ARD=True)
model2.updateModel(d2SolutionInput[:,:], -d1HypI_norm, [], [])
# Run acquisition function
acq2 = GPyOpt.acquisitions.AcquisitionEI(model2, des_space, jitter = 0.0, optimizer = acq_opt)
next_point2, y_next_est2 = acq2.optimize()
# Evaluate fitness and archive
y_next2 = evaluate_fitness(next_point2[0])

# Figure: calculated xHVI
plt.figure(figsize=[plot_size * 1.62, plot_size])
scat2 = plt.scatter(y_norm[:,0], y_norm[:,1], c = d1HypI[:,0], s=plot_size*25.0,
           cmap='RdYlGn', vmin=min(d1HypI), vmax=-min(d1HypI),
           linewidths=1, edgecolors='k')
plt.axis('equal')
plt.plot(reference[0], reference[1], marker = 'x', markersize = plot_size*2.0, color = 'k')
plt.text(0.51, 0.92, 'reference point $r$', fontsize=plot_size*2.0)
plt.xlabel('$f_1$ (normalised)', fontsize=plot_size*3.0)
plt.ylabel('$f_2$ (normalised)', fontsize=plot_size*3.0)
cb = plt.colorbar()
for t in cb.ax.get_yticklabels():
    t.set_fontsize(plot_size*2.0)
cb.set_label(label = 'HypI', fontsize=plot_size*3.0)
for tick in scat2.axes.get_xticklabels():
    tick.set_fontsize(plot_size*2.0)
for tick in scat2.axes.get_yticklabels():
    tick.set_fontsize(plot_size*2.0)
    
plt.savefig(os.path.join("img","figure_4_d1.svg"), facecolor=None, edgecolor=None)

# Figures: surrogate for normalised HypI
mu2, stdev2 = model2.predict(d2TestPoints)
ei_hypi = np.array(list( map(expected_improvement, mu2, stdev2, repeat(min(-d1HypI_norm)))))
d2hypi_pred = mu2.reshape(d2X.shape)
d2hypi_std_pred = stdev2.reshape(d2X.shape)
d2hypi_ei = ei_hypi.reshape(d2X.shape)

plt.figure(figsize=[plot_size, plot_size])
ax = plt.axes([0, 0.05, 0.9, 0.9 ])
contours = plt.contour(d2X, d2Y, d2hypi_pred, np.linspace(-0.5, 0.5, 5), colors='black')
plt.clabel(contours, inline=True, fontsize=plot_size*1.5)
im2 = plt.imshow(d2hypi_pred, extent=[0, 1.0, 0, 1.0], origin='lower',
           cmap='RdYlGn_r', alpha=0.5, vmin=-0.5, vmax=0.5)
plt.scatter(d2SolutionInput[:,0], d2SolutionInput[:,1], c=d1HypI_norm[:,0], s=plot_size*25.0,
            cmap='RdYlGn', vmin=-0.5, vmax=0.5,
            linewidths=1, edgecolors='k')
plt.xlabel('$x_1$', fontsize=plot_size*3.0)
plt.ylabel('$x_2$', fontsize=plot_size*3.0)
plt.title('Mean function from $-HypI_{norm}$ surrogate', fontsize=plot_size*3.0)
cax = plt.axes([0.95, 0.05, 0.05, 0.9])
cb = plt.colorbar(mappable=im2, cax=cax)
for t in cb.ax.get_yticklabels():
    t.set_fontsize(plot_size*2.0)
# cb.set_label(label = '$-HypI_{norm}$', fontsize=plot_size*3.0)
for tick in im2.axes.get_xticklabels():
    tick.set_fontsize(plot_size*2.0)
for tick in im2.axes.get_yticklabels():
    tick.set_fontsize(plot_size*2.0)

plt.savefig(os.path.join("img","figure_4_d2a.svg"), facecolor=None, edgecolor=None)

plt.figure(figsize=[plot_size, plot_size])
ax = plt.axes([0, 0.05, 0.9, 0.9])
contours = plt.contour(d2X, d2Y, d2hypi_std_pred, np.linspace(0.05, 0.2, 4), colors='black')
plt.clabel(contours, inline=True, fontsize=plot_size*1.5)
im2 = plt.imshow(d2hypi_std_pred, extent=[0, 1.0, 0, 1.0], origin='lower',
           cmap='RdYlGn_r', alpha=0.5, vmin=0.0, vmax=0.2)
plt.scatter(d2SolutionInput[:,0], d2SolutionInput[:,1], c ='g', s=plot_size*25.0,
            linewidths=1, edgecolors='k')
plt.xlabel('$x_1$', fontsize=plot_size*3.0)
plt.ylabel('$x_2$', fontsize=plot_size*3.0)
plt.title('Standard deviation from $-HypI_{norm}$ surrogate', fontsize=plot_size*3.0);
cax = plt.axes([0.95, 0.05, 0.05, 0.9])
cb = plt.colorbar(mappable=im2, cax=cax)
for t in cb.ax.get_yticklabels():
    t.set_fontsize(plot_size*2.0)
# cb.set_label(label = '$xHVC_{norm}$', fontsize=10)
for tick in im2.axes.get_xticklabels():
    tick.set_fontsize(plot_size*2.0)
for tick in im2.axes.get_yticklabels():
    tick.set_fontsize(plot_size*2.0)

plt.savefig(os.path.join("img","figure_4_d2b.svg"), facecolor=None, edgecolor=None)

# Figure: EI(HypI) acquisition
plt.figure(figsize=[plot_size, plot_size])
ax = plt.axes([0, 0.05, 0.9, 0.9 ])
contours = plt.contour(d2X, d2Y, d2hypi_ei, np.linspace(0.05, 0.2, 4), colors='black')
plt.clabel(contours, inline=True, fontsize=plot_size*1.5)
im2 = plt.imshow(d2hypi_ei, extent=[0, 1.0, 0, 1.0], origin='lower',
           cmap='RdYlGn', alpha=0.5, vmin=0.0, vmax=0.25)
plt.scatter(next_point[0][0], next_point[0][1], c='k', marker = 'x', 
            s=plot_size*25.0, linewidth=5)
plt.xlabel('$x_1$', fontsize=plot_size*3.0)
plt.ylabel('$x_2$', fontsize=plot_size*3.0)
plt.title('Expected Improvement Acquisition with HypI', fontsize=plot_size*3.0);
cax = plt.axes([0.95, 0.05, 0.05, 0.9])
cb = plt.colorbar(mappable=im2, cax=cax)
for t in cb.ax.get_yticklabels():
    t.set_fontsize(plot_size*2.0)
cb.set_label(label = 'Expected Improvement', fontsize=plot_size*3.0)
for tick in im2.axes.get_xticklabels():
    tick.set_fontsize(plot_size*2.0)
for tick in im2.axes.get_yticklabels():
    tick.set_fontsize(plot_size*2.0)

plt.savefig(os.path.join("img","figure_4_d3.svg"), facecolor=None, edgecolor=None)

# %% EHVI example
bounds = []
for q in range(len(domain)):
    bounds.append(domain[q]['domain'])
bounds = np.array(bounds)
optimizer = GPyOpt.optimization.optimizer.OptLbfgs(bounds)

# Fit GPs
f_norm = normalise_f(np.array([(d2SolutionOutput[:, 0])]).transpose(), ZETA)
for m in range(1, NUM_OBJECTIVES):
    f_norm = np.hstack((f_norm, normalise_f(np.array([(d2SolutionOutput[:, m])]).transpose(), ZETA)))
models = []
for m in range(NUM_OBJECTIVES):
    models.append(fit_model(d2SolutionInput, f_norm[:,m].reshape((-1,1))))
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
        [1., 1.],
        1000)
    return -ehvi # Maximise

# Run EHVI acquisition
ehvi_max = 0.0
x_next_ehvi = d2SolutionInput[0]
for n in range(10):
    # Multi-restart
    x_test = pf.getExcitingNewLocation(d2SolutionOutput, d2SolutionInput, bounds[:,0], bounds[:,1], jitter=0.2)
    print("EHVI optimisation, iteration {0}/10]".format(n+1))
    x_opt, f_opt = optimizer.optimize(np.array(x_test), f=ehvi_evaluate)
    #print("Reached [{0:0.3f}, {1:0.3f}], value {2:0.4f}".format(x_opt[0][0], x_opt[0][1], f_opt[0][0]))
    if f_opt[0][0] < ehvi_max:
        ehvi_max = f_opt[0][0]
        x_next_ehvi = x_opt[0]
        #print("New best.")
y_next_ehvi = evaluate_fitness(x_next_ehvi)

# Figure: y_norm
plt.figure(figsize=[plot_size * 1.62, plot_size])
scat1 = plt.scatter(f_norm[:,0], f_norm[:,1], c = 'k', s=plot_size*25.0,
           linewidths=1, edgecolors='k')
plt.axis('equal')
plt.plot(1., 1., marker = 'x', markersize = plot_size*2.0, color = 'k')
plt.text(0.51, 0.92, 'reference point $r$', fontsize=plot_size*2.0)
plt.xlabel('$f_1$ (normalised)', fontsize=plot_size*3.0)
plt.ylabel('$f_2$ (normalised)', fontsize=plot_size*3.0)
for tick in scat1.axes.get_xticklabels():
    tick.set_fontsize(plot_size*2.0)
for tick in scat1.axes.get_yticklabels():
    tick.set_fontsize(plot_size*2.0)

plt.savefig(os.path.join("img","figure_4_c0.svg"), facecolor=None, edgecolor=None)
    
# Figures: Surrogate models
# F1
mu_f1, stdev_f1 = models[0].predict(d2TestPoints)
d2f1_pred = mu_f1.reshape(d2X.shape)
d2f1_pred_std = stdev_f1.reshape(d2X.shape)

plt.figure(figsize=[plot_size, plot_size])
ax = plt.axes([0, 0.05, 0.9, 0.9])
contours = plt.contour(d2X, d2Y, d2f1_pred, 5, colors='black')
plt.clabel(contours, inline=True, fontsize=plot_size*1.5)
im1 = plt.imshow(d2f1_pred, extent=[0, 1.0, 0, 1.0], origin='lower',
           cmap='RdYlGn_r', alpha=0.5, vmin=min(mu_f1), vmax=max(mu_f1))
plt.scatter(d2SolutionInput[:,0], d2SolutionInput[:,1], c = f_norm[:,0], s=plot_size*25.0,
            linewidths=1, edgecolors='k', 
            cmap='RdYlGn_r', vmin=min(mu_f1), vmax=max(mu_f1))
# plt.xlabel('$x_1$', fontsize=plot_size*3.0)
plt.ylabel('$x_2$', fontsize=plot_size*3.0)
plt.title('Mean function from $f_{1_{norm}}$ surrogate', fontsize=plot_size*3.0)
cax = plt.axes([0.95, 0.05, 0.05, 0.9])
cb = plt.colorbar(mappable=im1, cax=cax)
for t in cb.ax.get_yticklabels():
    t.set_fontsize(plot_size*2.0)
for tick in im1.axes.get_xticklabels():
    tick.set_fontsize(fontsize=plot_size*2.0)
for tick in im1.axes.get_yticklabels():
    tick.set_fontsize(fontsize=plot_size*2.0)

plt.savefig(os.path.join("img","figure_4_c1a1.svg"), facecolor=None, edgecolor=None)

plt.figure(figsize=[plot_size, plot_size])
ax = plt.axes([0, 0.05, 0.9, 0.9])
contours = plt.contour(d2X, d2Y, d2f1_pred_std, 5, colors='black')
plt.clabel(contours, inline=True, fontsize=plot_size*1.5)
im1 = plt.imshow(d2f1_pred_std, extent=[0, 1.0, 0, 1.0], origin='lower',
           cmap='RdYlGn_r', alpha=0.5, vmin=0.0, vmax=max(stdev_f1))
plt.scatter(d2SolutionInput[:,0], d2SolutionInput[:,1], c ='g', s=plot_size*25.0,
            linewidths=1, edgecolors='k')
# plt.xlabel('$x_1$', fontsize=plot_size*3.0)
# plt.ylabel('$x_2$', fontsize=plot_size*3.0)
plt.title('Standard deviation from $f_{1_{norm}}$ surrogate', fontsize=plot_size*3.0)
cax = plt.axes([0.95, 0.05, 0.05, 0.9])
cb = plt.colorbar(mappable=im1, cax=cax)
for t in cb.ax.get_yticklabels():
    t.set_fontsize(plot_size*2.0)
for tick in im1.axes.get_xticklabels():
    tick.set_fontsize(fontsize=plot_size*2.0)
for tick in im1.axes.get_yticklabels():
    tick.set_fontsize(fontsize=plot_size*2.0)

plt.savefig(os.path.join("img","figure_4_c1a2.svg"), facecolor=None, edgecolor=None)

# F2
mu_f2, stdev_f2 = models[1].predict(d2TestPoints)
d2f2_pred = mu_f2.reshape(d2X.shape)
d2f2_pred_std = stdev_f2.reshape(d2X.shape)

plt.figure(figsize=[plot_size, plot_size])
ax = plt.axes([0, 0.05, 0.9, 0.9])
contours = plt.contour(d2X, d2Y, d2f2_pred, 5, colors='black')
plt.clabel(contours, inline=True, fontsize=plot_size*1.5)
im1 = plt.imshow(d2f2_pred, extent=[0, 1.0, 0, 1.0], origin='lower',
           cmap='RdYlGn_r', alpha=0.5, vmin=min(mu_f2), vmax=max(mu_f2))
plt.scatter(d2SolutionInput[:,0], d2SolutionInput[:,1], c = f_norm[:,1], s=plot_size*25.0,
            linewidths=1, edgecolors='k', 
            cmap='RdYlGn_r', vmin=min(mu_f2), vmax=max(mu_f2))
plt.xlabel('$x_1$', fontsize=plot_size*3.0)
plt.ylabel('$x_2$', fontsize=plot_size*3.0)
plt.title('Mean function from $f_{2_{norm}}$ surrogate', fontsize=plot_size*3.0)
cax = plt.axes([0.95, 0.05, 0.05, 0.9])
cb = plt.colorbar(mappable=im1, cax=cax)
for t in cb.ax.get_yticklabels():
    t.set_fontsize(plot_size*2.0)
for tick in im1.axes.get_xticklabels():
    tick.set_fontsize(fontsize=plot_size*2.0)
for tick in im1.axes.get_yticklabels():
    tick.set_fontsize(fontsize=plot_size*2.0)

plt.savefig(os.path.join("img","figure_4_c1b1.svg"), facecolor=None, edgecolor=None)

plt.figure(figsize=[plot_size, plot_size])
ax = plt.axes([0, 0.05, 0.9, 0.9])
contours = plt.contour(d2X, d2Y, d2f2_pred_std, 5, colors='black')
plt.clabel(contours, inline=True, fontsize=plot_size*1.5)
im1 = plt.imshow(d2f2_pred_std, extent=[0, 1.0, 0, 1.0], origin='lower',
           cmap='RdYlGn_r', alpha=0.5, vmin=0.0, vmax=max(stdev_f2))
plt.scatter(d2SolutionInput[:,0], d2SolutionInput[:,1], c ='g', s=plot_size*25.0,
            linewidths=1, edgecolors='k')
plt.xlabel('$x_1$', fontsize=plot_size*3.0)
# plt.ylabel('$x_2$', fontsize=plot_size*3.0)
plt.title('Standard deviation from $f_{2_{norm}}$ surrogate', fontsize=plot_size*3.0)
cax = plt.axes([0.95, 0.05, 0.05, 0.9])
cb = plt.colorbar(mappable=im1, cax=cax)
for t in cb.ax.get_yticklabels():
    t.set_fontsize(plot_size*2.0)
for tick in im1.axes.get_xticklabels():
    tick.set_fontsize(fontsize=plot_size*2.0)
for tick in im1.axes.get_yticklabels():
    tick.set_fontsize(fontsize=plot_size*2.0)

plt.savefig(os.path.join("img","figure_4_c1b2.svg"), facecolor=None, edgecolor=None)

# EHVI surface
d1EHVI = np.zeros([d2TestPoints.shape[0], 1])
for i in range(d2TestPoints.shape[0]):
    d1EHVI[i,0] = pf.calculateExpectedHypervolumeContributionMC(
        np.array([mu_f1[i], mu_f2[i]]),
        np.array([stdev_f1[i], stdev_f2[i]]),
        d2CurrentFrontNorm, 
        np.array([1., 1.]),
        1000)
d2ehvi_pred = d1EHVI.reshape(d2X.shape)

plt.figure(figsize=[plot_size, plot_size])
ax = plt.axes([0, 0.05, 0.9, 0.9])
contours = plt.contour(d2X, d2Y, d2ehvi_pred, np.linspace(0.05, 0.2, 4), colors='black')
plt.clabel(contours, inline=True, fontsize=plot_size*1.5)
im1 = plt.imshow(d2ehvi_pred, extent=[0, 1.0, 0, 1.0], origin='lower',
           cmap='RdYlGn', alpha=0.5, vmin=0.0, vmax=0.2)
plt.scatter(x_next_ehvi[0], x_next_ehvi[1], color='k', marker = 'x', 
            linewidth=5, s=plot_size*25.0)
plt.xlabel('$x_1$', fontsize=plot_size*3.0)
plt.ylabel('$x_2$', fontsize=plot_size*3.0)
cax = plt.axes([0.95, 0.05, 0.05, 0.9])
cb = plt.colorbar(mappable=im1, cax=cax)
cb.set_label(label = '$EHVI$', fontsize=plot_size*3.0)
for t in cb.ax.get_yticklabels():
    t.set_fontsize(plot_size*2.0)
for tick in im1.axes.get_xticklabels():
    tick.set_fontsize(fontsize=plot_size*2.0)
for tick in im1.axes.get_yticklabels():
    tick.set_fontsize(fontsize=plot_size*2.0)
    
plt.savefig(os.path.join("img","figure_4_c2.svg"), facecolor=None, edgecolor=None)
