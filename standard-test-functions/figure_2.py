# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 21:01:38 2020

@author: Clym Stock-Williams

Creates Figure 2 from the PPSN 2020 paper.
"""
import matplotlib.pyplot as plt
from matplotlib import rcParams, cycler
import numpy as np
from scipy import stats
import os
import GPy

x_train = np.array([3]) 
y_train = np.array([-1]) 

def ExpImp(mu, sigma, y_star, xi = 0):
    s = (y_star - mu - xi) / sigma
    return sigma * (s * stats.norm.cdf(s) + stats.norm.pdf(s))

# %% Kernel 
ker = GPy.kern.Matern52(1, 1, 1) + GPy.kern.White(1, 0.01)
# Mean function
meanf = GPy.core.Mapping(1,1)
meanf.f = lambda x: 0.0
meanf.update_gradients = lambda a,b: None
# Regression model with data
model = GPy.models.GPRegression(
    np.reshape(x_train,[len(x_train),1]),
    np.reshape(y_train,[len(y_train),1]),
    ker,
    mean_function = meanf,
    noise_var = 0.01)

# Prediction points
x_test = np.arange(0,6,0.1)
x_test = np.reshape(x_test,[len(x_test),1])

y_mean_test, y_variance_test = model.predict(x_test)

fig, ax = plt.subplots(2, 3)
plt.style.use('ggplot')
rcParams['font.sans-serif'] = "Segoe UI"
rcParams['font.family'] = "sans-serif"

plt.subplot(ax[0,0])
plt.plot(x_test, y_mean_test, lw=2, label='GP mean', color='blue')
plt.fill_between(x_test[:,0], 
                 y_mean_test[:,0] + np.sqrt(y_variance_test[:,0]), 
                 y_mean_test[:,0] - np.sqrt(y_variance_test[:,0]), 
                 facecolor='blue', alpha=0.5)
plt.plot(x_train, y_train, 'x', color='black', markersize=10)
plt.ylim((-2,2))
plt.xlim((0.,6.))
plt.tick_params(
    axis='both', 
    left=True,
    labelleft=True, 
    bottom=False,
    labelbottom=False)
for tick in ax[0,0].get_yticklabels():
    tick.set_fontsize(9)
plt.ylabel('Function value', fontsize=10)

y_best_pred, _ = model.predict(np.reshape(x_train, [1,1]))

plt.subplot(ax[1,0])
xi_test = np.concatenate(([0.01], np.arange(0.0, 1.1, 0.2)))
xi_test.sort()
cmap = plt.get_cmap('autumn_r')
rcParams['axes.prop_cycle'] = cycler(color=cmap(np.linspace(0, 1, len(xi_test))))
for i_xi in range(len(xi_test)):
    ei_test = ExpImp(y_mean_test, np.sqrt(y_variance_test), y_best_pred, xi_test[i_xi])
    plt.plot(x_test, ei_test, label = '$\zeta = {0: 0.2f}$'.format(xi_test[i_xi]))
    iBest = np.argmax(ei_test[(x_test < 3)]) # Find peak of LHS
    plt.plot([x_test[iBest], x_test[iBest]], [0, ei_test[iBest]], 'k', linewidth=1)
plt.legend(loc='upper center', fontsize='xx-small', labelspacing=0.25, ncol=2)
plt.ylabel('Expected improvement', fontsize=10)
plt.xlabel('Input value', fontsize=10)
plt.ylim((0., 0.35))
plt.xlim((0.,6.))
plt.tick_params(
    axis='both', 
    left=True,
    labelleft=True, 
    bottom=True,
    labelbottom=True)
for tick in ax[1,0].get_xticklabels():
    tick.set_fontsize(9)
for tick in ax[1,0].get_yticklabels():
    tick.set_fontsize(9)

# %% Lower Mean function
meanf2 = GPy.core.Mapping(1,1)
meanf2.f = lambda x: -0.5
meanf2.update_gradients = lambda a,b: None
# Regression model with data
model2 = GPy.models.GPRegression(
    np.reshape(x_train,[len(x_train),1]),
    np.reshape(y_train,[len(y_train),1]),
    ker,
    mean_function = meanf2,
    noise_var = 0.01)

y_mean_test2, y_variance_test2 = model2.predict(x_test)

plt.subplot(ax[0,1])
plt.plot(x_test, y_mean_test2, lw=2, label='GP mean', color='blue')
plt.fill_between(x_test[:,0], 
                 y_mean_test2[:,0] + np.sqrt(y_variance_test2[:,0]), 
                 y_mean_test2[:,0] - np.sqrt(y_variance_test2[:,0]), 
                 facecolor='blue', alpha=0.5)
plt.plot(x_train, y_train, 'x', color='black', markersize=10)
plt.ylim((-2,2))
plt.xlim((0.,6.))
plt.tick_params(
    axis='both', 
    left=False,
    labelleft=False, 
    bottom=False,
    labelbottom=False)

y_best_pred2, _ = model2.predict(np.reshape(x_train, [1,1]))

plt.subplot(ax[1,1])
jitter = 0.75
ei_test = ExpImp(y_mean_test, np.sqrt(y_variance_test), y_best_pred, jitter)
iBest = np.argmax(ei_test[(x_test <= 3)]) # Find peak of LHS
ei_test2 = ExpImp(y_mean_test2, np.sqrt(y_variance_test2), y_best_pred2, 0.0)
iBest2 = np.argmax(ei_test2[(x_test <= 3)]) # Find peak of LHS

plt.plot(x_test, ei_test, label = '$Mean = 0.0, \zeta = {0:.2f}$'.format(jitter), color='firebrick')
plt.plot(x_test, ei_test2, label = '$Mean = -0.5, \zeta = 0.0$', color='deepskyblue')
plt.legend(loc='upper center', labelspacing=0.25, fontsize='xx-small')
plt.plot([x_test[iBest], x_test[iBest]], [0, ei_test[iBest]], 'k', linewidth=1)
plt.plot([x_test[iBest2], x_test[iBest2]], [0, ei_test2[iBest2]], 'k', linewidth=1)
plt.xlabel('Input value', fontsize=10)
plt.ylim((0., 0.35))
plt.xlim((0.,6.))
plt.tick_params(
    axis='both', 
    left=False,
    labelleft=False, 
    bottom=True,
    labelbottom=True)
for tick in ax[1,1].get_xticklabels():
    tick.set_fontsize(9)

# %% Higher mean function
meanf2.f = lambda x: 1.0
model2 = GPy.models.GPRegression(
    np.reshape(x_train,[len(x_train),1]),
    np.reshape(y_train,[len(y_train),1]),
    ker,
    mean_function = meanf2,
    noise_var = 0.01)

y_mean_test2, y_variance_test2 = model2.predict(x_test)

plt.subplot(ax[0,2])
plt.plot(x_test, y_mean_test2, lw=2, label='GP mean', color='blue')
plt.fill_between(x_test[:,0], 
                 y_mean_test2[:,0] + np.sqrt(y_variance_test2[:,0]), 
                 y_mean_test2[:,0] - np.sqrt(y_variance_test2[:,0]), 
                 facecolor='blue', alpha=0.5)
plt.plot(x_train, y_train, 'x', color='black', markersize=10)
plt.ylim((-2,2))
plt.xlim((0.,6.))
plt.tick_params(
    axis='both', 
    left=False,
    labelleft=False, 
    bottom=False,
    labelbottom=False)

y_best_pred2, _ = model2.predict(np.reshape(x_train, [1,1]))

jitter = -0.3
ei_test = ExpImp(y_mean_test, np.sqrt(y_variance_test), y_best_pred, jitter)
iBest = np.argmax(ei_test[(x_test <= 3)]) # Find peak of LHS
ei_test2 = ExpImp(y_mean_test2, np.sqrt(y_variance_test2), y_best_pred2, 0.0)
iBest2 = np.argmax(ei_test2[(x_test <= 3)]) # Find peak of LHS

plt.subplot(ax[1,2])
plt.plot(x_test, ei_test, label = '$Mean = 0.0, \zeta = {0:.2f}$'.format(jitter), color='firebrick')
plt.plot(x_test, ei_test2, label = '$Mean = 1.0, \zeta = 0.0$', color='deepskyblue')
plt.legend(loc='center', labelspacing=0.25, fontsize='xx-small')
plt.plot([x_test[iBest], x_test[iBest]], [0, ei_test[iBest]], 'k', linewidth=1)
plt.plot([x_test[iBest2], x_test[iBest2]], [0, ei_test2[iBest2]], 'k', linewidth=1)
plt.xlabel('Input value', fontsize=10)
plt.ylim((0., 0.35))
plt.xlim((0.,6.))
plt.tick_params(
    axis='both', 
    left=False,
    labelleft=False, 
    bottom=True,
    labelbottom=True)
for tick in ax[1,2].get_xticklabels():
    tick.set_fontsize(9)
    
plt.savefig(os.path.join("img","acquisition_exploration2.svg"), facecolor=None, edgecolor=None)