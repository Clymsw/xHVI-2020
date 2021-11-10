# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 08:12:00 2021

@author: Clym Stock-Williams

Creates contour plots of EI behaviour
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams
import seaborn as sns; sns.set(color_codes=True)

plt.style.use('ggplot')
rcParams['font.sans-serif'] = "Segoe UI"
rcParams['font.family'] = "sans-serif"

plot_size = 10.0

def normalise_f(f, f_min, f_max, f_avg, exploration_param):
    offset = (1 - exploration_param) * f_avg + exploration_param * f_min
    return (f - offset) / (f_max - f_min + 1e-6)

def get_quantiles(acquisition_par, fmin, m, s):
    '''
    Quantiles of the Gaussian distribution useful to determine the acquisition function values
    :param acquisition_par: parameter of the acquisition function
    :param fmin: current minimum.
    :param m: vector of means.
    :param s: vector of standard deviations.
    '''
    if isinstance(s, np.ndarray):
        s[s<1e-10] = 1e-10
    elif s<1e-10:
        s = 1e-10
    u = (fmin - m - acquisition_par)/s # This is how it is defined in GPyOpt (negative)
    phi = np.exp(-0.5 * u**2) / np.sqrt(2*np.pi)
    Phi = 0.5 * erfc(-u / np.sqrt(2))
    return (phi, Phi, u)

def expected_improvement(mean, std_dev, f_min, f_max, f_avg, jitter_param, exploration_param):
    m_new = normalise_f(mean, f_min, f_max, f_avg, exploration_param)
    f_min_new = normalise_f(f_min, f_min, f_max, f_avg, exploration_param)
    phi, Phi, u = get_quantiles(-jitter_param, f_min_new, m_new, std_dev) # Negative jitter to compensate
    return s * (u * Phi + phi)

# Create contour plot

