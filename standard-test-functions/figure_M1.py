# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 21:48:04 2020

@author: Clym Stock-Williams

Creates summary boxplots of runs
"""
import os
#import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams
import seaborn as sns; sns.set(color_codes=True)

plt.style.use('ggplot')
rcParams['font.sans-serif'] = "Segoe UI"
rcParams['font.family'] = "sans-serif"

plot_size = 10.0

# %% Setup
FUNCTION_NAME = "BBOB suite problem f20 instance 1 in 10D"
NUM_INPUT_DIMS = 10

NUM_TOTAL_EVALUATIONS = 300
NUM_SAMPLES = NUM_INPUT_DIMS * 4

FOLDER = os.path.join("Results_Detailed_M1", f"{FUNCTION_NAME}")

# %% Load file
filename = os.path.join(FOLDER, f"{FUNCTION_NAME}_summary.csv")

df = pd.read_csv(filename, sep=',', names=['xi', 'jitter', 'iteration', 'best'])

# %% Plot

fig = plt.figure(figsize=[plot_size*1.62, plot_size])
ax = sns.boxplot(y='best', x='xi',
                   data=df,
                   palette='icefire',
                   hue='jitter')
plt.title(FUNCTION_NAME)
plt.savefig(os.path.join(FOLDER, f"{FUNCTION_NAME}_summary_boxplot1.png"), dpi=400)

fig = plt.figure(figsize=[plot_size*1.62, plot_size])
ax = sns.stripplot(y='best', x='xi',
                   data=df,
                   jitter=True, dodge=True,
                   marker='o',
                   alpha=0.5,
                   palette='icefire',
                   hue='jitter')
plt.title(FUNCTION_NAME)
plt.savefig(os.path.join(FOLDER, f"{FUNCTION_NAME}_summary_stripplot1.png"), dpi=400)

fig = plt.figure(figsize=[plot_size*1.62, plot_size])
ax = sns.boxplot(y='best', x='jitter',
                   data=df,
                   palette='icefire',
                   hue='xi')
plt.title(FUNCTION_NAME)
plt.savefig(os.path.join(FOLDER, f"{FUNCTION_NAME}_summary_boxplot2.png"), dpi=400)

fig = plt.figure(figsize=[plot_size*1.62, plot_size])
ax = sns.stripplot(y='best', x='jitter',
                   data=df,
                   jitter=True, dodge=True,
                   marker='o',
                   alpha=0.5,
                   palette='icefire',
                   hue='xi')
plt.title(FUNCTION_NAME)
plt.savefig(os.path.join(FOLDER, f"{FUNCTION_NAME}_summary_stripplot2.png"), dpi=400)