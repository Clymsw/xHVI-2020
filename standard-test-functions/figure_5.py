# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 21:48:04 2020

@author: Clym Stock-Williams

Creates elements of Figs. 5 and 7 (results) from the PPSN 2020 paper.
"""
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams
import seaborn as sns; sns.set(color_codes=True)
import ParetoFrontND as pf
import StandardTestFunctions as fn

plt.style.use('ggplot')
rcParams['font.sans-serif'] = "Segoe UI"
rcParams['font.family'] = "sans-serif"

plot_size = 10.0

# %% Get function properties
FUNCTION_NAME = "DTLZ7"
NUM_INPUT_DIMS = 10
NUM_OBJECTIVES = 2

NUM_TOTAL_EVALUATIONS = 300
NUM_SAMPLES = NUM_INPUT_DIMS * 4

d2F1F2_PF = fn.get_M2_pareto_front(FUNCTION_NAME)
d1Reference = [max(d2F1F2_PF[:,0]) * 1.1, max(d2F1F2_PF[:,1]) * 1.1]
max_hypervolume = pf.calculateHypervolume(d2F1F2_PF, d1Reference)

domain, fitnessfunc, _, NUM_INPUT_DIMS, NUM_OBJECTIVES = fn.get_function_definition(
    FUNCTION_NAME, NUM_INPUT_DIMS, NUM_OBJECTIVES)

# %% Load data 
def load_data(folder: str):
    filename_hv = 'summary_hv_ref_{0:.2f}_{1:.2f}.csv'.format(d1Reference[0], d1Reference[1])
    filename_igd = 'summary_igd.csv'
    filename_f1 = 'summary_finalset_f{}.csv'.format(1)
    filename_f2 = 'summary_finalset_f{}.csv'.format(2)
    
    all_hv = np.genfromtxt(os.path.join(folder, filename_hv), delimiter=',')
    all_igd = np.genfromtxt(os.path.join(folder, filename_igd), delimiter=',')
    
    d1NumPointsInNds = np.zeros((21, 1))
    d2AllNdsPoints = []
    with open(os.path.join(folder, filename_f1), 'r') as fid_f1:
        rdr_f1 = csv.reader(fid_f1)
        with open(os.path.join(folder, filename_f2), 'r') as fid_f2:
            rdr_f2 = csv.reader(fid_f2)
            for o in range(21):
                opt_f1s = next(rdr_f1)
                opt_f2s = next(rdr_f2)
                for p in range(len(opt_f1s)):
                    f1 = float(opt_f1s[p])
                    f2 = float(opt_f2s[p])
                    if f1 != 0.0 or f2 != 0.0:
                        d1NumPointsInNds[o] += 1
                        d2AllNdsPoints.append(np.array([f1, f2]))
    d2AllNdsPoints = np.array(d2AllNdsPoints)
    
    return d1NumPointsInNds, d2AllNdsPoints, all_hv, all_igd

# %% xHVI
ZETA = 0.0
folder_xhvi = os.path.join("Results_Detailed_Timed",
                            FUNCTION_NAME + "_D{1}_M{2}_Z{0:.2f}_Rnorm".format(ZETA, NUM_INPUT_DIMS, NUM_OBJECTIVES))

d1NumPointsInNds_xhvi, d2AllNdsPoints_xhvi, all_hv_xhvi, all_igd_xhvi = load_data(folder_xhvi)

# %% HypI
ZETA = 0.0
folder_hypi = os.path.join("Results_Detailed_HypI_Timed",
                           FUNCTION_NAME + "_D{1}_M{2}_Z{0:.2f}_Rnorm".format(ZETA, NUM_INPUT_DIMS, NUM_OBJECTIVES))

d1NumPointsInNds_hypi, d2AllNdsPoints_hypi, all_hv_hypi, all_igd_hypi = load_data(folder_hypi)

# %% EHVI
folder_ehvi = os.path.join("Results_Detailed_EHVI_Timed",
                            FUNCTION_NAME + "_D{0}_norm_M{1}".format(NUM_INPUT_DIMS, NUM_OBJECTIVES))
d1NumPointsInNds_ehvi, d2AllNdsPoints_ehvi, all_hv_ehvi, all_igd_ehvi = load_data(folder_ehvi)

# %% Plot HV and IGD learning traces
# def plot_trace(data, ylab, ylim_min, ylim_max):    
#     fig = plt.figure(figsize=[plot_size*1.62, plot_size])
#     x_vals = np.arange(NUM_SAMPLES, NUM_TOTAL_EVALUATIONS + 1)
#     plt.plot(x_vals, 
#               np.median(data[:, NUM_SAMPLES-1:NUM_TOTAL_EVALUATIONS], axis=0),
#               linewidth=2, color='k', label='Median')
#     plt.plot(x_vals, 
#               np.percentile(data[:, NUM_SAMPLES-1:NUM_TOTAL_EVALUATIONS], 75, axis=0),
#               linewidth=1, color='r', label='Upper quartile')
#     plt.plot(x_vals, 
#               np.percentile(data[:, NUM_SAMPLES-1:NUM_TOTAL_EVALUATIONS], 25, axis=0),
#               linewidth=1, color='g', label='Lower quartile')
#     iWorst = np.argmax(data[:,-1])
#     plt.plot(x_vals, 
#               data[iWorst, NUM_SAMPLES-1:NUM_TOTAL_EVALUATIONS],
#               linewidth=1, color='r', linestyle='--', label='Worst')
#     # plt.plot(range(d2HypervolumeLoss.shape[0]), 
#     #          np.max(d2HypervolumeLoss, axis=1),
#     #          linewidth=1, color='r', linestyle='--', label='Worst')
#     iBest = np.argmin(data[:,-1])
#     # plt.plot(range(d2HypervolumeLoss.shape[0]), 
#     #          np.min(d2HypervolumeLoss, axis=1),
#     #          linewidth=1, color='g', linestyle='--', label='Best')
#     plt.plot(x_vals, 
#               data[iBest, NUM_SAMPLES-1:NUM_TOTAL_EVALUATIONS],
#               linewidth=1, color='g', linestyle='--', label='Best')
#     plt.ylim(ylim_min, ylim_max)
#     plt.xlim([0, NUM_TOTAL_EVALUATIONS])
#     plt.legend(loc='upper right', fontsize=plot_size*2.0, labelspacing=0.25)
#     plt.ylabel(ylab, fontsize=plot_size*3.0)
#     plt.xlabel("Number of evaluations", fontsize=plot_size*3.0)
#     for tick in fig.get_axes()[0].get_xticklabels():
#         tick.set_fontsize(plot_size*2.0)
#     for tick in fig.get_axes()[0].get_yticklabels():
#         tick.set_fontsize(plot_size*2.0)

# plot_trace(all_hv_xhvi/max_hypervolume*100, "Hypervolume loss (%)", 0, 100)
# plt.savefig(os.path.join(folder_xhvi, FUNCTION_NAME + "_xhvi_hv_progress.svg"), facecolor=None, edgecolor=None)

# plot_trace(all_hv_hypi/max_hypervolume*100, "Hypervolume loss (%)", 0, 100)
# plt.savefig(os.path.join(folder_hypi, FUNCTION_NAME + "_hypi_hv_progress.svg"), facecolor=None, edgecolor=None)

# plot_trace(all_hv_ehvi/max_hypervolume*100, "Hypervolume loss (%)", 0, 100)
# plt.savefig(os.path.join(folder_ehvi, FUNCTION_NAME + "_ehvi_hv_progress.svg"), facecolor=None, edgecolor=None)

# plot_trace(all_igd_xhvi, "Inter-Generational Distance", 0, np.quantile(all_igd_xhvi, 0.9)*1.25)
# plt.savefig(os.path.join(folder_xhvi, FUNCTION_NAME + "_xhvi_igd_progress.svg"), facecolor=None, edgecolor=None)

# plot_trace(all_igd_hypi, "Inter-Generational Distance", 0, np.quantile(all_igd_hypi, 0.9)*1.25)
# plt.savefig(os.path.join(folder_hypi, FUNCTION_NAME + "_hypi_igd_progress.svg"), facecolor=None, edgecolor=None)

# plot_trace(all_igd_ehvi, "Inter-Generational Distance", 0, np.quantile(all_igd_xhvi, 0.9)*1.25)
# plt.savefig(os.path.join(folder_ehvi, FUNCTION_NAME + "_ehvi_igd_progress.svg"), facecolor=None, edgecolor=None)

# %% Final Non-dominated set
hvloss_xhvi = all_hv_xhvi[:, NUM_TOTAL_EVALUATIONS - 1]/max_hypervolume*100
hvloss_hypi = all_hv_hypi[:, NUM_TOTAL_EVALUATIONS - 1]/max_hypervolume*100
hvloss_ehvi = all_hv_ehvi[:, NUM_TOTAL_EVALUATIONS - 1]/max_hypervolume*100

igd_xhvi = all_igd_xhvi[:, NUM_TOTAL_EVALUATIONS - 1]
igd_hypi = all_igd_hypi[:, NUM_TOTAL_EVALUATIONS - 1]
igd_ehvi = all_igd_ehvi[:, NUM_TOTAL_EVALUATIONS - 1]

# def plot_violin(name: str, left_side, left_side_name, right_side, right_side_name, old_size, ymax):
#     fig = plt.figure(figsize=[plot_size*1.62, plot_size])
#     ax = sns.violinplot(old_size * 2 * [0.0],
#                     np.hstack([left_side, right_side]).tolist(),
#                     old_size * [left_side_name] + old_size * [right_side_name],
#                     inner="quartile", split=True, cut=0, scale='area',
#                     palette = ['lightskyblue','lemonchiffon'])
#     plt.ylim([0, ymax])
#     plt.ylabel(name, fontsize=plot_size*3.0)
#     ax.set_xticks([])
#     for tick in fig.get_axes()[0].get_yticklabels():
#         tick.set_fontsize(plot_size*2.5)
#     for child in ax.get_children():
#         if type(child) is matplotlib.legend.Legend:
#             child.remove()
#             break

def plot_box(data, names, old_size, y_label, y_max):
    fig = plt.figure(figsize=[plot_size*1.62, plot_size])
    labels = old_size * [names[0]]
    for n in range(1, len(names)):
        labels += old_size * [names[n]]
    ax1 = sns.violinplot(
        old_size * len(names) * [0.0],
        data, labels,
        cut=0, scale='width', inner='quartile', orient='v',
        palette = ['lightskyblue','lemonchiffon', 'palegreen'],
        linewidth=2)
    plt.ylim([0, y_max])
    plt.ylabel(y_label, fontsize=plot_size*3.0)
    ax1.set_xticks([])
    for tick in fig.get_axes()[0].get_yticklabels():
        tick.set_fontsize(plot_size*2.5)
    for child in ax1.get_children():
        if type(child) is matplotlib.legend.Legend:
            child.remove()
            break

plot_box(np.hstack([hvloss_ehvi, hvloss_xhvi, hvloss_hypi]).tolist(),
         ["EHVI","xHVI","HypI"],
         all_hv_xhvi.shape[0],
         "Hypervolume loss (%)", 100)
plt.savefig(os.path.join('img', FUNCTION_NAME + "_hv_violin_all.svg"), facecolor=None, edgecolor=None)

# plot_violin("Hypervolume loss (%)", hvloss_xhvi, "xHVI", hvloss_ehvi, "EHVI", all_hv_xhvi.shape[0], 100)
# plt.savefig(os.path.join('img', FUNCTION_NAME + "_hv_violin_xhvi_vs_ehvi.svg"), facecolor=None, edgecolor=None)

# plot_violin("Hypervolume loss (%)", hvloss_hypi, "HypI", hvloss_ehvi, "EHVI", all_hv_xhvi.shape[0], 100)
# plt.savefig(os.path.join('img', FUNCTION_NAME + "_hv_violin_hypi_vs_ehvi.svg"), facecolor=None, edgecolor=None)

# plot_violin("Hypervolume loss (%)", hvloss_xhvi, "xHVI", hvloss_hypi, "HypI", all_hv_xhvi.shape[0], 100)
# plt.savefig(os.path.join('img', FUNCTION_NAME + "_hv_violin_xhvi_vs_hypi.svg"), facecolor=None, edgecolor=None)


ymax = max([np.max(igd_xhvi), np.max(igd_hypi), np.max(igd_ehvi)])*1.2
plot_box(np.hstack([igd_ehvi, igd_xhvi, igd_hypi]).tolist(),
         ["EHVI","xHVI","HypI"],
         igd_xhvi.shape[0],
         "Inter-Generational Distance", ymax)
plt.savefig(os.path.join('img', FUNCTION_NAME + "_igd_violin_all.svg"), facecolor=None, edgecolor=None)

# plot_violin("Inter-Generational Distance", igd_xhvi, "xHVI", igd_ehvi, "EHVI", igd_xhvi.shape[0], ymax)
# plt.savefig(os.path.join('img', FUNCTION_NAME + "_igd_violin_xhvi_vs_ehvi.svg"), facecolor=None, edgecolor=None)

# plot_violin("Inter-Generational Distance", igd_hypi, "HypI", igd_ehvi, "EHVI", igd_xhvi.shape[0], ymax)
# plt.savefig(os.path.join('img', FUNCTION_NAME + "_igd_violin_hypi_vs_ehvi.svg"), facecolor=None, edgecolor=None)

# plot_violin("Inter-Generational Distance", igd_xhvi, "xHVI", igd_hypi, "HypI", igd_xhvi.shape[0], ymax)
# plt.savefig(os.path.join('img', FUNCTION_NAME + "_igd_violin_xhvi_vs_hypi.svg"), facecolor=None, edgecolor=None)

plot_box(np.hstack([d1NumPointsInNds_ehvi[:,0], d1NumPointsInNds_xhvi[:,0], d1NumPointsInNds_hypi[:,0]]).tolist(),
         ["EHVI","xHVI","HypI"],
         len(d1NumPointsInNds_xhvi),
         "Non-Dominated Set Size", 30)
plt.savefig(os.path.join('img', FUNCTION_NAME + "_nds_size_violin_all.svg"), facecolor=None, edgecolor=None)

# plot_violin("Non-Dominated Set Size", d1NumPointsInNds_xhvi[:,0], "xHVI", d1NumPointsInNds_ehvi[:,0], "EHVI", len(d1NumPointsInNds_xhvi), 30)
# plt.savefig(os.path.join('img', FUNCTION_NAME + "_nds_size_violin_xhvi_vs_ehvi.svg"), facecolor=None, edgecolor=None)

# plot_violin("Non-Dominated Set Size", d1NumPointsInNds_hypi[:,0], "HypI", d1NumPointsInNds_ehvi[:,0], "EHVI", len(d1NumPointsInNds_xhvi), 30)
# plt.savefig(os.path.join('img', FUNCTION_NAME + "_nds_size_violin_hypi_vs_ehvi.svg"), facecolor=None, edgecolor=None)

# plot_violin("Non-Dominated Set Size", d1NumPointsInNds_xhvi[:,0], "xHVI", d1NumPointsInNds_hypi[:,0], "HypI", len(d1NumPointsInNds_xhvi), 30)
# plt.savefig(os.path.join('img', FUNCTION_NAME + "_nds_size_violin_xhvi_vs_hypi.svg"), facecolor=None, edgecolor=None)

# %% Plot Pareto Front itself
# def plot_pareto_front(data, xmax, ymax):
#     fig = plt.figure(figsize=[plot_size*1.62, plot_size])
#     plt.hexbin(data[:,0], data[:,1], 
#                 gridsize=25, edgecolors='k', cmap='Blues', mincnt=1, bins='log',
#                 extent=(min(d2F1F2_PF[:,0]), xmax, min(d2F1F2_PF[:,1]), ymax),
#                 label = 'Distribution of non-dominated sets')
#     # ax = sns.kdeplot(d2AllNdsPoints_xhvi[:,0], d2AllNdsPoints_xhvi[:,1], bw=0.1,
#     #                  cmap="Blues", shade=True, shade_lowest=False, label = 'KDE of non-dominated sets')
#     # plt.scatter(d2AllNdsPoints_xhvi[:,0], d2AllNdsPoints_xhvi[:,1], marker='.', color='b', label = 'Non-dominated sets')
#     plt.scatter(d2F1F2_PF[::10, 0], d2F1F2_PF[::10, 1], c='g', s=plot_size*25.0,
#                 marker='.', label = 'True Pareto Front')
#     plt.xlabel("$f_1$", fontsize=plot_size*3.0)
#     plt.ylabel("$f_2$", fontsize=plot_size*3.0)
#     plt.xlim([min(d2F1F2_PF[:,0]) - 0.1, xmax])
#     plt.ylim([min(d2F1F2_PF[:,1]) - 0.1, ymax])
#     # plt.legend(loc='upper right', labelspacing=0.25, fontsize=plot_size*2.0)
#     for tick in fig.get_axes()[0].get_xticklabels():
#         tick.set_fontsize(plot_size*2.5)
#     for tick in fig.get_axes()[0].get_yticklabels():
#         tick.set_fontsize(plot_size*2.5)

# xmax = max([max(d2AllNdsPoints_xhvi[:,0]), max(d2AllNdsPoints_ehvi[:,0]), max(d2AllNdsPoints_hypi[:,0])]) * 1.01 + 0.1
# ymax = max([max(d2AllNdsPoints_xhvi[:,1]), max(d2AllNdsPoints_ehvi[:,1]), max(d2AllNdsPoints_hypi[:,1])]) * 1.01 + 0.1

# plot_pareto_front(d2AllNdsPoints_xhvi, xmax, ymax)
# plt.title('xHVI', fontsize=plot_size*3.0)
# plt.savefig(os.path.join('img', FUNCTION_NAME + "_xhvi_nds_kdemap.svg"), facecolor=None, edgecolor=None)

# plot_pareto_front(d2AllNdsPoints_hypi, xmax, ymax)
# plt.title('HypI', fontsize=plot_size*3.0)
# plt.savefig(os.path.join('img', FUNCTION_NAME + "_hypi_nds_kdemap.svg"), facecolor=None, edgecolor=None)

# plot_pareto_front(d2AllNdsPoints_ehvi, xmax, ymax)
# plt.title('EHVI', fontsize=plot_size*3.0)
# plt.savefig(os.path.join('img', FUNCTION_NAME + "_ehvi_nds_kdemap.svg"), facecolor=None, edgecolor=None)

# # %% Correlations
# plt.figure()
# plt.scatter(set_size, d2HypervolumeLoss[-1,:])
# plt.ylabel("Hypervolume loss")
# plt.xlabel("Final non-dominated set size")
# plt.title("{0} ($D={1}, M={2}$) \n {3}".format(
#     FUNCTION_NAME, NUM_INPUT_DIMS, NUM_OBJECTIVES, subtitle))
# plt.yscale('log')
# plt.grid()
# plt.xlim(0,30)
# plt.xticks(range(0,31,2))
# plt.savefig(os.path.join(FOLDER, "LossVsSetSize.png"), dpi=400)
