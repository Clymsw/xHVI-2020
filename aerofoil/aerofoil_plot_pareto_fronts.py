# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 17:28:39 2020

@author: stockwilliamscfw
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
import plotFunctions as plots
import pickle
import numpy as np
# from aerofoilSvg import Aerofoil
import paretoFrontND as pf

plt.style.use('ggplot')
rcParams['font.sans-serif'] = "Segoe UI"
rcParams['font.family'] = "sans-serif"
plot_size = 10.0

num_lhs = 13*8
max_iterations = 800

# commercial_foils_to_plot = [0,5]
commercial_foils_to_plot = np.arange(0,13)

# filePathOut = '.\\opt{}_results.pkl'.format(RUN)

x_hypi = []
y_hypi = []
t_hypi = []

x_xhvi = []
y_xhvi = []
t_xhvi = []

x_ehvi = []
y_ehvi = []
t_ehvi = []
for run in range(5):
    filePath = '.\\HypI\\Run{}\\aero_fitness_struc_TH18_xhvi_final.pkl'.format(run+1)
    with open(filePath, 'rb') as f:
        temp_x, temp_y,_,_,_,_,_,temp_t = pickle.load(f)
    x_hypi.append(temp_x[num_lhs:max_iterations,:])
    y_hypi.append(temp_y[num_lhs:max_iterations,:])
    t_hypi.append(temp_t[:max_iterations-num_lhs+1])
    
    filePath = '.\\xHVI\\Run{}\\aero_fitness_struc_TH18_xhvi_final.pkl'.format(run+1)
    with open(filePath, 'rb') as f:
        temp_x, temp_y,_,_,_,_,_,temp_t = pickle.load(f)
    x_xhvi.append(temp_x[num_lhs:max_iterations,:])
    y_xhvi.append(temp_y[num_lhs:max_iterations,:])
    t_xhvi.append(temp_t[:max_iterations-num_lhs+1])
    
    filePath = '.\\EHVI\\sim-{}\\aero_fitness_struc_TH18_xhvi_final.pkl'.format(run+6)
    with open(filePath, 'rb') as f:
        temp_x, temp_y,_,_,_,_,temp_t = pickle.load(f)
    x_ehvi.append(temp_x[num_lhs:max_iterations,:])
    y_ehvi.append(temp_y[num_lhs:max_iterations,:])
    t_ehvi.append(temp_t[:max_iterations-num_lhs+1])
    
filePathRef = 'existingfoils_TH18.pkl'
with open(filePathRef, 'rb') as f:
    ref_names,y_ref = pickle.load(f)

# fig = plt.figure(figsize=[plot_size * 1.62, plot_size * 1.0], tight_layout=True)

# plot1 = plt.plot(-ehvi_y_toplot[ehvi_y_toplot[:,0]<0,0], -ehvi_y_toplot[ehvi_y_toplot[:,1]<0,1], 
#                  '^k')
# plt.plot(-hypi_y_toplot[hypi_y_toplot[:,0]<0,0], -hypi_y_toplot[hypi_y_toplot[:,1]<0,1], 
#          'Hg')
# plt.plot(-xhvi_y_toplot[xhvi_y_toplot[:,0]<0,0], -xhvi_y_toplot[xhvi_y_toplot[:,0]<0,1], 
#          'vb')
# plt.plot(y_ref[commercial_foils_to_plot,0], y_ref[commercial_foils_to_plot,1], 
#          'or')
# plt.xlabel('Integrated lift:drag ratio', fontsize=plot_size*3.0)
# plt.ylabel('Flapwise bending stiffness ($Nm^2$)', fontsize=plot_size*3.0)
# plt.legend(["EHVI Pareto Front".format(max_iterations),
#             "HypI Pareto Front".format(max_iterations),
#             "xHVI Pareto Front".format(max_iterations),
#             'Reference aerofoils'],
#            loc='center left', fontsize=plot_size*2.5)
# [hypi_y, xhvi_y, ehvi_y]
# plt.xlim(0, 2250)
# plt.ylim(0, 8e-4)

# for tick in plot1[0].axes.get_xticklabels():
#     tick.set_fontsize(plot_size*2.5)
# for tick in plot1[0].axes.get_yticklabels():
#     tick.set_fontsize(plot_size*2.5)

# texts = []
# for i in range(len(commercial_foils_to_plot)):
#     name = ref_names[commercial_foils_to_plot[i]][:-4]
#     texts.append(
#         plt.text(y_ref[commercial_foils_to_plot[i],0],# + 20, 
#                  y_ref[commercial_foils_to_plot[i],1],# - 0.000005, 
#                  name, fontsize=plot_size*2))
    
# adjust_text(texts, arrowprops=dict(arrowstyle='-', color='red'))

# plt.savefig("aerofoil_pareto_front_R{0}_N{1}.eps".format(RUN, max_iterations), facecolor=None, edgecolor=None)
# plt.savefig("aerofoil_pareto_front_R{0}_N{1}.png".format(RUN, max_iterations), facecolor=None, edgecolor=None)

# best_aerofoil_clcd_index_ehvi = np.argmin(ehvi_y[:,0])
# best_aerofoil_clcd_index_ehvi = np.argmin(ehvi_y[:,0])
# best_aerofoil_clcd_index_ehvi = np.argmin(ehvi_y[:,0])

# aerofoil_thickness_ratio = 0.18

# def convert_to_aerofoil(ind):
#     return Aerofoil(aerofoil_thickness_ratio,
#                     ind[0], ind[1], ind[2], ind[3], ind[4], ind[5], ind[6],
#                     ind[7], ind[8], ind[9], ind[10], ind[11], ind[12])

# best_aerofoil_clcd_ehvi = convert_to_aerofoil(ehvi_x[best_aerofoil_clcd_index_ehvi,:])
# best_aerofoil_clcd_ehvi.write_to_file('aerofoil_ehvi_{}.svg'.format(best_aerofoil_clcd_index_ehvi))

# with open(filePathOut, 'wb') as f:
#     pickle.dump([hypi_y, xhvi_y, ehvi_y], f)

d1ClCd = np.linspace(0., 2250., 1000)
d1Stiffness = np.linspace(0., 8e-4, 1000)
[d2ClCd, d2Stiffness] = np.meshgrid(d1ClCd, d1Stiffness)

print('Calculating HypI EAF')
d2ClCd_hypi, d2Stiffness_hypi, d2Eaf_hypi = pf.calculateEmpiricalAttainmentFunction(
    y_hypi, np.array([[-2250., 0.], [-1e-3, 0.]]))
print('Calculating xHVI EAF')
d2ClCd_xhvi, d2Stiffness_xhvi, d2Eaf_xhvi = pf.calculateEmpiricalAttainmentFunction(
    y_xhvi, np.array([[-2250., 0.], [-1e-3, 0.]]))
print('Calculating EHVI EAF')
d2ClCd_ehvi, d2Stiffness_ehvi, d2Eaf_ehvi = pf.calculateEmpiricalAttainmentFunction(
    y_ehvi, np.array([[-2250., 0.], [-1e-3, 0.]]))

# %% Plot EAFs
print('Plotting...')
fig = plt.figure(figsize=[plot_size * 1., plot_size * 1.62], tight_layout=True)
gs = gridspec.GridSpec(ncols=1, nrows=3, figure=fig, height_ratios=[1,1,1], hspace=0.25, wspace=0.2)

names = []
for i in range(len(commercial_foils_to_plot)):
    names.append(ref_names[commercial_foils_to_plot[i]][:-4])

ax = fig.add_subplot(gs[0, 0]) 
names_to_plot = ['' for x in range(len(names))]
names_to_plot[::3] = names[::3]
plots.plot_contour_with_points(-d2ClCd_hypi, -d2Stiffness_hypi, d2Eaf_hypi,
                               [0,0.5,0.999], 'black', ['-','--','-'],
                               y_ref[commercial_foils_to_plot,0], y_ref[commercial_foils_to_plot,1],
                               names_to_plot,
                               sYLabel='Flapwise bending\n stiffness ($Nm^2$)',
                               sTitle='HypI')

ax = fig.add_subplot(gs[1, 0])
names_to_plot = ['' for x in range(len(names))]
names_to_plot[1::3] = names[1::3]
plots.plot_contour_with_points(-d2ClCd_xhvi, -d2Stiffness_xhvi, d2Eaf_xhvi,
                               [0,0.5,0.999], 'black', ['-','--','-'],
                               y_ref[commercial_foils_to_plot,0], y_ref[commercial_foils_to_plot,1],
                               names_to_plot,
                               sYLabel='Flapwise bending\n stiffness ($Nm^2$)',
                               sTitle='xHVI')

ax = fig.add_subplot(gs[2, 0])
names_to_plot = ['' for x in range(len(names))]
names_to_plot[2::3] = names[2::3]
plots.plot_contour_with_points(-d2ClCd_ehvi, -d2Stiffness_ehvi, d2Eaf_ehvi,
                               [0,0.5,0.999], 'black', ['-','--','-'],
                               y_ref[commercial_foils_to_plot,0], y_ref[commercial_foils_to_plot,1],
                               names_to_plot,
                               sXLabel='Integrated lift:drag ratio',
                               sYLabel='Flapwise bending\n stiffness ($Nm^2$)',
                               sTitle='EHVI')

plt.savefig("aerofoil_eafs_N{0}.eps".format(max_iterations), facecolor=None, edgecolor=None)
plt.savefig("aerofoil_eafs_N{0}.png".format(max_iterations), facecolor=None, edgecolor=None)
plt.savefig("aerofoil_eafs_N{0}.svg".format(max_iterations), facecolor=None, edgecolor=None)

# %% Plot EAF difference from EHVI
fig = plt.figure(figsize=[plot_size * 1., plot_size * 1.62], tight_layout=True)
gs = gridspec.GridSpec(ncols=1, nrows=2, figure=fig, height_ratios=[1,1], hspace=0.25, wspace=0.2)

masked_data = (d2Eaf_ehvi - d2Eaf_hypi)*100.
masked_data = np.ma.masked_where((masked_data < 10.) * (masked_data > -10.), masked_data)
ax = fig.add_subplot(gs[0, 0])
names_to_plot = ['' for x in range(len(names))]
names_to_plot[::2] = names[::2]
cb = plots.plot_contours_with_points(-d2ClCd_ehvi, -d2Stiffness_ehvi, masked_data,
                    np.arange(-100, 101, 20), 'RdYlGn_r',
                    -d2ClCd_hypi, -d2Stiffness_hypi, d2Eaf_hypi,
                    [0,0.5,0.999], 'black', ['-','--','-'],
                    y_ref[commercial_foils_to_plot,0], y_ref[commercial_foils_to_plot,1],
                    names_to_plot,
                    sXLabel='Integrated lift:drag ratio',
                    sYLabel='Flapwise bending\n stiffness ($Nm^2$)',
                    sTitle='HypI',
                    sColorbarLabel='Empirical Attainment Front Difference\n (compared with EHVI)')

masked_data = (d2Eaf_ehvi - d2Eaf_xhvi)*100.
masked_data = np.ma.masked_where((masked_data < 10.) * (masked_data > -10.), masked_data)
ax = fig.add_subplot(gs[1, 0])
names_to_plot = ['' for x in range(len(names))]
names_to_plot[1::2] = names[1::2]
plots.plot_contours_with_points(-d2ClCd_ehvi, -d2Stiffness_ehvi, masked_data,
                    np.arange(-100, 101, 20), 'RdYlGn_r',
                    -d2ClCd_xhvi, -d2Stiffness_xhvi, d2Eaf_xhvi,
                    [0,0.5,0.999], 'black', ['-','--','-'],
                    y_ref[commercial_foils_to_plot,0], y_ref[commercial_foils_to_plot,1],
                    names_to_plot,
                    sXLabel='Integrated lift:drag ratio',
                    sYLabel='Flapwise bending\n stiffness ($Nm^2$)',
                    sTitle='xHVI',
                    sColorbarLabel='Empirical Attainment Front Difference\n (compared with EHVI)')

plt.savefig("aerofoil_eaf_diffs_N{0}.eps".format(max_iterations), facecolor=None, edgecolor=None)
plt.savefig("aerofoil_eaf_diffs_N{0}.png".format(max_iterations), facecolor=None, edgecolor=None)
plt.savefig("aerofoil_eaf_diffs_N{0}.svg".format(max_iterations), facecolor=None, edgecolor=None)
