# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 17:28:39 2020

@author: stockwilliamscfw
"""

import matplotlib.pyplot as plt
from matplotlib import rcParams
import pickle
import numpy as np
from adjustText import adjust_text
from aerofoilSvg import Aerofoil
import paretoFrontND as pf

plt.style.use('ggplot')
rcParams['font.sans-serif'] = "Segoe UI"
rcParams['font.family'] = "sans-serif"
plot_size = 10.0

num_lhs = 13*8
max_iterations = 1000

# commercial_foils_to_plot = [0,5]
commercial_foils_to_plot = np.arange(0,13)

filePathOut = '.\\opt_results\\opt1_results.pkl'

filePathHypi = '.\\opt_results\\aero_fitness_struc_TH18_hypi_final_old.pkl'
# filePathHypiOld = 'C:\\Users\\stockwilliamscfw\\Documents\\Python Scripts\\xhvi-paper\\opt_results\\aero_fitness_struc_TH18_hypi_final_old.pkl'
# data = open(filePathHypiOld).read().replace('\r\n', '\n') # read and replace file contents
# open(filePathHypi, "w").write(data) 
with open(filePathHypi, 'rb') as f:
    hypi_x,hypi_y,hypi_aerofoil_optimisation_domain,hypi_reference_point,hypi_final_front,hypi_front_id,hypi_final_hypervolume,hypi_alpha,hypi_i,hypi_NUM_BATCH = pickle.load(f, encoding='latin1')
hypi_y_toplot = hypi_y[num_lhs:max_iterations,:]
# hypi_y_toplot,_ = pf.getNonDominatedFront(hypi_y[num_lhs:max_iterations,:])

filePathXhvi = '.\\opt_results\\aero_fitness_struc_TH18_xhvi_final.pkl'
with open(filePathXhvi, 'rb') as f:
   xhvi_x,xhvi_y,xhvi_aerofoil_optimisation_domain,xhvi_reference_point,xhvi_final_front,xhvi_front_id,xhvi_final_hypervolume,xhvi_alpha,xhvi_i,xhvi_NUM_BATCH = pickle.load(f)
xhvi_y_toplot = xhvi_y[num_lhs:max_iterations,:]
# xhvi_y_toplot,_ = pf.getNonDominatedFront(xhvi_y[num_lhs:max_iterations,:])

filePathEhvi = '.\\opt_results\\aero_fitness_struc_TH18_ehvi_final.pkl'
with open(filePathEhvi, 'rb') as f:
   ehvi_x,ehvi_y,ehvi_aerofoil_optimisation_domain,ehvi_reference_point,ehvi_final_front,ehvi_front_id,ehvi_final_hypervolume,ehvi_alpha,ehvi_i,ehvi_NUM_BATCH = pickle.load(f)
ehvi_y_toplot = ehvi_y[num_lhs:max_iterations,:]
# ehvi_y_toplot,_ = pf.getNonDominatedFront(ehvi_y[num_lhs:max_iterations,:])

filePathRef = 'existingfoils_TH18.pkl'
with open(filePathRef, 'rb') as f:
   ref_names,y_ref = pickle.load(f)

fig = plt.figure(figsize=[plot_size * 1.62, plot_size * 1.0], tight_layout=True)

plot1 = plt.plot(-ehvi_y_toplot[ehvi_y_toplot[:,0]<0,0], -ehvi_y_toplot[ehvi_y_toplot[:,1]<0,1], 
                 '^k')
plt.plot(-hypi_y_toplot[hypi_y_toplot[:,0]<0,0], -hypi_y_toplot[hypi_y_toplot[:,1]<0,1], 
         'Hg')
plt.plot(-xhvi_y_toplot[xhvi_y_toplot[:,0]<0,0], -xhvi_y_toplot[xhvi_y_toplot[:,0]<0,1], 
         'vb')
plt.plot(y_ref[commercial_foils_to_plot,0], y_ref[commercial_foils_to_plot,1], 
         'or')
plt.xlabel('Integrated lift:drag ratio', fontsize=plot_size*3.0)
plt.ylabel('Flapwise bending stiffness ($Nm^2$)', fontsize=plot_size*3.0)
plt.legend(["EHVI Pareto Front".format(max_iterations),
            "HypI Pareto Front".format(max_iterations),
            "xHVI Pareto Front".format(max_iterations),
            'Reference aerofoils'],
           loc='center left', fontsize=plot_size*2.5)
[hypi_y, xhvi_y, ehvi_y]
plt.xlim(0, 2000)
plt.ylim(0, 3e-4)

for tick in plot1[0].axes.get_xticklabels():
    tick.set_fontsize(plot_size*2.5)
for tick in plot1[0].axes.get_yticklabels():
    tick.set_fontsize(plot_size*2.5)

texts = []
for i in range(len(commercial_foils_to_plot)):
    name = ref_names[commercial_foils_to_plot[i]][:-4]
    texts.append(
        plt.text(y_ref[commercial_foils_to_plot[i],0],# + 20, 
                 y_ref[commercial_foils_to_plot[i],1],# - 0.000005, 
                 name, fontsize=plot_size*2))
    
adjust_text(texts, arrowprops=dict(arrowstyle='-', color='red'))

plt.savefig("aerofoil_pareto_front_{}.eps".format(max_iterations), facecolor=None, edgecolor=None)
plt.savefig("aerofoil_pareto_front_{}.png".format(max_iterations), facecolor=None, edgecolor=None)

best_aerofoil_clcd_index_ehvi = np.argmin(ehvi_y[:,0])
best_aerofoil_clcd_index_ehvi = np.argmin(ehvi_y[:,0])
best_aerofoil_clcd_index_ehvi = np.argmin(ehvi_y[:,0])

aerofoil_thickness_ratio = 0.18

def convert_to_aerofoil(ind):
    return Aerofoil(aerofoil_thickness_ratio,
                    ind[0], ind[1], ind[2], ind[3], ind[4], ind[5], ind[6],
                    ind[7], ind[8], ind[9], ind[10], ind[11], ind[12])

best_aerofoil_clcd_ehvi = convert_to_aerofoil(ehvi_x[best_aerofoil_clcd_index_ehvi,:])
best_aerofoil_clcd_ehvi.write_to_file('aerofoil_ehvi_{}.svg'.format(best_aerofoil_clcd_index_ehvi))

with open(filePathOut, 'wb') as f:
    pickle.dump([hypi_y, xhvi_y, ehvi_y], f)