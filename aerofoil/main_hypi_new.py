# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:43:24 2020

@author: yuw
"""

 # # This command is to clear all 
#from IPython import get_ipython;   
#get_ipython().magic('reset -sf')

# import GPy
import GPyOpt
#GPy.plotting.change_plotting_library('plotly')
import numpy as np
# import scipy as sp
import matplotlib; matplotlib.rcParams['figure.figsize'] = (8,6)
from matplotlib import pyplot as plt
plt.style.use('ggplot')
# from IPython.display import display

import pyDOE as doe
import paretoFrontND as pf

#from scipy.interpolate import interp1d
import time
import os
import pickle
# from deap.tools._hypervolume import hv as hv
# from initialize import initialization_run
# from shape_fit import shape_fit
from coord_sort import sort_te_le
import aerofoilSvg as afl
from run_Rfoil import Rfoil
from spar_cap_BM import spar_cap_BM_fun

## # # set designing space         
OPT_RERUN = 1
np.random.seed(10000 + OPT_RERUN)
CONTINUE_OPT = True

NUM_OBJECTIVES = 2

#class PathManager:
#    def __init__(self):
#        self.idx = -1
#    
#    def getUniqueString(self):
#        self.idx = self.idx + 1
#        return str(self.idx).zfill(5)
    
def getUniqueString(idx):
    return str(idx).zfill(5)     

#path_name_mgr = PathManager() 
 
save_path= '.' + os.sep + 'geo'

if not CONTINUE_OPT:
    import shutil
    if os.path.exists(save_path):
        shutil.rmtree(save_path)  # # # delete read only files in a folder 
    os.makedirs(save_path)
    shutil.copy('rfoil.exe', save_path) 

alpha= range(0,11)
Re=6e6
Mach=0
Ncrit=9
obj=1  # # # 0 to maximize cl , obj=1 to maximize cl/cd 

def evaluate_fitness(ind,idx):
    assert len(ind) == len(afl.aerofoil_optimisation_domain)
    # Create unique name for foil
    foil_name = "foil_" + getUniqueString(idx)   
    aerofoil = afl.convert_to_aerofoil(ind)
    # aerofoil.write_to_file(svg_save_path + "\\{}.svg".format(foil_name))
    pnts = aerofoil.get_points(301)
    coord1= np.asarray(pnts)
    coord=sort_te_le(coord1)
#    plt.plot(coord[:,0],coord[:,1],'-o')
#    p1.savefig(coord_foil_name+'.png', bbox_inches='tight')    
    coord_foil_name=foil_name+'.dat'
    np.savetxt(os.path.join(save_path, coord_foil_name), coord) 
    #Evaluate with RFoil
    polar=Rfoil(coord_foil_name,foil_name,alpha,Re,Mach,Ncrit,obj)
    #Evaluate structure
    FBM=spar_cap_BM_fun(coord) 
    
    #Get objective function values
    # cl = np.vstack(polar.get('lift'))
    # cd = np.vstack(polar.get('drag'))
    clocd = np.array(polar.get('OBJ'))
    convergence = polar.get('conver_alpha')
    
    stiff_fb=np.array(FBM)
    
    if clocd<0.001 or convergence < 0.5:
        # Assign penalty
        clocd=0.0
        stiff_fb=0.0
    
    return np.hstack((-clocd, -stiff_fb))
    
#### # initialized settings 
NUM_DIM=len(afl.aerofoil_optimisation_domain)
NUM_INIT= 8*NUM_DIM
EVALUATION_BUDGET = 1000

reference_point = [0., 0.] # Not important for the logic, just for progress reporting.

i = NUM_INIT
k = 0

### Start timer
start_time = time.perf_counter()

if not CONTINUE_OPT:
    x = np.empty((EVALUATION_BUDGET, NUM_DIM))
    y = np.empty((EVALUATION_BUDGET, NUM_OBJECTIVES))
    times = np.empty((EVALUATION_BUDGET - NUM_INIT + 1)) * -1
    
    ### evaluate the initial settings
    x[:NUM_INIT,:] = doe.lhs(NUM_DIM, samples = NUM_INIT, criterion='center')
    for j in range(NUM_DIM):
        dim_range = afl.aerofoil_optimisation_domain[j]['domain'][1] - afl.aerofoil_optimisation_domain[j]['domain'][0]
        x[:,j] = x[:,j] * dim_range + afl.aerofoil_optimisation_domain[j]['domain'][0]
    # x[:NUM_INIT,:] = initialization_run(aerofoil_optimisation_domain, NUM_DIM, NUM_INIT)
    y[:NUM_INIT,:] = np.array(list( map(evaluate_fitness, x[:NUM_INIT,:], range(NUM_INIT)) ))
    
    t_incr = time.perf_counter() - start_time
    times[k] = t_incr
    
else:
    with open('aero_fitness_struc_TH18_xhvi_intermediate.pkl', 'rb') as f:
        x,y,i,k,times = pickle.load(f)

# %% Step 3: Run optimisation

# 3.1 Normalisation and exploration/exploitation
def normalise_f(f, exploration_param):
    avg = np.mean(f)
    low = np.min(f)
    high = np.max(f)
    offset = (1 - exploration_param) * avg + exploration_param * low
    return (f - offset) / (high - low + 1e-6)

EXPLORATION_PARAM = 0.0   
    
# 3.2 Loop over acquisitions and evaluations
des_space = GPyOpt.core.task.space.Design_space(afl.aerofoil_optimisation_domain)
acq_opt = GPyOpt.optimization.AcquisitionOptimizer(des_space, optimizer='lbfgs')
NUM_BATCH = 1

while i < EVALUATION_BUDGET:
    hypervolume_achieved = pf.calculateHypervolume(y[:i,:], reference_point)
    print("{0} evaluations done, current hypervolume: {1:0.3f}".format(i, hypervolume_achieved))
    
    # Normalise solution space
    y_norm = normalise_f(np.array([(y[:i, 0])]).transpose(), 0.0)
    for m in range(1, NUM_OBJECTIVES):
        y_norm = np.hstack((y_norm, normalise_f(np.array([(y[:i, m])]).transpose(), 0.0)))

    # Calculate infill criterion
    d1HypI = pf.calculateHypIs(y_norm, np.repeat(1.0, NUM_OBJECTIVES).tolist())
    d1HypI_norm = normalise_f(d1HypI, EXPLORATION_PARAM)    
    
    # Fit a GP model to xHVI
    # N.B. Don't trust the 'maximize' parameter! 
    bo_step = GPyOpt.methods.BayesianOptimization(
        f=None, domain=afl.aerofoil_optimisation_domain, 
        X=x[:i,:], Y=-d1HypI_norm,
        normalize_Y=False, exact_feval=True,
        model_type='GP', ARD=True, maximize=False, 
        acquisition_type='EI', acquisition_optimizer_type='lbfgs', acquisition_jitter=0.0,
        batch_size=NUM_BATCH, evaluator_type='local_penalization')
    
    ### evaluate the current step 
    x[i:i+NUM_BATCH,:] = bo_step.suggest_next_locations()
    y[i:i+NUM_BATCH,:] = np.array(list( map(evaluate_fitness, x[i:i+NUM_BATCH,:], list(range(i,i+NUM_BATCH))) ))
    
    if (i) % 2 ==0:
        with open('aero_fitness_struc_TH18_xhvi_intermediate.pkl', 'wb') as f:
            pickle.dump([x,y,i,k,times], f) 
    
    i += NUM_BATCH
    
    k += 1
    t_incr = time.perf_counter() - start_time
    times[k] = t_incr

# %% Step 4: Plot results
final_front, index = pf.getNonDominatedFront(y[:i,:])
front_id=np.where(index)[0]
final_hypervolume = pf.calculateHypervolume(y[:i,:], reference_point)

for d in range(1, NUM_OBJECTIVES):
    plt.figure()

    plt.plot(-y[:i,0], -y[:i,d], 
             linestyle='', marker = '.', color = 'r', markersize=3, 
             label = 'All evaluated solutions')
    
    plt.plot(-y[:NUM_INIT,0], -y[:NUM_INIT,d], 
             linestyle='', marker = '.', color = 'b', markersize=3, 
             label = 'Initial Latin Hypercube')

    plt.plot(-final_front[:,0], -final_front[:,d], 
             linestyle = '', marker = '.', color = 'g', markersize = 4, 
             label = 'Final non-dominated set')

    plt.xlabel('$f_1$'), plt.ylabel("$f_{}$".format(d+1))
    plt.legend(loc='upper right', facecolor='w')
    plt.title("Total hypervolume: {0:4g}".format(final_hypervolume))
    plt.legend(loc='best')

with open('aero_fitness_struc_TH18_xhvi_final.pkl', 'wb') as f:
    pickle.dump(
        [x,y,afl.aerofoil_optimisation_domain,reference_point,alpha,NUM_BATCH,NUM_INIT,times], f)  
      