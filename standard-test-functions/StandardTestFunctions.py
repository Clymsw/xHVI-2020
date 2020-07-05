# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 11:56:23 2020

@author: Clym Stock-Williams
"""
from deap import benchmarks
import numpy as np
import os

def get_function_definition(function_name: str, num_input_dims: int, num_objectives: int):
    domain = []
    fitnessfunc = None
    d1x_opt = []
    
    if function_name == "ZDT1":
        fitnessfunc = lambda ind: benchmarks.zdt1(ind)
        num_objectives = 2
        for i in range(num_input_dims):
            domain.append({'name': "x_{}]".format(i+1), 'type': 'continuous', 'domain': (0.0, 1.0)})
        d1x_opt = np.repeat(0.0, num_input_dims - 1).reshape([1,-1])
        d1x_opt = np.repeat(d1x_opt, 1000, 0)
        d1x_opt = np.hstack((np.linspace(0.0, 1.0, 1000).reshape([-1,1]), d1x_opt))
        
    elif function_name == "ZDT2":
        fitnessfunc = lambda ind: benchmarks.zdt2(ind)
        num_objectives = 2
        for i in range(num_input_dims):
            domain.append({'name': "x_{}]".format(i+1), 'type': 'continuous', 'domain': (0.0, 1.0)})
        d1x_opt = np.repeat(0.0, num_input_dims - 1).reshape([1,-1])
        d1x_opt = np.repeat(d1x_opt, 1000, 0)
        d1x_opt = np.hstack((np.linspace(0.0, 1.0, 1000).reshape([-1,1]), d1x_opt))
        
    elif function_name == "ZDT3":
        fitnessfunc = lambda ind: benchmarks.zdt3(ind)
        num_objectives = 2
        for i in range(num_input_dims):
            domain.append({'name': "x_{}]".format(i+1), 'type': 'continuous', 'domain': (0.0, 1.0)})
        d1x_opt = np.repeat(0.0, num_input_dims - 1).reshape([1,-1])
        d1x_opt = np.repeat(d1x_opt, 1000, 0)
        d1x_opt = np.hstack((np.linspace(0.0, 1.0, 1000).reshape([-1,1]), d1x_opt))
        
    elif function_name == "ZDT4":
        fitnessfunc = lambda ind: benchmarks.zdt4(ind)
        num_objectives = 2
        domain.append({'name': "x_{}]".format(1), 'type': 'continuous', 'domain': (0.0, 1.0)})
        for i in range(1, num_input_dims):
            domain.append({'name': "x_{}]".format(i+1), 'type': 'continuous', 'domain': (-5.0, 5.0)})
        d1x_opt = np.repeat(0.0, num_input_dims - 1).reshape([1,-1])
        d1x_opt = np.repeat(d1x_opt, 1000, 0)
        d1x_opt = np.hstack((np.linspace(0.0, 1.0, 1000).reshape([-1,1]), d1x_opt))
        
    elif function_name == "ZDT6":
        fitnessfunc = lambda ind: benchmarks.zdt6(ind)
        num_objectives = 2
        for i in range(num_input_dims):
            domain.append({'name': "x_{}]".format(i+1), 'type': 'continuous', 'domain': (0.0, 1.0)})
        d1x_opt = np.repeat(0.0, num_input_dims - 1).reshape([1,-1])
        d1x_opt = np.repeat(d1x_opt, 1000, 0)
        d1x_opt = np.hstack((np.linspace(0.0, 1.0, 1000).reshape([-1,1]), d1x_opt))
        
    elif function_name == "KURSAWE":
        fitnessfunc = lambda ind: benchmarks.kursawe(ind)
        
    elif function_name == "POLONI":
        fitnessfunc = lambda ind: benchmarks.kursawe(ind)
        
    elif function_name == "DTLZ1":
        fitnessfunc = lambda ind: benchmarks.dtlz1(ind, num_objectives)
        for i in range(num_input_dims):
            domain.append({'name': "x_{}]".format(i+1), 'type': 'continuous', 'domain': (0.0, 1.0)})
        d1x_opt = np.repeat(0.5, num_input_dims - 1).reshape([1,-1])
        d1x_opt = np.repeat(d1x_opt, 1000, 0)
        d1x_opt = np.hstack((np.linspace(0.0, 1.0, 1000).reshape([-1,1]), d1x_opt))
        
    elif function_name == "DTLZ2":
        fitnessfunc = lambda ind: benchmarks.dtlz2(ind, num_objectives)
        for i in range(num_input_dims):
            domain.append({'name': "x_{}]".format(i+1), 'type': 'continuous', 'domain': (0.0, 1.0)})
        d1x_opt = np.repeat(0.5, num_input_dims - 1).reshape([1,-1])
        d1x_opt = np.repeat(d1x_opt, 1000, 0)
        d1x_opt = np.hstack((np.linspace(0.0, 1.0, 1000).reshape([-1,1]), d1x_opt))
        
    elif function_name == "DTLZ3":
        fitnessfunc = lambda ind: benchmarks.dtlz3(ind, num_objectives)
        for i in range(num_input_dims):
            domain.append({'name': "x_{}]".format(i+1), 'type': 'continuous', 'domain': (0.0, 1.0)})
        d1x_opt = np.repeat(0.5, num_input_dims - 1).reshape([1,-1])
        d1x_opt = np.repeat(d1x_opt, 1000, 0)
        d1x_opt = np.hstack((np.linspace(0.0, 1.0, 1000).reshape([-1,1]), d1x_opt))
            
    elif function_name == "DTLZ4":
        fitnessfunc = lambda ind: benchmarks.dtlz4(ind, num_objectives, 100)
        for i in range(num_input_dims):
            domain.append({'name': "x_{}]".format(i+1), 'type': 'continuous', 'domain': (0.0, 1.0)})
        d1x_opt = np.repeat(0.5, num_input_dims - 1).reshape([1,-1])
        d1x_opt = np.repeat(d1x_opt, 1000, 0)
        d1x_opt = np.hstack((np.linspace(0.0, 1.0, 1000).reshape([-1,1]), d1x_opt))
        
    elif function_name == "DTLZ5":
        fitnessfunc = lambda ind: benchmarks.dtlz5(ind, num_objectives)
        for i in range(num_input_dims):
            domain.append({'name': "x_{}]".format(i+1), 'type': 'continuous', 'domain': (0.0, 1.0)})
        d1x_opt = np.repeat(0.5, num_input_dims - 1).reshape([1,-1])
        d1x_opt = np.repeat(d1x_opt, 1000, 0)
        d1x_opt = np.hstack((np.linspace(0.0, 1.0, 1000).reshape([-1,1]), d1x_opt))
    
    elif function_name == "DTLZ6":
        fitnessfunc = lambda ind: benchmarks.dtlz6(ind, num_objectives)
        for i in range(num_input_dims):
            domain.append({'name': "x_{}]".format(i+1), 'type': 'continuous', 'domain': (0.0, 1.0)})
        d1x_opt = np.repeat(0.5, num_input_dims - 1).reshape([1,-1])
        d1x_opt = np.repeat(d1x_opt, 1000, 0)
        d1x_opt = np.hstack((np.linspace(0.0, 1.0, 1000).reshape([-1,1]), d1x_opt))
    
    elif function_name == "DTLZ7":
        fitnessfunc = lambda ind: benchmarks.dtlz7(ind, num_objectives)
        for i in range(num_input_dims):
            domain.append({'name': "x_{}]".format(i+1), 'type': 'continuous', 'domain': (0.0, 1.0)})
        d1x_opt = np.repeat(0.0, num_input_dims - 1).reshape([1,-1])
        d1x_opt = np.repeat(d1x_opt, 1000, 0)
        d1x_opt = np.hstack((np.linspace(0.0, 1.0, 1000).reshape([-1,1]), d1x_opt))
        
    return domain, fitnessfunc, d1x_opt, num_input_dims, num_objectives

def get_M2_pareto_front(function_name: str):
    folder_path = 'Pareto_Fronts_M2'
    file_name = 'PF_' + function_name + '.csv'
    return np.genfromtxt(os.path.join(folder_path, file_name),
                         delimiter=',', skip_header=0)
        