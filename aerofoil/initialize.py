# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 11:58:15 2019

@author: yuw
"""


import lhsmdu
import numpy as np

def initialization_run(domain,num_DIM,num_init):
    start_points_norm = lhsmdu.sample(num_DIM,num_init)
    start_points_norm=start_points_norm.transpose()
    start_points=np.zeros([num_init,num_DIM])
    
    bounds = np.array([[domain[0].get('domain')[0], domain[0].get('domain')[1]],
                       [domain[1].get('domain')[0], domain[1].get('domain')[1]],
                       [domain[2].get('domain')[0], domain[2].get('domain')[1]],
                       [domain[3].get('domain')[0], domain[3].get('domain')[1]],
                       [domain[4].get('domain')[0], domain[4].get('domain')[1]],
                       [domain[5].get('domain')[0], domain[5].get('domain')[1]],
                       [domain[6].get('domain')[0], domain[6].get('domain')[1]],
                       [domain[7].get('domain')[0], domain[7].get('domain')[1]],
                       [domain[8].get('domain')[0], domain[8].get('domain')[1]],
                       [domain[9].get('domain')[0], domain[9].get('domain')[1]],
                       [domain[10].get('domain')[0], domain[10].get('domain')[1]],
                       [domain[11].get('domain')[0], domain[11].get('domain')[1]],   
                       [domain[12].get('domain')[0], domain[12].get('domain')[1]]                                           
                        ])
    for i in range(num_init):
        for j in range(num_DIM):
            start_points[i,j]= bounds[j,0]+ start_points_norm[i,j]*(bounds[j,1]-bounds[j,0])
            
    return start_points
            
            