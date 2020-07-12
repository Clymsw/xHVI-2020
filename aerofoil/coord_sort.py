# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 10:57:40 2020

@author: yuw
"""
import numpy as np
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt

def sort_te_le(coord):
##### sort into from  te -le -te     
    delta_x=coord[1:,0]-coord[0:-1,0]
    index=np.argmax(coord[:,0])
    part1=coord[0:index,:]
    part2=coord[index:-1,:]
    end_point=np.array([[1.0,0.0]])  
    coord_sort=np.concatenate((end_point,part2,part1,end_point), axis=0)
    
##### interpolate to use the same x axis for upper and lower surfaces     
#    index_list=np.where(np.sign(coord_sort[:-1,1]) != np.sign(coord_sort[1:,1]))[0]+1
    idx= np.flatnonzero((coord_sort == [0.0,0.0]).all(1))
    index1=int(idx)
    lo=coord_sort[0:index1+1,:]
    up=coord_sort[:index1-1:-1,:]      
    

    
    ang_lo = np.linspace(np.pi,0, num=150, endpoint=True)
    x_lo = 0.5*np.cos(ang_lo)+0.5
    f = interp1d(lo[:,0], lo[:,1])  
    y_lo=f(x_lo)

    
    ang_up = np.linspace(0, np.pi, num=150, endpoint=True)    
    x_up = 0.5*np.cos(ang_up)+0.5 
    f1 = interp1d(up[:,0], up[:,1])  
    y_up=f1(x_up)     
    

    coord_up=np.vstack((x_up,y_up)).T
    coord_lo=np.vstack((x_lo,y_lo)).T
    
    coord_new=np.concatenate((coord_up[:-1],coord_lo), axis=0) 

#    p1=plt.figure(figsize=(10,10))
#    plt.plot(lo[:,0],lo[:,1],'-o',x_lo,y_lo,'*')     
#    p2=plt.figure(figsize=(10,10))    
#    plt.plot(up[:,0],up[:,1],'-o',x_up,y_up,'*')  
     
    return coord_new







#print(first_neg(x))
            
        
        
        

#def sort_te_le(coord):
#    delta_x=coord[1:,0]-coord[0:-1,0]
#    index=first_neg(delta_x)
#    upper=coord[0:index,:]
#    lower=coord[index+1:,:]
#    coord_sort=[upper,lower]
#    return coord_sort        
    
#def first_neg(list):
#    count = 0
#    for number in list:
#        count += 1      #moved it outside of the if
#        if number < 0:
#            return count