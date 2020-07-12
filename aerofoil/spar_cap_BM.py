# -*- coding: utf-8 -*-
"""
Created on Sun Feb 02 21:52:00 2020

@author: yuw
"""
import parsec_12
import numpy as np
from matplotlib import pyplot as plt


def  flapwise_bending_moment(coordinates_out,coordinates_inn,N1,N2): 
    for i in range(N1,N2):
        if i==N1: 
            area_lower=np.abs(coordinates_out[i+1,0]*(coordinates_inn[i,1]-coordinates_inn[i+1,1])+coordinates_inn[i,0]*(coordinates_inn[i+1,1]-coordinates_out[i+1,1])+coordinates_inn[i+1,0]*( coordinates_out[i+1,1]- coordinates_inn[i,1]))/2
            area_upper=np.abs(coordinates_inn[i,0]*(coordinates_out[i+1,1]-coordinates_out[i,1])+coordinates_out[i+1,0]*(coordinates_out[i,1]-coordinates_inn[i,1])+coordinates_out[i,0]*( coordinates_inn[i,1]- coordinates_out[i+1,1]))/2
            y_mid=(coordinates_out[i,1]+coordinates_out[i+1,1]+coordinates_inn[i,1]+coordinates_inn[i+1,1])/4
        else: 
            area_lower=np.vstack((area_lower,np.abs(coordinates_out[i+1,0]*(coordinates_inn[i,1]-coordinates_inn[i+1,1])+coordinates_inn[i,0]*(coordinates_inn[i+1,1]-coordinates_out[i+1,1])+coordinates_inn[i+1,0]*( coordinates_out[i+1,1]- coordinates_inn[i,1]))/2))
            area_upper=np.vstack((area_upper,np.abs(coordinates_inn[i,0]*(coordinates_out[i+1,1]-coordinates_out[i,1])+coordinates_out[i+1,0]*(coordinates_out[i,1]-coordinates_inn[i,1])+coordinates_out[i,0]*( coordinates_inn[i,1]- coordinates_out[i+1,1]))/2))
            y_mid=np.vstack((y_mid,(coordinates_out[i,1]+coordinates_out[i+1,1]+coordinates_inn[i,1]+coordinates_inn[i+1,1])/4))
        BM=(area_lower+area_upper)*y_mid**2
    return BM

def spar_cap_BM_fun(coordinates): 
    coefficients = parsec_12.coordinates_to_coefficients_experimental(coordinates)
    coordinates_out = parsec_12.coefficients_to_coordinates_full_cosine(coefficients,1001)       
#### find the location of maxiumal thickness 
    idx= np.flatnonzero((coordinates == [0.0,0.0]).all(1))
    index=int(idx)   
    up=coordinates[0:index+1,:]
    lo=coordinates[:index-1:-1,:]    
    dis=up[:,1]-lo[:,1]  
    
    index_max = np.argmax(abs(dis))
    x_u=up[index_max,0]     ### x loc of max thickness 
    x1=x_u-0.25/2.0
    x2=x_u+0.25/2.0
    loc_spar=parsec_12.artificial_bound(coefficients,1001,0.9,x1,x2)       
    coordinates_inn = loc_spar.get('coord')
    N1=loc_spar.get('N1')
    N2=loc_spar.get('N2')
    N3=loc_spar.get('N3')
    N4=loc_spar.get('N4')
                                                  
    # plt.figure(figsize=(10,5))
    # plt.plot(coordinates_out[:,0],coordinates_out[:,1],'b-o',label='PARSEC')
    # plt.plot(coordinates_inn[N1:N2,0],coordinates_inn[N1:N2,1],'r-<',label='PARSEC')
    # plt.plot(coordinates_inn[N3:N4,0],coordinates_inn[N3:N4,1],'r-<',label='PARSEC')
    # plt.grid(True)

    BM1=np.sum(flapwise_bending_moment(coordinates_out,coordinates_inn,N1,N2))
    BM2=np.sum(flapwise_bending_moment(coordinates_out,coordinates_inn,N3,N4))


    BM_total=BM1+BM2 
    return BM_total