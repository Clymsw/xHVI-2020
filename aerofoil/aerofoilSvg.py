# -*- coding: utf-8 -*-
"""
Created on Sun May 17 22:01:29 2020

@author: stockwilliamscfw
"""
import svgpathtools as svg
import numpy as np

class Aerofoil:
    svg_width = 100.0
    
    def __init__(self,
                 thickness_ratio,   # relative to chord c
                 y1,                # fraction of yB-yA above A
                 yB,                # fraction of t above A
                 xB,                # fraction of c right of A
                 x2,                # fraction of xB-xA left of B
                 x3,                # fraction of xC-xB right of B
                 x4,                # fraction of xC-x3 left of C
                 y4,                # fraction of yB-yC above C
                 y5,                # fraction of y4-yD below 4
                 x5,                # fraction of x4-x6 left of 4
                 xD,                # fraction of c right of A
                 x6,                # fraction of xC-xD right of D
                 x7,                # fraction of xD-xA left of D
                 y8):               # fraction of yA-yD below A
        self.thickness_ratio = thickness_ratio
        assert(all(np.array([y1, xB, x2, x3, x4, y4, x5, y5, xD, x6, x7, y8]) >= 0.0))
        assert(all(np.array([y1, xB, x2, x3, x4, y4, x5, y5, xD, x6, x7, y8]) <= 1.0))
        self.y1 = y1
        self.yB = yB
        self.xB = xB
        self.x2 = x2
        self.x3 = x3
        self.x4 = x4
        self.y4 = y4
        self.y5 = y5
        self.x5 = x5
        self.xD = xD
        self.x6 = x6
        self.x7 = x7
        self.y8 = y8
        
        self.path = self.__convert_to_svgpath()
        
    def __convert_to_svgpath(self):
        location_A = (0 + 0j)
        location_B = (self.xB + self.yB*self.thickness_ratio*1j)
        location_C = (1 + 0j)
        location_D = (self.xD + (self.yB - 1.0)*self.thickness_ratio*1j)
        
        control_1 = (0 + self.y1*location_B.imag*1j)
        control_2 = (self.x2*location_B.real + location_B.imag*1j)
        control_3 = (self.x3*(1.0-location_B.real)+location_B.real + 
                     location_B.imag*1j)
        
        control_6 = (self.x6*(1.0-location_D.real)+location_D.real + 
                     location_D.imag*1j)
        control_7 = (self.x2*location_D.real + location_D.imag*1j)
        control_8 = (0 + self.y8*location_D.imag*1j)
        
        y4 = location_C.imag + self.y4*(location_B.imag-location_C.imag)
        x4 = location_C.real - self.x4*(location_C.real - control_3.real)
        control_4 = (x4 + y4*1j)
        
        x5 = control_4.real - self.x5*(control_4.real-control_6.real)
        y5 = control_4.imag - self.y5*(control_4.imag-control_6.imag)
        control_5 = (x5 + y5*1j)
        
        return svg.Path(
            svg.CubicBezier(start=location_A,
                            control1=control_1,
                            control2=control_2,
                            end=location_B),
            svg.CubicBezier(start=location_B,
                            control1=control_3,
                            control2=control_4,
                            end=location_C),
            svg.CubicBezier(start=location_C,
                            control1=control_5,
                            control2=control_6,
                            end=location_D),
            svg.CubicBezier(start=location_D,
                            control1=control_7,
                            control2=control_8,
                            end=location_A))
    
    def get_points(self, number_of_points):
        points = []
        for i in range(number_of_points):
            loc = self.path.point(i/(float(number_of_points)-1))
            points.append([loc.real, loc.imag])
        return points
    
    def write_to_file(self, filepath):
        svg.wsvg(self.path, filename=filepath)

# %% Optimisation setup
# aerofoil_optimisation_domain = [
#     {'name': 'y_1', 'type': 'continuous', 'domain': (0.4, 0.95)}, # fraction of yB-yA above A
#     {'name': 'y_B', 'type': 'continuous', 'domain': (0.3, 0.95)}, # fraction of t above A
#     {'name': 'x_B', 'type': 'continuous', 'domain': (0.25, 0.55)}, # fraction of c right of A
#     {'name': 'x_2', 'type': 'continuous', 'domain': (0.2, 0.8)}, # fraction of xB-xA left of B
#     {'name': 'x_3', 'type': 'continuous', 'domain': (0.3, 0.75)}, # fraction of xC-xB right of B
#     {'name': 'alpha_4', 'type': 'continuous', 'domain': (20, 80)}, # angle subtended by 4, C and the horizontal
#     {'name': 'y_4', 'type': 'continuous', 'domain': (0.1, 1.0)}, # fraction of xC-xB left of C
#     {'name': 'alpha_45', 'type': 'continuous', 'domain': (15, 60)}, # angle subtended by 4, C and 5
#     {'name': 'x_5', 'type': 'continuous', 'domain': (0.02, 0.1)}, # fraction of xC-xD left of C
#     {'name': 'x_D', 'type': 'continuous', 'domain': (0.25, 0.55)}, # fraction of c right of A
#     {'name': 'x_6', 'type': 'continuous', 'domain': (0.3, 0.75)}, # fraction of xC-xD right of D
#     {'name': 'x_7', 'type': 'continuous', 'domain': (0.2, 0.8)}, # fraction of xD-xA left of D
#     {'name': 'y_8', 'type': 'continuous', 'domain': (0.4, 0.95)}] # fraction of yA-yD below A
aerofoil_optimisation_domain = [
    {'name': 'y_1', 'type': 'continuous', 'domain': (0.2, 0.9)}, # fraction of yB-yA above A
    {'name': 'y_B', 'type': 'continuous', 'domain': (0.5, 0.95)}, # fraction of t above A
    {'name': 'x_B', 'type': 'continuous', 'domain': (0.15, 0.7)}, # fraction of c right of A
    {'name': 'x_2', 'type': 'continuous', 'domain': (0.1, 0.9)}, # fraction of xB-xA left of B
    {'name': 'x_3', 'type': 'continuous', 'domain': (0.2, 0.8)}, # fraction of xC-xB right of B
    {'name': 'x_4', 'type': 'continuous', 'domain': (0.05, 0.75)}, # fraction of xC-x3 left of C
    {'name': 'y_4', 'type': 'continuous', 'domain': (0.05, 0.95)}, # fraction of yB-yC above C
    {'name': 'y_5', 'type': 'continuous', 'domain': (0.05, 0.5)}, # fraction of y4-yD below 4
    {'name': 'x_5', 'type': 'continuous', 'domain': (0.05, 0.75)}, # fraction of x4-x6 left of 4
    {'name': 'x_D', 'type': 'continuous', 'domain': (0.15, 0.7)}, # fraction of c right of A
    {'name': 'x_6', 'type': 'continuous', 'domain': (0.2, 0.8)}, # fraction of xC-xD right of D
    {'name': 'x_7', 'type': 'continuous', 'domain': (0.1, 0.9)}, # fraction of xD-xA left of D
    {'name': 'y_8', 'type': 'continuous', 'domain': (0.2, 0.9)}] # fraction of yA-yD below A

aerofoil_thickness_ratio = 0.18

def convert_to_aerofoil(ind):
    assert len(ind) == 13
    return Aerofoil(aerofoil_thickness_ratio,
                    ind[0], ind[1], ind[2], ind[3], ind[4], ind[5], ind[6],
                    ind[7], ind[8], ind[9], ind[10], ind[11], ind[12])

# import pyDOE as doe
# import numpy as np
# import matplotlib.pyplot as plt
# np.random.seed(1234)

# NUM_SAMPLES = 1000
# test_aerofoil_design = doe.lhs(len(aerofoil_optimisation_domain), samples = NUM_SAMPLES, criterion='center')
# for i in range(len(aerofoil_optimisation_domain)):
#     dim_range = aerofoil_optimisation_domain[i]['domain'][1] - aerofoil_optimisation_domain[i]['domain'][0]
#     test_aerofoil_design[:,i] = test_aerofoil_design[:,i] * dim_range + aerofoil_optimisation_domain[i]['domain'][0]

# for foil in range(NUM_SAMPLES):
#     test_aerofoil = convert_to_aerofoil(test_aerofoil_design[foil].tolist())
#     test_points = np.array(test_aerofoil.get_points(200))
#     plt.plot(test_points[:,0], test_points[:,1], linewidth=0.25, color='k')
#     # test_aerofoil.write_to_file("example_{}.svg".format(foil))

# plt.show()