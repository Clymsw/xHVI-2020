# # # to calculate the shape fitness of the airfoil, give each a score accodring to the badness of each criteria

import itertools
import numpy as np
def shape_fit(coord,thickness_TE):
    index_list=np.where(np.sign(coord[:-1,1]) != np.sign(coord[1:,1]))[0]+1
    index=index_list[1]
    lo=coord[0:index+1,:]
    up=coord[:index-1:-1,:]  
    
    
    
    dis=up[:,1]-lo[:,1]
    # # # This is to avoid the intersecting surface 
    if min(dis[3:-10])<thickness_TE:
        th=1.0+min(dis[3:-10])-thickness_TE
    else:
        th=1.0
        
    # # # # this is to avoid the wavy surface  
    dup=(up[1:,1]-up[0:-1,1])/(up[1:,0]-up[0:-1,0])
    sign_up=len(list(itertools.groupby(dup, lambda dup: dup > 0)))
   
    dlo=(lo[1:,1]-lo[0:-1,1])/(lo[1:,0]-lo[0:-1,0])
    sign_lo=len(list(itertools.groupby(dlo, lambda dlo: dlo > 0)))        

#    if sign_up>3 or sign_lo>3:
#        wavy=0
#    else:
#        wavy=1
   # # # It's better to quantify these values 
    if sign_up>3:
        wavy_up = 0.5- (sign_up-3)/10.0
    else:
        wavy_up = 0.5
        
    if sign_lo>3:
        wavy_lo = 0.5- (sign_lo-3)/10.0
    else:
        wavy_lo = 0.5
        
    wavy=wavy_up+wavy_lo
    
     
#    return th*wavy
    return {'wavy':wavy, 'th':th}