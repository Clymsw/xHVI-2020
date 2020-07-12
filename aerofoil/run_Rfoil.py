import numpy as np
import os
import shutil




def Rfoil(coord_foil_name,foil_name,alpha,Re,Mach,Ncrit,obj): 
#  
## for function test 
##alpha=7
##Re=1e6
##Mach=0
##Ncrit=9
##savepath='geo'

  

    os.chdir('geo')   
    with open('rfoil.inp','w') as f:
        f.write('load \n')
    #    f.write('airfoil.dat \n')
        f.write('%s\n'%coord_foil_name)
        f.write('airfoil\n')
        f.write('ppar\n')
        f.write('N \n')
        f.write('300 \n\n')
        f.write('isav \n')
        f.write('isav.dat \n\n\n')
        f.write('oper\n')
        f.write('visc\n')
        f.write('%e\n'%Re)
        f.write('Mach\n')
        f.write('%e\n'%Mach)
        if Ncrit == 0:
            f.write('vpar\n')
            f.write('XTR\n')
            f.write('0.05 \n')
            f.write('0.05 \n\n')
        else:
            f.write('vpar\n')
            f.write('N\n')
            f.write('%g\n\n'%Ncrit)
        f.write('pacc\n')
        f.write('polar.dat \n')
        f.write('\n')
        f.write('arbl\n')
        f.write('aseq\n')
        f.write('%g\n'%np.min(alpha))
        f.write('%g\n'%np.max(alpha))
        f.write('1 \n')
        
        #tf = strcmp(version,'RFOIL.v4.0');
        #if tf == 1
        #    fprintf(fid,'y \n');
        #end
        f.write('svbl\n');
        f.write('1 \n');
        f.write('bl \n');
        f.write('pacc\n');
        f.write('\n');
        f.write('\nquit\n');
        f.close();    
    
    os.system("rfoil.exe < rfoil.inp")
    
    # # # CALCULATE THE cl/cd 
    alpha_out=np.zeros(np.size(alpha), dtype=np.float)
    cl=np.zeros(np.size(alpha), dtype=np.float)
    cd=np.zeros(np.size(alpha), dtype=np.float)
    polar_file = open('polar.dat', "r")
    data = [row for row in polar_file]
    for  j in range(0,len(data)-13):   # # # j is the number of AOAs 
         str_list = list(filter(None, data[j+13].split(' '))) 
         alpha_out[j]=float(str_list[0])
         cl[j]=float(str_list[1])
         cd[j]=float(str_list[2])
    polar_file.close()
    clOcd=cl/cd
    if np.count_nonzero(alpha_out)==0 :
        f2=(np.count_nonzero(alpha_out)+1.0)*1.0/len(alpha)        
    else:       
        f2=(np.count_nonzero(alpha_out))*1.0/len(alpha)
    
    if obj==0:  # # max cl 
        if len(cl)>1:
            area=(cl[0:-1]+cl[1:])*(alpha_out[1:]-alpha_out[0:-1])/2 # # # for target max(cl) 
            area = area[~np.isnan(area)]  # # # remove nan
            target=np.array([np.sum(area)])
        else:
            target=np.array([cl])    
    else:
        if len(cl)>1:
            area=(clOcd[0:-1]+clOcd[1:])*(alpha_out[1:]-alpha_out[0:-1])/2 # # # for target max(cl/cd)
            area = area[~np.isnan(area)]  # # # remove nan
            target=np.array([np.sum(area)])
        else:
            target=np.array([clOcd])
        
        
        
     # # # save the aerodynamic data into sub_folder 
    files=os.listdir('.') 
    if not os.path.exists(foil_name):
        os.makedirs(foil_name) 
        
    shutil.move(coord_foil_name,foil_name)  
    shutil.move('rfoil.inp',foil_name) 
    shutil.move('polar.dat',foil_name) 
    shutil.move('isav.dat',foil_name)  
    shutil.move('bl.cfx',foil_name)  
    shutil.move('bl.cpx',foil_name)  
    shutil.move('bl.dst',foil_name)  
    shutil.move('bl.tet',foil_name)     
    #for f in files:
    #    if not f.endswith("exe") and os.path.isfile(os.path.join(os.getcwd()+f)):            
    #        shutil.move(f,foil_name)
    os.chdir("..")
    return {'lift':cl, 'drag':cd,'OBJ':target,'conver_alpha':f2}
    

    
    