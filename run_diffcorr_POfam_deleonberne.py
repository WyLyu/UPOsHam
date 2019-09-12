# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 10:02:48 2019

@author: wl16298
"""

"""Plot peridoic orbits with 5 different energies using the new method"""

# For deleonberne problem

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint,quad,trapz,solve_ivp
import math
from IPython.display import Image # for Notebook
from IPython.core.display import HTML
from mpl_toolkits.mplot3d import Axes3D
from numpy import linalg as LA
import scipy.linalg as linalg
from scipy.optimize import fsolve
import time
from functools import partial
import diffcorr_UPOsHam2dof ### import module xxx where xxx is the name of the python file xxx.py 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from matplotlib import cm
from pylab import rcParams
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
#%matplotlib
from scipy import optimize
#%% Setting up parameters and global variables
N = 4          # dimension of phase space
MASS_A = 8.0 
MASS_B = 8.0 # De Leon, Marston (1989)
EPSILON_S = 1.0
D_X = 10.0
ALPHA = 1.00
LAMBDA = 1.5
parameters = np.array([MASS_A, MASS_B, EPSILON_S, D_X, LAMBDA, ALPHA])
eqNum = 1  
model = 'deleonberne'
eqPt = diffcorr_UPOsHam2dof.get_eq_pts(eqNum,model, parameters)


eSaddle = diffcorr_UPOsHam2dof.get_total_energy(model,[eqPt[0],eqPt[1],0,0], parameters) #energy of the saddle eq pt
#If orbit is an n-array, e.g. orbit = [orbit_0, orbit_1, ..., orbit_n]
# 
#e = np.zeros(n,1)    
#for i in range(n):
#    e[i] = get_total_energy_deleonberne(orbit[i], parameters)
#e = np.mean(e)






#%% 
nFam = 100 # use nFam = 10 for low energy

# first two amplitudes for continuation procedure to get p.o. family
Ax1  = 2.e-5 # initial amplitude (1 of 2) values to use: 2.e-3
Ax2  = 2*Ax1 # initial amplitude (2 of 2)

t = time.time()

#  get the initial conditions and periods for a family of periodic orbits


po_fam_file = open("1111x0_tp_fam_eqPt%s_deleonberne.txt" %eqNum,'a+')
[po_x0Fam,po_tpFam] = diffcorr_UPOsHam2dof.get_POFam(eqNum, Ax1, Ax2,nFam, po_fam_file, parameters,model)  

poFamRuntime = time.time()-t
x0podata = np.concatenate((po_x0Fam, po_tpFam),axis=1)
po_fam_file.close()



#%%
# begins with a family of periodic orbits and steps until crossing the
# initial condition with target energy 
# fileName = 'x0po_T_energy_case1_L41.txt'
# fileName = 'x0po_T.txt'

deltaE = 0.10
po_fam_file = open("1111x0_tp_fam_eqPt%s_deleonberne.txt" %eqNum ,'a+')
eTarget = eSaddle + deltaE 
print('Loading the periodic orbit family from data file',po_fam_file.name,'\n') 
x0podata = np.loadtxt(po_fam_file.name)
po_fam_file.close()


po_brac_file = open("1111x0po_T_energyPO_eqPt%s_brac%s_deleonberne.txt" %(eqNum,deltaE),'a+')
t = time.time()
# [x0poTarget,TTarget] = bracket_POEnergy_bp(eTarget, x0podata, po_brac_file)
x0poTarget,TTarget = diffcorr_UPOsHam2dof.poBracketEnergy(eTarget, x0podata,po_brac_file, parameters,model)
poTarE_runtime = time.time()-t
model_parameters_file = open("1111model_parameters_eqPt%s_DelE%s_deleonberne.txt" %(eqNum,deltaE),'a+')
np.savetxt(model_parameters_file.name, parameters,fmt='%1.16e')
model_parameters_file.close()
po_brac_file.close()

# target specific periodic orbit
# Target PO of specific energy with high precision does not work for the
# model 

po_target_file = open("1111x0_diffcorr_deltaE%s_deleonberne.txt" %(deltaE),'a+')
                
[x0po_1, T_1,energyPO_1] = diffcorr_UPOsHam2dof.poTargetEnergy(x0poTarget,eTarget,po_target_file,parameters,model)

po_target_file.close()


#%%
# begins with a family of periodic orbits and steps until crossing the
# initial condition with target energy 
# fileName = 'x0po_T_energy_case1_L41.txt'
# fileName = 'x0po_T.txt'

deltaE = 1.0
po_fam_file = open("1111x0_tp_fam_eqPt%s_deleonberne.txt" %eqNum ,'a+')
eTarget = eSaddle + deltaE 
print('Loading the periodic orbit family from data file',po_fam_file.name,'\n') 
x0podata = np.loadtxt(po_fam_file.name)
po_fam_file.close()


po_brac_file = open("1111x0po_T_energyPO_eqPt%s_brac%s_deleonberne.txt" %(eqNum,deltaE),'a+')
t = time.time()
# [x0poTarget,TTarget] = bracket_POEnergy_bp(eTarget, x0podata, po_brac_file)
x0poTarget,TTarget = diffcorr_UPOsHam2dof.poBracketEnergy(eTarget, x0podata,po_brac_file, parameters,model)
poTarE_runtime = time.time()-t
model_parameters_file = open("1111model_parameters_eqPt%s_DelE%s_deleonberne.txt" %(eqNum,deltaE),'a+')
np.savetxt(model_parameters_file.name, parameters,fmt='%1.16e')
model_parameters_file.close()
po_brac_file.close()

# target specific periodic orbit
# Target PO of specific energy with high precision does not work for the
# model 

po_target_file = open("1111x0_diffcorr_deltaE%s_deleonberne.txt" %(deltaE),'a+')
                
[x0po_2, T_2,energyPO_2] = diffcorr_UPOsHam2dof.poTargetEnergy(x0poTarget,eTarget,po_target_file,parameters,model)

po_target_file.close()
#%%
# begins with a family of periodic orbits and steps until crossing the
# initial condition with target energy 
# fileName = 'x0po_T_energy_case1_L41.txt'
# fileName = 'x0po_T.txt'

deltaE = 2.0
po_fam_file = open("1111x0_tp_fam_eqPt%s_deleonberne.txt" %eqNum ,'a+')
eTarget = eSaddle + deltaE 
print('Loading the periodic orbit family from data file',po_fam_file.name,'\n') 
x0podata = np.loadtxt(po_fam_file.name)
po_fam_file.close()


po_brac_file = open("1111x0po_T_energyPO_eqPt%s_brac%s_deleonberne.txt" %(eqNum,deltaE),'a+')
t = time.time()
# [x0poTarget,TTarget] = bracket_POEnergy_bp(eTarget, x0podata, po_brac_file)
x0poTarget,TTarget = diffcorr_UPOsHam2dof.poBracketEnergy(eTarget, x0podata,po_brac_file, parameters,model)
poTarE_runtime = time.time()-t
model_parameters_file = open("1111model_parameters_eqPt%s_DelE%s_deleonberne.txt" %(eqNum,deltaE),'a+')
np.savetxt(model_parameters_file.name, parameters,fmt='%1.16e')
model_parameters_file.close()
po_brac_file.close()

# target specific periodic orbit
# Target PO of specific energy with high precision does not work for the
# model 

po_target_file = open("1111x0_diffcorr_deltaE%s_deleonberne.txt" %(deltaE),'a+')
                
[x0po_3, T_3,energyPO_3] = diffcorr_UPOsHam2dof.poTargetEnergy(x0poTarget,eTarget,po_target_file,parameters,model)

po_target_file.close()
#%%
# begins with a family of periodic orbits and steps until crossing the
# initial condition with target energy 
# fileName = 'x0po_T_energy_case1_L41.txt'
# fileName = 'x0po_T.txt'

deltaE = 4.0
po_fam_file = open("1111x0_tp_fam_eqPt%s_deleonberne.txt" %eqNum ,'a+')
eTarget = eSaddle + deltaE 
print('Loading the periodic orbit family from data file',po_fam_file.name,'\n') 
x0podata = np.loadtxt(po_fam_file.name)
po_fam_file.close()


po_brac_file = open("1111x0po_T_energyPO_eqPt%s_brac%s_deleonberne.txt" %(eqNum,deltaE),'a+')
t = time.time()
# [x0poTarget,TTarget] = bracket_POEnergy_bp(eTarget, x0podata, po_brac_file)
x0poTarget,TTarget = diffcorr_UPOsHam2dof.poBracketEnergy(eTarget, x0podata,po_brac_file, parameters,model)
poTarE_runtime = time.time()-t
model_parameters_file = open("1111model_parameters_eqPt%s_DelE%s_deleonberne.txt" %(eqNum,deltaE),'a+')
np.savetxt(model_parameters_file.name, parameters,fmt='%1.16e')
model_parameters_file.close()
po_brac_file.close()

# target specific periodic orbit
# Target PO of specific energy with high precision does not work for the
# model 

po_target_file = open("1111x0_diffcorr_deltaE%s_deleonberne.txt" %(deltaE),'a+')
                
[x0po_4, T_4,energyPO_4] = diffcorr_UPOsHam2dof.poTargetEnergy(x0poTarget,eTarget,po_target_file,parameters,model)

po_target_file.close()

#%% Load Data
deltaE = 0.1
po_fam_file = open("1111x0_diffcorr_deltaE%s_deleonberne.txt" %(deltaE),'a+')
print('Loading the periodic orbit family from data file',po_fam_file.name,'\n') 
x0podata = np.loadtxt(po_fam_file.name)
po_fam_file.close()
x0po_1 = x0podata[0:4]

deltaE = 1.0
po_fam_file = open("1111x0_diffcorr_deltaE%s_deleonberne.txt" %(deltaE),'a+')
print('Loading the periodic orbit family from data file',po_fam_file.name,'\n') 
x0podata = np.loadtxt(po_fam_file.name)
po_fam_file.close()
x0po_2 = x0podata[0:4]

deltaE = 2.0
po_fam_file = open("1111x0_diffcorr_deltaE%s_deleonberne.txt" %(deltaE),'a+')
print('Loading the periodic orbit family from data file',po_fam_file.name,'\n') 
x0podata = np.loadtxt(po_fam_file.name)
po_fam_file.close()
x0po_3 = x0podata[0:4]

deltaE = 4.0
po_fam_file = open("1111x0_diffcorr_deltaE%s_deleonberne.txt" %(deltaE),'a+')
print('Loading the periodic orbit family from data file',po_fam_file.name,'\n') 
x0podata = np.loadtxt(po_fam_file.name)
po_fam_file.close()
x0po_4 = x0podata[0:4]


#%% Plotting the Family
TSPAN = [0,30]
plt.close('all')
axis_fs = 15
RelTol = 3.e-10
AbsTol = 1.e-10
f= lambda t,x: diffcorr_UPOsHam2dof.Ham2dof(model,t,x,parameters)
soln = solve_ivp(f, TSPAN, x0po_1,method='RK45',dense_output=True, events = lambda t,x : diffcorr_UPOsHam2dof.half_period(t,x,model),rtol=RelTol, atol=AbsTol)
te = soln.t_events[0]
tt = [0,te[1]]
t,x,phi_t1,PHI = diffcorr_UPOsHam2dof.stateTransitMat(tt,x0po_1,parameters,model)
ax = plt.gca(projection='3d')
ax.plot(x[:,0],x[:,1],x[:,2],'-',color='g',label='$\Delta E$ = 0.1')
ax.plot(x[:,0],x[:,1],-x[:,2],'-',color='g')
ax.scatter(x[0,0],x[0,1],x[0,2],s=20,marker='*')
ax.scatter(x[0,0],x[0,1],-x[0,2],s=20,marker='o')
ax.plot(x[:,0], x[:,1], zs=0, zdir='z')

f= lambda t,x: diffcorr_UPOsHam2dof.Ham2dof(model,t,x,parameters)
soln = solve_ivp(f, TSPAN, x0po_2,method='RK45',dense_output=True, events = lambda t,x : diffcorr_UPOsHam2dof.half_period(t,x,model),rtol=RelTol, atol=AbsTol)
te = soln.t_events[0]
tt = [0,te[1]]
t,x,phi_t1,PHI = diffcorr_UPOsHam2dof.stateTransitMat(tt,x0po_2,parameters,model)
ax = plt.gca(projection='3d')
ax.plot(x[:,0],x[:,1],x[:,2],'-',color='r',label='$\Delta E$ = 1.0')
ax.plot(x[:,0],x[:,1],-x[:,2],'-',color='r')
ax.scatter(x[0,0],x[0,1],x[0,2],s=20,marker='*')
ax.scatter(x[0,0],x[0,1],-x[0,2],s=20,marker='o')
ax.plot(x[:,0], x[:,1], zs=0, zdir='z')

f= lambda t,x: diffcorr_UPOsHam2dof.Ham2dof(model,t,x,parameters)
soln = solve_ivp(f, TSPAN, x0po_3,method='RK45',dense_output=True, events = lambda t,x : diffcorr_UPOsHam2dof.half_period(t,x,model),rtol=RelTol, atol=AbsTol)
te = soln.t_events[0]
tt = [0,te[1]]
t,x,phi_t1,PHI = diffcorr_UPOsHam2dof.stateTransitMat(tt,x0po_3,parameters,model)
ax = plt.gca(projection='3d')
ax.plot(x[:,0],x[:,1],x[:,2],'-',color='b',label='$\Delta E$ = 2.0')
ax.plot(x[:,0],x[:,1],-x[:,2],'-',color='b')
ax.scatter(x[0,0],x[0,1],x[0,2],s=20,marker='*')
ax.scatter(x[0,0],x[0,1],-x[0,2],s=20,marker='o')
ax.plot(x[:,0], x[:,1], zs=0, zdir='z')


f= lambda t,x: diffcorr_UPOsHam2dof.Ham2dof(model,t,x,parameters)
soln = solve_ivp(f, TSPAN, x0po_4,method='RK45',dense_output=True, events = lambda t,x : diffcorr_UPOsHam2dof.half_period(t,x,model),rtol=RelTol, atol=AbsTol)
te = soln.t_events[0]
tt = [0,te[1]]
t,x,phi_t1,PHI = diffcorr_UPOsHam2dof.stateTransitMat(tt,x0po_4,parameters,model)
ax = plt.gca(projection='3d')
ax.plot(x[:,0],x[:,1],x[:,2],'-',color='m',label='$\Delta E$ = 4.0')
ax.plot(x[:,0],x[:,1],-x[:,2],'-',color='m')
ax.scatter(x[0,0],x[0,1],x[0,2],s=20,marker='*')
ax.scatter(x[0,0],x[0,1],-x[0,2],s=20,marker='o')
ax.plot(x[:,0], x[:,1], zs=0, zdir='z')


ax = plt.gca(projection='3d')
resX = 100
xVec = np.linspace(-1,1,resX)
yVec = np.linspace(-2,2,resX)
xMat, yMat = np.meshgrid(xVec, yVec)
cset1 = ax.contour(xMat, yMat, diffcorr_UPOsHam2dof.get_pot_surf_proj(model,xVec, yVec,parameters), [1.1,2,3,5],zdir='z', offset=0,
                       linewidths = 1.0, cmap=cm.viridis, alpha = 0.8)

ax.scatter(eqPt[0], eqPt[1], s = 200, c = 'r', marker = 'X')
ax.set_xlabel('$x$', fontsize=axis_fs)
ax.set_ylabel('$y$', fontsize=axis_fs)
ax.set_zlabel('$p_x$', fontsize=axis_fs)
#ax.set_title('$\Delta E$ = %1.e,%1.e,%1.e,%1.e,%1.e' %(energyPO_1-parameters[2],energyPO_2-parameters[2],energyPO_3-parameters[2],energyPO_4-parameters[2],energyPO_5-parameters[2]) ,fontsize=axis_fs)
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_zlim(-4, 4)
legend = ax.legend(loc='upper left')


plt.grid()
plt.show()

plt.savefig('diffcorr_POfam_deleonberne.pdf',format='pdf',bbox_inches='tight')