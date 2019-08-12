# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 10:02:48 2019

@author: wl16298
"""

"""Plot peridoic orbits with 5 different energies using differential correction"""

# For the coupled problem

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
import coupled_diffcorr ### import module xxx where xxx is the name of the python file xxx.py 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from matplotlib import cm
from pylab import rcParams
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
#%matplotlib

#%% Setting up parameters and global variables
N = 4          # dimension of phase space
omega=1.00
EPSILON_S = 0.0 #Energy of the saddle
alpha = 1.00
beta = 1.00
epsilon= 1e-1
parameters = np.array([1,omega, EPSILON_S, alpha, beta,omega,epsilon]);
eqNum = 1 
eqPt = coupled_diffcorr.get_eq_pts_coupled(eqNum, parameters)



eSaddle = coupled_diffcorr.get_total_energy_coupled([eqPt[0],eqPt[1],0,0], parameters) #energy of the saddle eq pt
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


po_fam_file = open("1111x0_tp_fam_eqPt%s_coupled.txt" %eqNum,'a+')
[po_x0Fam,po_tpFam] = coupled_diffcorr.get_POFam_coupled(eqNum, Ax1, Ax2,nFam, po_fam_file, parameters) ; 

poFamRuntime = time.time()-t
x0podata = np.concatenate((po_x0Fam, po_tpFam),axis=1)
po_fam_file.close()

#%%
# begins with a family of periodic orbits and steps until crossing the
# initial condition with target energy 
# fileName = 'x0po_T_energy_case1_L41.txt';
# fileName = 'x0po_T.txt';

deltaE = 0.01

po_fam_file = open("1111x0_tp_fam_eqPt%s_coupled.txt" %eqNum ,'a+');
eTarget = eSaddle + deltaE; 
print('Loading the periodic orbit family from data file',po_fam_file.name,'\n'); 
x0podata = np.loadtxt(po_fam_file.name)
po_fam_file.close()


#%
po_brac_file = open("1111x0po_T_energyPO_eqPt%s_brac%s_coupled.txt" %(eqNum,deltaE),'a+');
t = time.time()
# [x0poTarget,TTarget] = bracket_POEnergy_bp(eTarget, x0podata, po_brac_file);
x0poTarget,TTarget = coupled_diffcorr.poBracketEnergy_coupled(eTarget, x0podata,po_brac_file, parameters);
poTarE_runtime = time.time()-t
model_parameters_file = open("1111model_parameters_eqPt%s_DelE%s_coupled.txt" %(eqNum,deltaE),'a+')
np.savetxt(model_parameters_file.name, parameters,fmt='%1.16e');
model_parameters_file.close()
po_brac_file.close()


#%
# target specific periodic orbit
# Target PO of specific energy with high precision; does not work for the
# model 

po_target_file = open("1111x0_diffcorr_deltaE%s_coupled.txt" %(deltaE),'a+')
                
[x0po_1, T_1,energyPO_1] = coupled_diffcorr.poTargetEnergy_coupled(x0poTarget,eTarget,po_target_file,parameters);

po_target_file.close()

#%%
# begins with a family of periodic orbits and steps until crossing the
# initial condition with target energy 
# fileName = 'x0po_T_energy_case1_L41.txt';
# fileName = 'x0po_T.txt';

deltaE = 0.1

po_fam_file = open("1111x0_tp_fam_eqPt%s_coupled.txt" %eqNum ,'a+');
eTarget = eSaddle + deltaE; 
print('Loading the periodic orbit family from data file',po_fam_file.name,'\n'); 
x0podata = np.loadtxt(po_fam_file.name)
po_fam_file.close()


#%
po_brac_file = open("1111x0po_T_energyPO_eqPt%s_brac%s_coupled.txt" %(eqNum,deltaE),'a+');
t = time.time()
# [x0poTarget,TTarget] = bracket_POEnergy_bp(eTarget, x0podata, po_brac_file);
x0poTarget,TTarget = coupled_diffcorr.poBracketEnergy_coupled(eTarget, x0podata,po_brac_file, parameters);
poTarE_runtime = time.time()-t
model_parameters_file = open("1111model_parameters_eqPt%s_DelE%s_coupled.txt" %(eqNum,deltaE),'a+')
np.savetxt(model_parameters_file.name, parameters,fmt='%1.16e');
model_parameters_file.close()
po_brac_file.close()


#%
# target specific periodic orbit
# Target PO of specific energy with high precision; does not work for the
# model 

po_target_file = open("1111x0_diffcorr_deltaE%s_coupled.txt" %(deltaE),'a+')
                
[x0po_2, T_2,energyPO_2] = coupled_diffcorr.poTargetEnergy_coupled(x0poTarget,eTarget,po_target_file,parameters);

po_target_file.close()
 
#%%
# begins with a family of periodic orbits and steps until crossing the
# initial condition with target energy 
# fileName = 'x0po_T_energy_case1_L41.txt';
# fileName = 'x0po_T.txt';

deltaE = 1.0

po_fam_file = open("1111x0_tp_fam_eqPt%s_coupled.txt" %eqNum ,'a+');
eTarget = eSaddle + deltaE; 
print('Loading the periodic orbit family from data file',po_fam_file.name,'\n'); 
x0podata = np.loadtxt(po_fam_file.name)
po_fam_file.close()


#%
po_brac_file = open("1111x0po_T_energyPO_eqPt%s_brac%s_coupled.txt" %(eqNum,deltaE),'a+');
t = time.time()
# [x0poTarget,TTarget] = bracket_POEnergy_bp(eTarget, x0podata, po_brac_file);
x0poTarget,TTarget = coupled_diffcorr.poBracketEnergy_coupled(eTarget, x0podata,po_brac_file, parameters);
poTarE_runtime = time.time()-t
model_parameters_file = open("1111model_parameters_eqPt%s_DelE%s_coupled.txt" %(eqNum,deltaE),'a+')
np.savetxt(model_parameters_file.name, parameters,fmt='%1.16e');
model_parameters_file.close()
po_brac_file.close()


#%
# target specific periodic orbit
# Target PO of specific energy with high precision; does not work for the
# model 

po_target_file = open("1111x0_diffcorr_deltaE%s_coupled.txt" %(deltaE),'a+')
                
[x0po_3, T_3,energyPO_3] = coupled_diffcorr.poTargetEnergy_coupled(x0poTarget,eTarget,po_target_file,parameters);

po_target_file.close()

#%%
# begins with a family of periodic orbits and steps until crossing the
# initial condition with target energy 
# fileName = 'x0po_T_energy_case1_L41.txt';
# fileName = 'x0po_T.txt';

deltaE = 2.00

po_fam_file = open("1111x0_tp_fam_eqPt%s_coupled.txt" %eqNum ,'a+');
eTarget = eSaddle + deltaE; 
print('Loading the periodic orbit family from data file',po_fam_file.name,'\n'); 
x0podata = np.loadtxt(po_fam_file.name)
po_fam_file.close()


#%
po_brac_file = open("1111x0po_T_energyPO_eqPt%s_brac%s_coupled.txt" %(eqNum,deltaE),'a+');
t = time.time()
# [x0poTarget,TTarget] = bracket_POEnergy_bp(eTarget, x0podata, po_brac_file);
x0poTarget,TTarget = coupled_diffcorr.poBracketEnergy_coupled(eTarget, x0podata,po_brac_file, parameters);
poTarE_runtime = time.time()-t
model_parameters_file = open("1111model_parameters_eqPt%s_DelE%s_coupled.txt" %(eqNum,deltaE),'a+')
np.savetxt(model_parameters_file.name, parameters,fmt='%1.16e');
model_parameters_file.close()
po_brac_file.close()


#%
# target specific periodic orbit
# Target PO of specific energy with high precision; does not work for the
# model 

po_target_file = open("1111x0_diffcorr_deltaE%s_coupled.txt" %(deltaE),'a+')
                
[x0po_4, T_4,energyPO_4] = coupled_diffcorr.poTargetEnergy_coupled(x0poTarget,eTarget,po_target_file,parameters);

po_target_file.close()

#%%
# begins with a family of periodic orbits and steps until crossing the
# initial condition with target energy 
# fileName = 'x0po_T_energy_case1_L41.txt';
# fileName = 'x0po_T.txt';

deltaE = 4.0

po_fam_file = open("1111x0_tp_fam_eqPt%s_coupled.txt" %eqNum ,'a+');
eTarget = eSaddle + deltaE; 
print('Loading the periodic orbit family from data file',po_fam_file.name,'\n'); 
x0podata = np.loadtxt(po_fam_file.name)
po_fam_file.close()


#%
po_brac_file = open("1111x0po_T_energyPO_eqPt%s_brac%s_coupled.txt" %(eqNum,deltaE),'a+');
t = time.time()
# [x0poTarget,TTarget] = bracket_POEnergy_bp(eTarget, x0podata, po_brac_file);
x0poTarget,TTarget = coupled_diffcorr.poBracketEnergy_coupled(eTarget, x0podata,po_brac_file, parameters);
poTarE_runtime = time.time()-t
model_parameters_file = open("1111model_parameters_eqPt%s_DelE%s_coupled.txt" %(eqNum,deltaE),'a+')
np.savetxt(model_parameters_file.name, parameters,fmt='%1.16e');
model_parameters_file.close()
po_brac_file.close()


#%
# target specific periodic orbit
# Target PO of specific energy with high precision; does not work for the
# model 

po_target_file = open("1111x0_diffcorr_deltaE%s_coupled.txt" %(deltaE),'a+')
                
[x0po_5, T_5,energyPO_5] = coupled_diffcorr.poTargetEnergy_coupled(x0poTarget,eTarget,po_target_file,parameters);

po_target_file.close()


#%% Load Data
deltaE = 0.010
po_fam_file = open("1111x0_diffcorr_deltaE%s_coupled.txt" %(deltaE),'a+');
print('Loading the periodic orbit family from data file',po_fam_file.name,'\n'); 
x0podata = np.loadtxt(po_fam_file.name)
po_fam_file.close()
x0po_1 = x0podata[0:4]

deltaE = 0.10
po_fam_file = open("1111x0_diffcorr_deltaE%s_coupled.txt" %(deltaE),'a+');
print('Loading the periodic orbit family from data file',po_fam_file.name,'\n'); 
x0podata = np.loadtxt(po_fam_file.name)
po_fam_file.close()
x0po_2 = x0podata[0:4]

deltaE = 1.0
po_fam_file = open("1111x0_diffcorr_deltaE%s_coupled.txt" %(deltaE),'a+');
print('Loading the periodic orbit family from data file',po_fam_file.name,'\n'); 
x0podata = np.loadtxt(po_fam_file.name)
po_fam_file.close()
x0po_3 = x0podata[0:4]

deltaE = 2.0
po_fam_file = open("1111x0_diffcorr_deltaE%s_coupled.txt" %(deltaE),'a+');
print('Loading the periodic orbit family from data file',po_fam_file.name,'\n'); 
x0podata = np.loadtxt(po_fam_file.name)
po_fam_file.close()
x0po_4 = x0podata[0:4]

deltaE = 4.0
po_fam_file = open("1111x0_diffcorr_deltaE%s_coupled.txt" %(deltaE),'a+');
print('Loading the periodic orbit family from data file',po_fam_file.name,'\n'); 
x0podata = np.loadtxt(po_fam_file.name)
po_fam_file.close()
x0po_5 = x0podata[0:4]
#%% Plotting the Family
TSPAN = [0,30]
plt.close('all')
axis_fs = 15
RelTol = 3.e-10
AbsTol = 1.e-10
f = partial(coupled_diffcorr.coupled2dof, par=parameters) 
soln = solve_ivp(f, TSPAN, x0po_1,method='RK45',dense_output=True, events = coupled_diffcorr.half_period,rtol=RelTol, atol=AbsTol)
te = soln.t_events[0]
tt = [0,te[1]]
t,x,phi_t1,PHI = coupled_diffcorr.stateTransitMat_coupled(tt,x0po_1,parameters)
ax = plt.gca(projection='3d')
ax.plot(x[:,0],x[:,1],x[:,3],'-',color='b',label='$\Delta E$ = 0.01')
ax.plot(x[:,0],x[:,1],-x[:,3],'-',color='b')
ax.scatter(x[0,0],x[0,1],x[0,3],s=20,marker='*');
ax.scatter(x[0,0],x[0,1],-x[0,3],s=20,marker='o');
ax.plot(x[:,0], x[:,1], zs=0, zdir='z')

f = partial(coupled_diffcorr.coupled2dof, par=parameters) 
soln = solve_ivp(f, TSPAN, x0po_2,method='RK45',dense_output=True, events = coupled_diffcorr.half_period,rtol=RelTol, atol=AbsTol)
te = soln.t_events[0]
tt = [0,te[1]]
t,x,phi_t1,PHI = coupled_diffcorr.stateTransitMat_coupled(tt,x0po_2,parameters)
ax = plt.gca(projection='3d')
ax.plot(x[:,0],x[:,1],x[:,3],'-',color='r',label='$\Delta E$ = 0.1')
ax.plot(x[:,0],x[:,1],-x[:,3],'-',color='r')
ax.scatter(x[0,0],x[0,1],x[0,3],s=20,marker='*');
ax.scatter(x[0,0],x[0,1],-x[0,3],s=20,marker='o');
ax.plot(x[:,0], x[:,1], zs=0, zdir='z')

f = partial(coupled_diffcorr.coupled2dof, par=parameters) 
soln = solve_ivp(f, TSPAN, x0po_3,method='RK45',dense_output=True, events = coupled_diffcorr.half_period,rtol=RelTol, atol=AbsTol)
te = soln.t_events[0]
tt = [0,te[1]]
t,x,phi_t1,PHI = coupled_diffcorr.stateTransitMat_coupled(tt,x0po_3,parameters)
ax = plt.gca(projection='3d')
ax.plot(x[:,0],x[:,1],x[:,3],'-',color='g',label='$\Delta E$ = 1.0')
ax.plot(x[:,0],x[:,1],-x[:,3],'-',color='g')
ax.scatter(x[0,0],x[0,1],x[0,3],s=20,marker='*');
ax.scatter(x[0,0],x[0,1],-x[0,3],s=20,marker='o');
ax.plot(x[:,0], x[:,1], zs=0, zdir='z')


f = partial(coupled_diffcorr.coupled2dof, par=parameters) 
soln = solve_ivp(f, TSPAN, x0po_4,method='RK45',dense_output=True, events = coupled_diffcorr.half_period,rtol=RelTol, atol=AbsTol)
te = soln.t_events[0]
tt = [0,te[1]]
t,x,phi_t1,PHI = coupled_diffcorr.stateTransitMat_coupled(tt,x0po_4,parameters)
ax = plt.gca(projection='3d')
ax.plot(x[:,0],x[:,1],x[:,3],'-',color='m',label='$\Delta E$ = 2.0')
ax.plot(x[:,0],x[:,1],-x[:,3],'-',color='m')
ax.scatter(x[0,0],x[0,1],x[0,3],s=20,marker='*');
ax.scatter(x[0,0],x[0,1],-x[0,3],s=20,marker='o');
ax.plot(x[:,0], x[:,1], zs=0, zdir='z')


f = partial(coupled_diffcorr.coupled2dof, par=parameters) 
soln = solve_ivp(f, TSPAN, x0po_5,method='RK45',dense_output=True, events = coupled_diffcorr.half_period,rtol=RelTol, atol=AbsTol)
te = soln.t_events[0]
tt = [0,te[1]]
t,x,phi_t1,PHI = coupled_diffcorr.stateTransitMat_coupled(tt,x0po_5,parameters)
ax = plt.gca(projection='3d')
ax.plot(x[:,0],x[:,1],x[:,3],'-',color='c',label='$\Delta E$ = 4.0')
ax.plot(x[:,0],x[:,1],-x[:,3],'-',color='c')
ax.scatter(x[0,0],x[0,1],x[0,3],s=20,marker='*');
ax.scatter(x[0,0],x[0,1],-x[0,3],s=20,marker='o');
ax.plot(x[:,0], x[:,1], zs=0, zdir='z')


ax = plt.gca(projection='3d')
resX = 100
xVec = np.linspace(-4,4,resX)
yVec = np.linspace(-4,4,resX)
xMat, yMat = np.meshgrid(xVec, yVec)
cset1 = ax.contour(xMat, yMat, coupled_diffcorr.get_pot_surf_proj(xVec, yVec,parameters), [0.01,0.1,1,2,4],zdir='z', offset=0,
                       linewidths = 1.0, cmap=cm.viridis, alpha = 0.8)

ax.scatter(eqPt[0], eqPt[1], s = 200, c = 'r', marker = 'X')
ax.set_xlabel('$x$', fontsize=axis_fs)
ax.set_ylabel('$y$', fontsize=axis_fs)
ax.set_zlabel('$p_y$', fontsize=axis_fs)
#ax.set_title('$\Delta E$ = %1.e,%1.e,%1.e,%1.e,%1.e' %(energyPO_1,energyPO_2,energyPO_3,energyPO_4,energyPO_5) ,fontsize=axis_fs)

legend = ax.legend(loc='upper left')
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_zlim(-4, 4)

plt.grid()
plt.show()

plt.savefig('diffcorr_POfam_coupled.pdf',format='pdf',bbox_inches='tight')