# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 16:53:45 2019

@author: wl16298
"""

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
import turningpoint_UPOsHam2dof ### import module xxx where xxx is the name of the python file xxx.py 
import diffcorr_UPOsHam2dof


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
eqPt = turningpoint_UPOsHam2dof.get_eq_pts(eqNum, model,parameters)



eSaddle = turningpoint_UPOsHam2dof.get_total_energy(model,[eqPt[0],eqPt[1],0,0], parameters) #energy of the saddle eq pt
#If orbit is an n-array, e.g. orbit = [orbit_0, orbit_1, ..., orbit_n]
#%% Load Data
deltaE = 1.0
eSaddle = 1.0 # energy of the saddle

data_path = "./data/"
#po_fam_file = open("1111x0_tpcd_deltaE%s_deleonberne.txt" %(deltaE),'a+')
po_fam_file = "1111x0_tpcd_deltaE%s_deleonberne.txt" %(deltaE)
print('Loading the periodic orbit family from data file',po_fam_file,'\n') 
#x0podata = np.loadtxt(po_fam_file.name)
#po_fam_file.close()

x0podata = np.loadtxt(data_path + po_fam_file)

x0po_1_tpcd = x0podata


#po_fam_file = open("1111x0_turningpoint_deltaE%s_deleonberne.txt" %(deltaE),'a+')
po_fam_file = "1111x0_turningpoint_deltaE%s_deleonberne.txt" %(deltaE)
print('Loading the periodic orbit family from data file',po_fam_file,'\n') 
#x0podata = np.loadtxt(po_fam_file.name)
#po_fam_file.close()
x0podata = np.loadtxt(data_path + po_fam_file)

x0po_1_turningpoint = x0podata


#po_fam_file = open("1111x0_diffcorr_deltaE%s_deleonberne.txt" %(deltaE),'a+')
po_fam_file = "1111x0_diffcorr_deltaE%s_deleonberne.txt" %(deltaE)

print('Loading the periodic orbit family from data file',po_fam_file,'\n') 
#x0podata = np.loadtxt(po_fam_file.name)
#po_fam_file.close()
x0podata = np.loadtxt(data_path + po_fam_file)

x0po_1_diffcorr = x0podata[0:4]

#%%
TSPAN = [0,30]
plt.close('all')
axis_fs = 15
RelTol = 3.e-10
AbsTol = 1.e-10
f = lambda t,x : turningpoint_UPOsHam2dof.Ham2dof(model,t,x,parameters)  
soln = solve_ivp(f, TSPAN, x0po_1_tpcd[-1,0:4],method='RK45',dense_output=True, events = lambda t,x : turningpoint_UPOsHam2dof.half_period(t,x,model),rtol=RelTol, atol=AbsTol)
te = soln.t_events[0]
tt = [0,te[2]]
t,x,phi_t1,PHI = turningpoint_UPOsHam2dof.stateTransitMat(tt,x0po_1_tpcd[-1,0:4],parameters,model)

ax = plt.gca(projection='3d')
ax.plot(x[:,0],x[:,1],x[:,2],':',label='$\Delta E$ = 0.1, using tpcd')
ax.scatter(x[0,0],x[0,1],x[0,2],s=20,marker='*')
ax.plot(x[:,0], x[:,1], zs=0, zdir='z')

f = lambda t,x : turningpoint_UPOsHam2dof.Ham2dof(model,t,x,parameters)  
soln = solve_ivp(f, TSPAN, x0po_1_diffcorr,method='RK45',dense_output=True, events = lambda t,x : turningpoint_UPOsHam2dof.half_period(t,x,model),rtol=RelTol, atol=AbsTol)
te = soln.t_events[0]
tt = [0,te[2]]
t,x,phi_t1,PHI = turningpoint_UPOsHam2dof.stateTransitMat(tt,x0po_1_diffcorr,parameters,model)

ax = plt.gca(projection='3d')
ax.plot(x[:,0],x[:,1],x[:,2],'-',label='$\Delta E$ = 0.1, using dcnc')
ax.scatter(x[0,0],x[0,1],x[0,2],s=20,marker='*')
ax.plot(x[:,0], x[:,1], zs=0, zdir='z')

f = lambda t,x : turningpoint_UPOsHam2dof.Ham2dof(model,t,x,parameters)  
soln = solve_ivp(f, TSPAN, x0po_1_turningpoint[-1,0:4],method='RK45',dense_output=True, events = lambda t,x : turningpoint_UPOsHam2dof.half_period(t,x,model),rtol=RelTol, atol=AbsTol)
te = soln.t_events[0]
tt = [0,te[2]]
t,x,phi_t1,PHI = turningpoint_UPOsHam2dof.stateTransitMat(tt,x0po_1_turningpoint[-1,0:4],parameters,model)

ax = plt.gca(projection='3d')
ax.plot(x[:,0],x[:,1],x[:,2],'-.',label='$\Delta E$ = 0.1, using tp')
ax.scatter(x[0,0],x[0,1],x[0,2],s=20,marker='*')
ax.plot(x[:,0], x[:,1], zs=0, zdir='z')



ax = plt.gca(projection='3d')
resX = 100
xVec = np.linspace(-1,1,resX)
yVec = np.linspace(-2,2,resX)
xMat, yMat = np.meshgrid(xVec, yVec)
#cset1 = ax.contour(xMat, yMat, uncoupled_tpcd.get_pot_surf_proj(xVec, yVec,parameters), [0.001,0.1,1,2,4],
#                       linewidths = 1.0, cmap=cm.viridis, alpha = 0.8)
cset2 = ax.contour(xMat, yMat, turningpoint_UPOsHam2dof.get_pot_surf_proj(model,xVec, yVec,parameters), 2.0,zdir='z', offset=0,
                       linewidths = 1.0, cmap=cm.viridis, alpha = 0.8)
ax.scatter(eqPt[0], eqPt[1], s = 100, c = 'r', marker = 'X')
ax.set_xlabel('$x$', fontsize=axis_fs)
ax.set_ylabel('$y$', fontsize=axis_fs)
ax.set_zlabel('$p_x$', fontsize=axis_fs)
legend = ax.legend(loc='upper left')

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-4, 4)
plt.grid()
plt.show()

plt.savefig('comparison_deleonberne.pdf',format='pdf',bbox_inches='tight')