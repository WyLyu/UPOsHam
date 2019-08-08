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
import deleonberne_newmethod  ### import module xxx where xxx is the name of the python file xxx.py 
import deleonberne_turningpoint2
import deleonberne_diffcorr


from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from matplotlib import cm
from pylab import rcParams
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
#%matplotlib
from scipy import optimize


#%% Setting up parameters and global variables
N = 4;          # dimension of phase space
MASS_A = 8.0; MASS_B = 8.0; # De Leon, Marston (1989)
EPSILON_S = 1.0;
D_X = 10.0;
ALPHA = 1.00;
LAMBDA = 1.5;
parameters = np.array([MASS_A, MASS_B, EPSILON_S, D_X, LAMBDA, ALPHA]);
eqNum = 1;  

eqPt = deleonberne_diffcorr.get_eq_pts_deleonberne(eqNum, parameters)



eSaddle = deleonberne_diffcorr.get_total_energy_deleonberne([eqPt[0],eqPt[1],0,0], parameters) #energy of the saddle eq pt
#If orbit is an n-array, e.g. orbit = [orbit_0, orbit_1, ..., orbit_n]
#%% Load Data
deltaE = 1.0
eSaddle = 1.0 # energy of the saddle
po_fam_file = open("1111x0_newmethod_deltaE%s_deleonberne.txt" %(deltaE),'a+');
print('Loading the periodic orbit family from data file',po_fam_file.name,'\n'); 
x0podata = np.loadtxt(po_fam_file.name)
po_fam_file.close()
x0po_1_newmethod = x0podata


po_fam_file = open("1111x0_turningpoint_deltaE%s_deleonberne.txt" %(deltaE),'a+');
print('Loading the periodic orbit family from data file',po_fam_file.name,'\n'); 
x0podata = np.loadtxt(po_fam_file.name)
po_fam_file.close()
x0po_1_turningpoint = x0podata


po_fam_file = open("1111x0_diffcorr_deltaE%s_deleonberne.txt" %(deltaE),'a+');
print('Loading the periodic orbit family from data file',po_fam_file.name,'\n'); 
x0podata = np.loadtxt(po_fam_file.name)
po_fam_file.close()
x0po_1_diffcorr = x0podata[0:4]

#%%
TSPAN = [0,30]
plt.close('all')
axis_fs = 15
RelTol = 3.e-10
AbsTol = 1.e-10
f = partial(deleonberne_diffcorr.deleonberne2dof, par=parameters) 
soln = solve_ivp(f, TSPAN, x0po_1_newmethod[-1,0:4],method='RK45',dense_output=True, events = deleonberne_diffcorr.half_period,rtol=RelTol, atol=AbsTol)
te = soln.t_events[0]
tt = [0,te[2]]
t,x,phi_t1,PHI = deleonberne_diffcorr.stateTransitMat_deleonberne(tt,x0po_1_newmethod[-1,0:4],parameters)

ax = plt.gca(projection='3d')
ax.plot(x[:,0],x[:,1],x[:,2],':',label='$\Delta E$ = 0.1, using newmethod')
ax.scatter(x[0,0],x[0,1],x[0,2],s=20,marker='*');
ax.plot(x[:,0], x[:,1], zs=0, zdir='z')

f = partial(deleonberne_diffcorr.deleonberne2dof, par=parameters) 
soln = solve_ivp(f, TSPAN, x0po_1_diffcorr,method='RK45',dense_output=True, events = deleonberne_diffcorr.half_period,rtol=RelTol, atol=AbsTol)
te = soln.t_events[0]
tt = [0,te[2]]
t,x,phi_t1,PHI = deleonberne_diffcorr.stateTransitMat_deleonberne(tt,x0po_1_diffcorr,parameters)

ax = plt.gca(projection='3d')
ax.plot(x[:,0],x[:,1],x[:,2],'-',label='$\Delta E$ = 0.1, using differential correction')
ax.scatter(x[0,0],x[0,1],x[0,2],s=20,marker='*');
ax.plot(x[:,0], x[:,1], zs=0, zdir='z')

f = partial(deleonberne_diffcorr.deleonberne2dof, par=parameters) 
soln = solve_ivp(f, TSPAN, x0po_1_turningpoint[-1,0:4],method='RK45',dense_output=True, events = deleonberne_diffcorr.half_period,rtol=RelTol, atol=AbsTol)
te = soln.t_events[0]
tt = [0,te[2]]
t,x,phi_t1,PHI = deleonberne_diffcorr.stateTransitMat_deleonberne(tt,x0po_1_turningpoint[-1,0:4],parameters)

ax = plt.gca(projection='3d')
ax.plot(x[:,0],x[:,1],x[:,2],'-.',label='$\Delta E$ = 0.1, using turning point method')
ax.scatter(x[0,0],x[0,1],x[0,2],s=20,marker='*');
ax.plot(x[:,0], x[:,1], zs=0, zdir='z')



ax = plt.gca(projection='3d')
resX = 100
xVec = np.linspace(-1,1,resX)
yVec = np.linspace(-2,2,resX)
xMat, yMat = np.meshgrid(xVec, yVec)
#cset1 = ax.contour(xMat, yMat, uncoupled_newmethod.get_pot_surf_proj(xVec, yVec,parameters), [0.001,0.1,1,2,4],
#                       linewidths = 1.0, cmap=cm.viridis, alpha = 0.8)
cset2 = ax.contour(xMat, yMat, deleonberne_diffcorr.get_pot_surf_proj(xVec, yVec,parameters), 2.0,zdir='z', offset=0,
                       linewidths = 1.0, cmap=cm.viridis, alpha = 0.8)
ax.scatter(eqPt[0], eqPt[1], s = 100, c = 'r', marker = 'X')
ax.set_xlabel('$x$', fontsize=axis_fs)
ax.set_ylabel('$y$', fontsize=axis_fs)
ax.set_zlabel('$v_x$', fontsize=axis_fs)
legend = ax.legend(loc='best')

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-4, 4)
plt.grid()
plt.show()