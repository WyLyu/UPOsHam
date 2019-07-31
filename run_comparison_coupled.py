# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 16:45:58 2019

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
import coupled_newmethod  ### import module xxx where xxx is the name of the python file xxx.py 
import coupled_turningpoint2
import coupled_diffcorr


from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from matplotlib import cm
from pylab import rcParams
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
%matplotlib
from scipy import optimize


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
#%% Load Data
deltaE = 0.10
eSaddle = 0.0 # energy of the saddle
po_fam_file = open("1111x0_newmethod_deltaE%s_coupled.txt" %(deltaE),'a+');
print('Loading the periodic orbit family from data file',po_fam_file.name,'\n'); 
x0podata = np.loadtxt(po_fam_file.name)
po_fam_file.close()
x0po_1_newmethod = x0podata


po_fam_file = open("1111x0_turningpoint_deltaE%s_coupled.txt" %(deltaE),'a+');
print('Loading the periodic orbit family from data file',po_fam_file.name,'\n'); 
x0podata = np.loadtxt(po_fam_file.name)
po_fam_file.close()
x0po_1_turningpoint = x0podata


po_fam_file = open("1111x0_diffcorr_deltaE%s_coupled.txt" %(deltaE),'a+');
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
f = partial(coupled_diffcorr.coupled2dof, par=parameters) 
soln = solve_ivp(f, TSPAN, x0po_1_newmethod[-1,0:4],method='RK45',dense_output=True, events = coupled_diffcorr.half_period,rtol=RelTol, atol=AbsTol)
te = soln.t_events[0]
tt = [0,te[2]]
t,x,phi_t1,PHI = coupled_diffcorr.stateTransitMat_coupled(tt,x0po_1_newmethod[-1,0:4],parameters)

ax = plt.gca(projection='3d')
ax.plot(x[:,0],x[:,1],x[:,3],'-',label='$\Delta E$ = 0.1, using newmethod')
ax.scatter(x[0,0],x[0,1],x[0,3],s=20,marker='*');
ax.plot(x[:,0], x[:,1], zs=0, zdir='z')

f = partial(coupled_diffcorr.coupled2dof, par=parameters) 
soln = solve_ivp(f, TSPAN, x0po_1_diffcorr,method='RK45',dense_output=True, events = coupled_diffcorr.half_period,rtol=RelTol, atol=AbsTol)
te = soln.t_events[0]
tt = [0,te[2]]
t,x,phi_t1,PHI = coupled_diffcorr.stateTransitMat_coupled(tt,x0po_1_diffcorr,parameters)

ax = plt.gca(projection='3d')
ax.plot(x[:,0],x[:,1],x[:,3],':',label='$\Delta E$ = 0.1, using differential correction')
ax.scatter(x[0,0],x[0,1],x[0,3],s=20,marker='*');
ax.plot(x[:,0], x[:,1], zs=0, zdir='z')

f = partial(coupled_diffcorr.coupled2dof, par=parameters) 
soln = solve_ivp(f, TSPAN, x0po_1_turningpoint[-1,0:4],method='RK45',dense_output=True, events = coupled_diffcorr.half_period,rtol=RelTol, atol=AbsTol)
te = soln.t_events[0]
tt = [0,te[2]]
t,x,phi_t1,PHI = coupled_diffcorr.stateTransitMat_coupled(tt,x0po_1_turningpoint[-1,0:4],parameters)

ax = plt.gca(projection='3d')
ax.plot(x[:,0],x[:,1],x[:,3],'-.',label='$\Delta E$ = 0.1, using turning point method')
ax.scatter(x[0,0],x[0,1],x[0,3],s=20,marker='*');
ax.plot(x[:,0], x[:,1], zs=0, zdir='z')



ax = plt.gca(projection='3d')
resX = 100
xVec = np.linspace(-4,4,resX)
yVec = np.linspace(-4,4,resX)
xMat, yMat = np.meshgrid(xVec, yVec)
#cset1 = ax.contour(xMat, yMat, uncoupled_newmethod.get_pot_surf_proj(xVec, yVec,parameters), [0.001,0.1,1,2,4],
#                       linewidths = 1.0, cmap=cm.viridis, alpha = 0.8)
cset2 = ax.contour(xMat, yMat, coupled_diffcorr.get_pot_surf_proj(xVec, yVec,parameters), 0.1,zdir='z', offset=0,
                       linewidths = 1.0, cmap=cm.viridis, alpha = 0.8)
ax.scatter(eqPt[0], eqPt[1], s = 100, c = 'r', marker = 'X')
ax.set_xlabel('$x$', fontsize=axis_fs)
ax.set_ylabel('$y$', fontsize=axis_fs)
ax.set_zlabel('$v_y$', fontsize=axis_fs)
legend = ax.legend(loc='best')

ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-0.5, 0.5)
plt.grid()
plt.show()