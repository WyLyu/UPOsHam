# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 10:02:48 2019

@author: wl16298
"""

"""Plot peridoic orbits with 5 different energies using the new method"""

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
import coupled_tpcd ### import module xxx where xxx is the name of the python file xxx.py 
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
eqNum = 1;  
eqPt = coupled_tpcd.get_eq_pts_coupled(eqNum, parameters)

#%%
# e is the total energy, n is the number of intervals we want to divide, n_turn is the nth turning point we want to choose.
e=0.01
n=4
n_turn = 2
deltaE = e-parameters[2]
"""Trial initial Condition s.t. one initial condition is on the LHS of the UPO and the other one is on the RHS of the UPO"""
state0_2 = [0.0,coupled_tpcd.get_y(0.0,e,parameters),0.0,0.0]
state0_3 = [0.05, coupled_tpcd.get_y(0.05,e,parameters),0.0,0.0]


po_fam_file = open("1111x0_tpcd_deltaE%s_coupled.txt" %(deltaE),'a+')
[x0po_1, T_1,energyPO_1] = coupled_tpcd.tpcd_coupled(state0_2,state0_3 ,parameters,e,n,n_turn,po_fam_file) ; 
po_fam_file.close()

#%%
e=0.1
n=4
n_turn = 2
deltaE = e-parameters[2]
"""Trial initial Condition s.t. one initial condition is on the LHS of the UPO and the other one is on the RHS of the UPO"""
state0_2 = [-0.05,coupled_tpcd.get_y(-0.05,e,parameters),0.0,0.0]
state0_3 = [0.10, coupled_tpcd.get_y(0.10,e,parameters),0.0,0.0]


po_fam_file = open("1111x0_tpcd_deltaE%s_coupled.txt" %(deltaE),'a+')
[x0po_2, T_2,energyPO_2] = coupled_tpcd.tpcd_coupled(state0_2,state0_3 ,parameters,e,n,n_turn,po_fam_file) ; 
po_fam_file.close()

#%%
e=1.0
n=4
n_turn = 2
deltaE = e-parameters[2]
"""Trial initial Condition s.t. one initial condition is on the LHS of the UPO and the other one is on the RHS of the UPO"""
state0_2 = [0.01,coupled_tpcd.get_y(0.01,e,parameters),0.0,0.0]
state0_3 = [0.18, coupled_tpcd.get_y(0.18,e,parameters),0.0,0.0]


po_fam_file = open("1111x0_tpcd_deltaE%s_coupled.txt" %(deltaE),'a+')
[x0po_3, T_3,energyPO_3] = coupled_tpcd.tpcd_coupled(state0_2,state0_3 ,parameters,e,n,n_turn,po_fam_file) ; 
po_fam_file.close()

#%%
e=2.0
n=4
n_turn = 2
deltaE = e-parameters[2]
"""Trial initial Condition s.t. one initial condition is on the LHS of the UPO and the other one is on the RHS of the UPO"""
state0_2 = [0.01,coupled_tpcd.get_y(0.01,e,parameters),0.0,0.0]
state0_3 = [0.18, coupled_tpcd.get_y(0.18,e,parameters),0.0,0.0]


po_fam_file = open("1111x0_tpcd_deltaE%s_coupled.txt" %(deltaE),'a+')
[x0po_4, T_4,energyPO_4] = coupled_tpcd.tpcd_coupled(state0_2,state0_3 ,parameters,e,n,n_turn,po_fam_file) ; 
po_fam_file.close()

#%%
e=4.0
n=4
n_turn = 2
deltaE = e-parameters[2]
"""Trial initial Condition s.t. one initial condition is on the LHS of the UPO and the other one is on the RHS of the UPO"""
state0_2 = [0.01,coupled_tpcd.get_y(0.01,e,parameters),0.0,0.0]
state0_3 = [0.18, coupled_tpcd.get_y(0.18,e,parameters),0.0,0.0]


po_fam_file = open("1111x0_tpcd_deltaE%s_coupled.txt" %(deltaE),'a+')
[x0po_5, T_5,energyPO_5] = coupled_tpcd.tpcd_coupled(state0_2,state0_3 ,parameters,e,n,n_turn,po_fam_file) ; 
po_fam_file.close()


#%% Load Data
deltaE = 0.010
po_fam_file = open("1111x0_tpcd_deltaE%s_coupled.txt" %(deltaE),'a+');
print('Loading the periodic orbit family from data file',po_fam_file.name,'\n'); 
x0podata = np.loadtxt(po_fam_file.name)
po_fam_file.close()
x0po_1 = x0podata[-1,0:4]

deltaE = 0.10
po_fam_file = open("1111x0_tpcd_deltaE%s_coupled.txt" %(deltaE),'a+');
print('Loading the periodic orbit family from data file',po_fam_file.name,'\n'); 
x0podata = np.loadtxt(po_fam_file.name)
po_fam_file.close()
x0po_2 = x0podata[-1,0:4]

deltaE = 1.0
po_fam_file = open("1111x0_tpcd_deltaE%s_coupled.txt" %(deltaE),'a+');
print('Loading the periodic orbit family from data file',po_fam_file.name,'\n'); 
x0podata = np.loadtxt(po_fam_file.name)
po_fam_file.close()
x0po_3 = x0podata[-1,0:4]

deltaE = 2.0
po_fam_file = open("1111x0_tpcd_deltaE%s_coupled.txt" %(deltaE),'a+');
print('Loading the periodic orbit family from data file',po_fam_file.name,'\n'); 
x0podata = np.loadtxt(po_fam_file.name)
po_fam_file.close()
x0po_4 = x0podata[-1,0:4]

deltaE = 4.0
po_fam_file = open("1111x0_tpcd_deltaE%s_coupled.txt" %(deltaE),'a+');
print('Loading the periodic orbit family from data file',po_fam_file.name,'\n'); 
x0podata = np.loadtxt(po_fam_file.name)
po_fam_file.close()
x0po_5 = x0podata[-1,0:4]
#%% Plotting the Family
TSPAN = [0,30]
plt.close('all')
axis_fs = 15
RelTol = 3.e-10
AbsTol = 1.e-10
f = partial(coupled_tpcd.coupled2dof, par=parameters) 
soln = solve_ivp(f, TSPAN, x0po_1,method='RK45',dense_output=True, events = coupled_tpcd.half_period,rtol=RelTol, atol=AbsTol)
te = soln.t_events[0]
tt = [0,te[2]]
t,x,phi_t1,PHI = coupled_tpcd.stateTransitMat_coupled(tt,x0po_1,parameters)
ax = plt.gca(projection='3d')
ax.plot(x[:,0],x[:,1],x[:,3],'-',color='b',label='$\Delta E$ = 0.01')
ax.scatter(x[0,0],x[0,1],x[0,3],s=20,marker='*');
ax.plot(x[:,0], x[:,1], zs=0, zdir='z')

f = partial(coupled_tpcd.coupled2dof, par=parameters) 
soln = solve_ivp(f, TSPAN, x0po_2,method='RK45',dense_output=True, events = coupled_tpcd.half_period,rtol=RelTol, atol=AbsTol)
te = soln.t_events[0]
tt = [0,te[2]]
t,x,phi_t1,PHI = coupled_tpcd.stateTransitMat_coupled(tt,x0po_2,parameters)
ax = plt.gca(projection='3d')
ax.plot(x[:,0],x[:,1],x[:,3],'-',color='r',label='$\Delta E$ = 0.1')
ax.scatter(x[0,0],x[0,1],x[0,3],s=20,marker='*');
ax.plot(x[:,0], x[:,1], zs=0, zdir='z')

f = partial(coupled_tpcd.coupled2dof, par=parameters) 
soln = solve_ivp(f, TSPAN, x0po_3,method='RK45',dense_output=True, events = coupled_tpcd.half_period,rtol=RelTol, atol=AbsTol)
te = soln.t_events[0]
tt = [0,te[2]]
t,x,phi_t1,PHI = coupled_tpcd.stateTransitMat_coupled(tt,x0po_3,parameters)
ax = plt.gca(projection='3d')
ax.plot(x[:,0],x[:,1],x[:,3],'-',color='g',label='$\Delta E$ = 1.0')
ax.scatter(x[0,0],x[0,1],x[0,3],s=20,marker='*');
ax.plot(x[:,0], x[:,1], zs=0, zdir='z')


f = partial(coupled_tpcd.coupled2dof, par=parameters) 
soln = solve_ivp(f, TSPAN, x0po_4,method='RK45',dense_output=True, events = coupled_tpcd.half_period,rtol=RelTol, atol=AbsTol)
te = soln.t_events[0]
tt = [0,te[2]]
t,x,phi_t1,PHI = coupled_tpcd.stateTransitMat_coupled(tt,x0po_4,parameters)
ax = plt.gca(projection='3d')
ax.plot(x[:,0],x[:,1],x[:,3],'-',color='m',label='$\Delta E$ = 2.0')
ax.scatter(x[0,0],x[0,1],x[0,3],s=20,marker='*');
ax.plot(x[:,0], x[:,1], zs=0, zdir='z')


f = partial(coupled_tpcd.coupled2dof, par=parameters) 
soln = solve_ivp(f, TSPAN, x0po_5,method='RK45',dense_output=True, events = coupled_tpcd.half_period,rtol=RelTol, atol=AbsTol)
te = soln.t_events[0]
tt = [0,te[2]]
t,x,phi_t1,PHI = coupled_tpcd.stateTransitMat_coupled(tt,x0po_5,parameters)
ax = plt.gca(projection='3d')
ax.plot(x[:,0],x[:,1],x[:,3],'-',color='c',label='$\Delta E$ = 4.0')
ax.scatter(x[0,0],x[0,1],x[0,3],s=20,marker='*');
ax.plot(x[:,0], x[:,1], zs=0, zdir='z')


ax = plt.gca(projection='3d')
resX = 100
xVec = np.linspace(-4,4,resX)
yVec = np.linspace(-4,4,resX)
xMat, yMat = np.meshgrid(xVec, yVec)
cset1 = ax.contour(xMat, yMat, coupled_tpcd.get_pot_surf_proj(xVec, yVec,parameters), [0.01,0.1,1,2,4],zdir='z', offset=0,
                       linewidths = 1.0, cmap=cm.viridis, alpha = 0.8)

ax.scatter(eqPt[0], eqPt[1], s = 200, c = 'r', marker = 'X')
ax.set_xlabel('$x$', fontsize=axis_fs)
ax.set_ylabel('$y$', fontsize=axis_fs)
ax.set_zlabel('$p_y$', fontsize=axis_fs)
#ax.set_title('$\Delta E$ = %1.e,%1.e,%1.e,%1.e,%1.e' %(energyPO_1[-1],energyPO_2[-1],energyPO_3[-1],energyPO_4[-1],energyPO_5[-1]) ,fontsize=axis_fs)
legend = ax.legend(loc='upper left')
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_zlim(-4, 4)

plt.grid()
plt.show()

plt.savefig('tpcd_POfam_coupled.pdf',format='pdf',bbox_inches='tight')