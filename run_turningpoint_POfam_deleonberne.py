# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 10:02:48 2019

@author: wl16298
"""

"""Plot peridoic orbits with 5 different energies using turning point method"""

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
import tp_UPOsHam2dof ### import module xxx where xxx is the name of the python file xxx.py 
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
eqPt = tp_UPOsHam2dof.get_eq_pts(eqNum, model,parameters)
#%%
# e is the total energy, n is the number of intervals we want to divide, n_turn is the nth turning point we want to choose.
e=1.1
n=12
n_turn = 1
deltaE = e-parameters[2] #In this case deltaE = 0.1
"""Trial initial Condition s.t. one initial condition is on the top of the UPO and the other one is on the bottom of the UPO"""
f1 = lambda x: tp_UPOsHam2dof.get_coordinate(model,x,0.06,e,parameters)
x0_2 = optimize.newton(f1,-0.15)
state0_2 = [x0_2,0.06,0.0,0.0]
f2 = lambda x: tp_UPOsHam2dof.get_coordinate(model,x,-0.05,e,parameters)
x0_3 = optimize.newton(f2,-0.15)
state0_3 = [x0_3, -0.05,0.0,0.0]

po_fam_file = open("1111x0_turningpoint_deltaE%s_deleonberne.txt" %(0.1),'a+')
[x0po_1, T_1,energyPO_1] = tp_UPOsHam2dof.turningPoint(model,state0_3,state0_2 ,parameters,e,n,n_turn,po_fam_file)  

po_fam_file.close()
#%%
# e is the total energy, n is the number of intervals we want to divide, n_turn is the nth turning point we want to choose.
e=2.0
n=12
n_turn = 1
deltaE = e-parameters[2]
"""Trial initial Condition s.t. one initial condition is on the LHS of the UPO and the other one is on the RHS of the UPO"""
f1 = lambda x: tp_UPOsHam2dof.get_coordinate(model,x,0.06,e,parameters)
x0_2 = optimize.newton(f1,-0.15)
state0_2 = [x0_2,0.06,0.0,0.0]
f2 = lambda x: tp_UPOsHam2dof.get_coordinate(model,x,-0.05,e,parameters)
x0_3 = optimize.newton(f2,-0.15)
state0_3 = [x0_3, -0.05,0.0,0.0]

po_fam_file = open("1111x0_turningpoint_deltaE%s_deleonberne.txt" %(deltaE),'a+')
[x0po_2, T_2,energyPO_2] = tp_UPOsHam2dof.turningPoint(model,state0_3,state0_2 ,parameters,e,n,n_turn,po_fam_file)  

po_fam_file.close()
#%%
# e is the total energy, n is the number of intervals we want to divide, n_turn is the nth turning point we want to choose.
e=3.0
n=12
n_turn = 1
deltaE = e-parameters[2]
"""Trial initial Condition s.t. one initial condition is on the LHS of the UPO and the other one is on the RHS of the UPO"""
f1 = lambda x: tp_UPOsHam2dof.get_coordinate(model,x,0.06,e,parameters)
x0_2 = optimize.newton(f1,-0.15)
state0_2 = [x0_2,0.06,0.0,0.0]
f2 = lambda x: tp_UPOsHam2dof.get_coordinate(model,x,-0.05,e,parameters)
x0_3 = optimize.newton(f2,-0.15)
state0_3 = [x0_3, -0.05,0.0,0.0]

po_fam_file = open("1111x0_turningpoint_deltaE%s_deleonberne.txt" %(deltaE),'a+')
[x0po_3, T_3,energyPO_3] = tp_UPOsHam2dof.turningPoint(model,state0_3,state0_2 ,parameters,e,n,n_turn,po_fam_file)  

po_fam_file.close()
#%%
# e is the total energy, n is the number of intervals we want to divide, n_turn is the nth turning point we want to choose.
e=5.0
n=12
n_turn = 1
deltaE = e-parameters[2]
"""Trial initial Condition s.t. one initial condition is on the LHS of the UPO and the other one is on the RHS of the UPO"""
f1 = lambda x: tp_UPOsHam2dof.get_coordinate(model,x,0.06,e,parameters)
x0_2 = optimize.newton(f1,-0.15)
state0_2 = [x0_2,0.06,0.0,0.0]
f2 = lambda x: tp_UPOsHam2dof.get_coordinate(model,x,-0.05,e,parameters)
x0_3 = optimize.newton(f2,-0.15)
state0_3 = [x0_3, -0.05,0.0,0.0]

po_fam_file = open("1111x0_turningpoint_deltaE%s_deleonberne.txt" %(deltaE),'a+')
[x0po_4, T_4,energyPO_4] = tp_UPOsHam2dof.turningPoint(model,state0_3,state0_2 ,parameters,e,n,n_turn,po_fam_file)  

po_fam_file.close()

#%% Load Data
deltaE = 0.10
po_fam_file = open("1111x0_turningpoint_deltaE%s_deleonberne.txt" %(deltaE),'a+')
print('Loading the periodic orbit family from data file',po_fam_file.name,'\n') 
x0podata = np.loadtxt(po_fam_file.name)
po_fam_file.close()
x0po_1 = x0podata

deltaE = 1.0
po_fam_file = open("1111x0_turningpoint_deltaE%s_deleonberne.txt" %(deltaE),'a+')
print('Loading the periodic orbit family from data file',po_fam_file.name,'\n') 
x0podata = np.loadtxt(po_fam_file.name)
po_fam_file.close()
x0po_2 = x0podata

deltaE = 2.0
po_fam_file = open("1111x0_turningpoint_deltaE%s_deleonberne.txt" %(deltaE),'a+')
print('Loading the periodic orbit family from data file',po_fam_file.name,'\n') 
x0podata = np.loadtxt(po_fam_file.name)
po_fam_file.close()
x0po_3 = x0podata

deltaE = 4.0
po_fam_file = open("1111x0_turningpoint_deltaE%s_deleonberne.txt" %(deltaE),'a+')
print('Loading the periodic orbit family from data file',po_fam_file.name,'\n') 
x0podata = np.loadtxt(po_fam_file.name)
po_fam_file.close()
x0po_4 = x0podata

#%% Plotting the Family
TSPAN = [0,30]
plt.close('all')
axis_fs = 15
RelTol = 3.e-10
AbsTol = 1.e-10
f = lambda t,x : tp_UPOsHam2dof.Ham2dof(model,t,x,parameters) 
soln = solve_ivp(f, TSPAN, x0po_1[-1,0:4],method='RK45',dense_output=True, events = lambda t,x : tp_UPOsHam2dof.half_period(t,x,model),rtol=RelTol, atol=AbsTol)
te = soln.t_events[0]
tt = [0,te[1]]
t,x,phi_t1,PHI = tp_UPOsHam2dof.stateTransitMat(tt,x0po_1[-1,0:4],parameters,model)
ax = plt.gca(projection='3d')
ax.plot(x[:,0],x[:,1],x[:,2],'-',color='b',label='$\Delta E$ = 0.1')
ax.plot(x[:,0],x[:,1],-x[:,2],'-',color='b')
ax.scatter(x[0,0],x[0,1],x[0,2],s=20,marker='*')
ax.scatter(x[0,0],x[0,1],-x[0,2],s=20,marker='o')
ax.plot(x[:,0], x[:,1], zs=0, zdir='z')


f = lambda t,x : tp_UPOsHam2dof.Ham2dof(model,t,x,parameters) 
soln = solve_ivp(f, TSPAN, x0po_2[-1,0:4],method='RK45',dense_output=True, events = lambda t,x : tp_UPOsHam2dof.half_period(t,x,model),rtol=RelTol, atol=AbsTol)
te = soln.t_events[0]
tt = [0,te[1]]
t,x,phi_t1,PHI = tp_UPOsHam2dof.stateTransitMat(tt,x0po_2[-1,0:4],parameters,model)
ax = plt.gca(projection='3d')
ax.plot(x[:,0],x[:,1],x[:,2],'-',color='r',label='$\Delta E$ = 1.0')
ax.plot(x[:,0],x[:,1],-x[:,2],'-',color='r')
ax.scatter(x[0,0],x[0,1],x[0,2],s=20,marker='*')
ax.scatter(x[0,0],x[0,1],-x[0,2],s=20,marker='o')
ax.plot(x[:,0], x[:,1], zs=0, zdir='z')

f = lambda t,x : tp_UPOsHam2dof.Ham2dof(model,t,x,parameters) 
soln = solve_ivp(f, TSPAN, x0po_3[-1,0:4],method='RK45',dense_output=True, events = lambda t,x : tp_UPOsHam2dof.half_period(t,x,model),rtol=RelTol, atol=AbsTol)
te = soln.t_events[0]
tt = [0,te[1]]
t,x,phi_t1,PHI = tp_UPOsHam2dof.stateTransitMat(tt,x0po_3[-1,0:4],parameters,model)
ax = plt.gca(projection='3d')
ax.plot(x[:,0],x[:,1],x[:,2],'-',color='g',label='$\Delta E$ = 2.0')
ax.plot(x[:,0],x[:,1],-x[:,2],'-',color='g')
ax.scatter(x[0,0],x[0,1],x[0,2],s=20,marker='*')
ax.scatter(x[0,0],x[0,1],-x[0,2],s=20,marker='o')
ax.plot(x[:,0], x[:,1], zs=0, zdir='z')


f = lambda t,x : tp_UPOsHam2dof.Ham2dof(model,t,x,parameters) 
soln = solve_ivp(f, TSPAN, x0po_4[-1,0:4],method='RK45',dense_output=True, events = lambda t,x : tp_UPOsHam2dof.half_period(t,x,model),rtol=RelTol, atol=AbsTol)
te = soln.t_events[0]
tt = [0,te[1]]
t,x,phi_t1,PHI = tp_UPOsHam2dof.stateTransitMat(tt,x0po_4[-1,0:4],parameters,model)
ax = plt.gca(projection='3d')
ax.plot(x[:,0],x[:,1],x[:,2],'-',color='m',label='$\Delta E$ = 4.0')
ax.plot(x[:,0],x[:,1],-x[:,2],'-',color='m')
ax.scatter(x[0,0],x[0,1],x[0,2],s=20,marker='*')
ax.scatter(x[0,0],x[0,1],-x[0,2],s=20,marker='o')
ax.plot(x[:,0], x[:,1], zs=0, zdir='z')

ax = plt.gca(projection='3d')
resX = 100
xVec = np.linspace(-1,2,resX)
yVec = np.linspace(-2,2,resX)
xMat, yMat = np.meshgrid(xVec, yVec)
cset1 = ax.contour(xMat, yMat, tp_UPOsHam2dof.get_pot_surf_proj(model,xVec, yVec,parameters), [1.1,2,3,5],zdir='z', offset=0,
                       linewidths = 1.0, cmap=cm.viridis, alpha = 0.8)

ax.scatter(eqPt[0], eqPt[1], s = 200, c = 'r', marker = 'X')
ax.set_xlabel('$x$', fontsize=axis_fs)
ax.set_ylabel('$y$', fontsize=axis_fs)
ax.set_zlabel('$p_x$', fontsize=axis_fs)
#ax.set_title('$\Delta E$ = %1.e,%1.e,%1.e,%1.e,%1.e' %(energyPO_1[-1]-parameters[2],energyPO_2[-1]-parameters[2],energyPO_3[-1]-parameters[2],energyPO_4[-1]-parameters[2],energyPO_5[-1]-parameters[2]) ,fontsize=axis_fs)
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_zlim(-4, 4)
legend = ax.legend(loc='upper left')

plt.grid()
plt.show()

plt.savefig('turningpoint_POfam_deleonberne.pdf',format='pdf',bbox_inches='tight')