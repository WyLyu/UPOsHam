
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 10:02:48 2019

@author: wl16298
"""

"""Plot peridoic orbits with 5 different energies using turning point method"""

# For the uncoupled problem

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
import uncoupled_turningpoint ### import module xxx where xxx is the name of the python file xxx.py 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from matplotlib import cm
from pylab import rcParams
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
#%matplotlib

#%%
alpha = 1.0
beta = 1.0
omega = 1.0
EPSILON_S = 0.0 #Energy of the saddle
parameters = np.array([1,omega, EPSILON_S, alpha, beta,omega]);
eqNum = 1;  
eqPt = uncoupled_turningpoint.get_eq_pts_uncoupled(eqNum, parameters)
"""Initial Condition for the true periodic orbit
H = T + V where T is the kinetic energy and V is the potential energy. In our example, 
the potential energy V = -0.5*alpha*x**2+0.25*beta*x**4+0.5*omega*y**2.
If we fix x, then y = +- math.sqrt((V +0.5*alpha*x**2-0.25*beta*x**4)/(0.5*omega) ) so that
the point starts at the potential energy surface V.
"""

#%%
e=0.01
"""Trial initial Condition s.t. one initial condition is on the LHS of the UPO and the other one is on the RHS of the UPO"""
state0_2 = [-0.1 , -math.sqrt(2*e+0.1**2-0.5*0.1**4),0.0,0.0]
state0_3 = [0.11 , -math.sqrt(2*e+0.11**2-0.5*0.11**4),0.0,0.0]
n=12
n_turn=1
deltaE = e-parameters[2]
po_fam_file = open("1111x0_turningpoint_deltaE%s_uncoupled.txt" %(deltaE),'a+')
[x0po_1, T_1,energyPO_1] = uncoupled_turningpoint.TurningPoint_uncoupled(state0_2,state0_3 ,parameters,e,n,n_turn,po_fam_file) ; 

po_fam_file.close()

#%%
e=0.1
"""Trial initial Condition s.t. one initial condition is on the LHS of the UPO and the other one is on the RHS of the UPO"""
state0_2 = [-0.1 , -math.sqrt(2*e+0.1**2-0.5*0.1**4),0.0,0.0]
state0_3 = [0.11 , -math.sqrt(2*e+0.11**2-0.5*0.11**4),0.0,0.0]
n=12
n_turn=1
deltaE = e-parameters[2]
po_fam_file = open("1111x0_turningpoint_deltaE%s_uncoupled.txt" %(deltaE),'a+')
[x0po_2, T_2,energyPO_2] = uncoupled_turningpoint.TurningPoint_uncoupled(state0_2,state0_3 ,parameters,e,n,n_turn,po_fam_file) ; 

po_fam_file.close()
 
#%%
e=1.0
"""Trial initial Condition s.t. one initial condition is on the LHS of the UPO and the other one is on the RHS of the UPO"""
state0_2 = [-0.077 , -math.sqrt(2*e+0.077**2-0.5*0.077**4),0.0,0.0]
state0_3 = [0.09 , -math.sqrt(2*e+0.09**2-0.5*0.09**4),0.0,0.0]
n=12
n_turn=1
deltaE = e-parameters[2]
po_fam_file = open("1111x0_turningpoint_deltaE%s_uncoupled.txt" %(deltaE),'a+')
[x0po_3, T_3,energyPO_3] = uncoupled_turningpoint.TurningPoint_uncoupled(state0_2,state0_3 ,parameters,e,n,n_turn,po_fam_file) ; 

po_fam_file.close()

#%%
e=2.0
"""Trial initial Condition s.t. one initial condition is on the LHS of the UPO and the other one is on the RHS of the UPO"""
state0_2 = [-0.1 , -math.sqrt(2*e+0.1**2-0.5*0.1**4),0.0,0.0]
state0_3 = [0.11 , -math.sqrt(2*e+0.11**2-0.5*0.11**4),0.0,0.0]
n=12
n_turn=1
deltaE = e-parameters[2]
po_fam_file = open("1111x0_turningpoint_deltaE%s_uncoupled.txt" %(deltaE),'a+')
[x0po_4, T_4,energyPO_4] = uncoupled_turningpoint.TurningPoint_uncoupled(state0_2,state0_3 ,parameters,e,n,n_turn,po_fam_file) ; 

po_fam_file.close()

#%%
e=4.0
"""Trial initial Condition s.t. one initial condition is on the LHS of the UPO and the other one is on the RHS of the UPO"""
state0_2 = [-0.1 , -math.sqrt(2*e+0.1**2-0.5*0.1**4),0.0,0.0]
state0_3 = [0.11 , -math.sqrt(2*e+0.11**2-0.5*0.11**4),0.0,0.0]
n=12
n_turn=1
deltaE = e-parameters[2]
po_fam_file = open("1111x0_turningpoint_deltaE%s_uncoupled.txt" %(deltaE),'a+')
[x0po_5, T_5,energyPO_5] = uncoupled_turningpoint.TurningPoint_uncoupled(state0_2,state0_3 ,parameters,e,n,n_turn,po_fam_file) ; 

po_fam_file.close()


#%% Load Data
deltaE = 0.010
po_fam_file = open("1111x0_turningpoint_deltaE%s_uncoupled.txt" %(deltaE),'a+');
print('Loading the periodic orbit family from data file',po_fam_file.name,'\n'); 
x0podata = np.loadtxt(po_fam_file.name)
po_fam_file.close()
x0po_1 = x0podata

deltaE = 0.10
po_fam_file = open("1111x0_turningpoint_deltaE%s_uncoupled.txt" %(deltaE),'a+');
print('Loading the periodic orbit family from data file',po_fam_file.name,'\n'); 
x0podata = np.loadtxt(po_fam_file.name)
po_fam_file.close()
x0po_2 = x0podata

deltaE = 1.0
po_fam_file = open("1111x0_turningpoint_deltaE%s_uncoupled.txt" %(deltaE),'a+');
print('Loading the periodic orbit family from data file',po_fam_file.name,'\n'); 
x0podata = np.loadtxt(po_fam_file.name)
po_fam_file.close()
x0po_3 = x0podata

deltaE = 2.0
po_fam_file = open("1111x0_turningpoint_deltaE%s_uncoupled.txt" %(deltaE),'a+');
print('Loading the periodic orbit family from data file',po_fam_file.name,'\n'); 
x0podata = np.loadtxt(po_fam_file.name)
po_fam_file.close()
x0po_4 = x0podata

deltaE = 4.0
po_fam_file = open("1111x0_turningpoint_deltaE%s_uncoupled.txt" %(deltaE),'a+');
print('Loading the periodic orbit family from data file',po_fam_file.name,'\n'); 
x0podata = np.loadtxt(po_fam_file.name)
po_fam_file.close()
x0po_5 = x0podata


#%% Plotting the Family
TSPAN = [0,30]
plt.close('all')
axis_fs = 15
RelTol = 3.e-10
AbsTol = 1.e-10
f = partial(uncoupled_turningpoint.uncoupled2dof, par=parameters) 
soln = solve_ivp(f, TSPAN, x0po_1[-1,0:4],method='RK45',dense_output=True, events = uncoupled_turningpoint.half_period,rtol=RelTol, atol=AbsTol)
te = soln.t_events[0]
tt = [0,te[1]]
t,x,phi_t1,PHI = uncoupled_turningpoint.stateTransitMat_uncoupled(tt,x0po_1[-1,0:4],parameters)

ax = plt.gca(projection='3d')
ax.plot(x[:,0],x[:,1],x[:,3],'-',color='b',label='$\Delta E$ = 0.01')
ax.plot(x[:,0],x[:,1],-x[:,3],'-',color='b')
ax.scatter(x[0,0],x[0,1],x[0,3],s=20,marker='*');
ax.scatter(x[0,0],x[0,1],-x[0,3],s=20,marker='o');
ax.plot(x[:,0], x[:,1], zs=0, zdir='z')

f = partial(uncoupled_turningpoint.uncoupled2dof, par=parameters) 
soln = solve_ivp(f, TSPAN, x0po_2[-1,0:4],method='RK45',dense_output=True, events = uncoupled_turningpoint.half_period,rtol=RelTol, atol=AbsTol)
te = soln.t_events[0]
tt = [0,te[1]]
t,x,phi_t1,PHI = uncoupled_turningpoint.stateTransitMat_uncoupled(tt,x0po_2[-1,0:4],parameters)
ax = plt.gca(projection='3d')
ax.plot(x[:,0],x[:,1],x[:,3],'-',color='r',label='$\Delta E$ = 0.1')
ax.plot(x[:,0],x[:,1],-x[:,3],'-',color='r')
ax.scatter(x[0,0],x[0,1],x[0,3],s=20,marker='*');
ax.scatter(x[0,0],x[0,1],-x[0,3],s=20,marker='o');
ax.plot(x[:,0], x[:,1], zs=0, zdir='z')

f = partial(uncoupled_turningpoint.uncoupled2dof, par=parameters) 
soln = solve_ivp(f, TSPAN, x0po_3[-1,0:4],method='RK45',dense_output=True, events = uncoupled_turningpoint.half_period,rtol=RelTol, atol=AbsTol)
te = soln.t_events[0]
tt = [0,te[1]]
t,x,phi_t1,PHI = uncoupled_turningpoint.stateTransitMat_uncoupled(tt,x0po_3[-1,0:4],parameters)
ax = plt.gca(projection='3d')
ax.plot(x[:,0],x[:,1],x[:,3],'-',color='g',label='$\Delta E$ = 1.0')
ax.plot(x[:,0],x[:,1],-x[:,3],'-',color='g')
ax.scatter(x[0,0],x[0,1],x[0,3],s=20,marker='*');
ax.scatter(x[0,0],x[0,1],-x[0,3],s=20,marker='o');
ax.plot(x[:,0], x[:,1], zs=0, zdir='z')


f = partial(uncoupled_turningpoint.uncoupled2dof, par=parameters) 
soln = solve_ivp(f, TSPAN, x0po_4[-1,0:4],method='RK45',dense_output=True, events = uncoupled_turningpoint.half_period,rtol=RelTol, atol=AbsTol)
te = soln.t_events[0]
tt = [0,te[1]]
t,x,phi_t1,PHI = uncoupled_turningpoint.stateTransitMat_uncoupled(tt,x0po_4[-1,0:4],parameters)
ax = plt.gca(projection='3d')
ax.plot(x[:,0],x[:,1],x[:,3],'-',color='m',label='$\Delta E$ = 2.0')
ax.plot(x[:,0],x[:,1],-x[:,3],'-',color='m')
ax.scatter(x[0,0],x[0,1],x[0,3],s=20,marker='*');
ax.scatter(x[0,0],x[0,1],-x[0,3],s=20,marker='o');
ax.plot(x[:,0], x[:,1], zs=0, zdir='z')


f = partial(uncoupled_turningpoint.uncoupled2dof, par=parameters) 
soln = solve_ivp(f, TSPAN, x0po_5[-1,0:4],method='RK45',dense_output=True, events = uncoupled_turningpoint.half_period,rtol=RelTol, atol=AbsTol)
te = soln.t_events[0]
tt = [0,te[1]]
t,x,phi_t1,PHI = uncoupled_turningpoint.stateTransitMat_uncoupled(tt,x0po_5[-1,0:4],parameters)
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
#cset1 = ax.contour(xMat, yMat, uncoupled_turningpoint.get_pot_surf_proj(xVec, yVec,parameters), [0.001,0.1,1,2,4],
#                       linewidths = 1.0, cmap=cm.viridis, alpha = 0.8)
cset2 = ax.contour(xMat, yMat, uncoupled_turningpoint.get_pot_surf_proj(xVec, yVec,parameters), [0.01,0.1,1,2,4],zdir='z', offset=0,
                       linewidths = 1.0, cmap=cm.viridis, alpha = 0.8)
ax.scatter(eqPt[0], eqPt[1], s = 100, c = 'r', marker = 'X')
ax.set_xlabel('$x$', fontsize=axis_fs)
ax.set_ylabel('$y$', fontsize=axis_fs)
ax.set_zlabel('$p_y$', fontsize=axis_fs)
#ax.set_title('$\Delta E$ = %1.e,%1.e,%1.e,%1.e,%1.e' %(energyPO_1[-1],energyPO_2[-1],energyPO_3[-1],energyPO_4[-1],energyPO_5[-1]) ,fontsize=axis_fs)
legend = ax.legend(loc='best')
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_zlim(-2, 2)

plt.grid()
plt.show()

plt.savefig('turningpoint_POfam_uncoupled.pdf',format='pdf',bbox_inches='tight')