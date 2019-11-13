
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 10:02:48 2019

@author: Wenyang Lyu and Shibabrat Naik

Compute unstable peridoic orbits at different energies using turning point method
"""

# For the coupled problem
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import math
from scipy import optimize
import sys

import matplotlib as mpl
from matplotlib import cm
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'

# This needs testing for installation 
import sys
import importlib
sys.path.insert(0, './src/')

import turning_point
importlib.reload(turning_point)
import turning_point as tp

import uncoupled_quartic_hamiltonian
importlib.reload(uncoupled_quartic_hamiltonian)
import uncoupled_quartic_hamiltonian as uncoupled
# This needs testing for installation


#%% Setting up parameters and global variables

save_final_plot = True
show_final_plot = False
show_itrsteps_plots = False # show iteration of the UPOs in plots
N = 4          # dimension of phase space
alpha = 1.0
beta = 1.0
omega = 1.0
EPSILON_S = 0.0 #Energy of the saddle
parameters = np.array([1,omega, EPSILON_S, alpha, beta,omega])
eqNum = 1 
eqPt = tp.get_eq_pts(eqNum, uncoupled.init_guess_eqpt_uncoupled, \
                    uncoupled.grad_pot_uncoupled, parameters)


#%%

#deltaE_vals = [0.01, 0.1, 1.00, 2.0, 4.0]
#xLeft = [0.0,-0.05,0.01,0.01,0.18]
#xRight = [0.05,0.10,0.18,0.18,0.18]
#linecolor = ['b','r','g','m','c']
deltaE_vals = [0.1, 1.00]
xLeft = [0.0,0.01]
xRight = [0.05,0.11]
linecolor = ['b','r']


for i in range(len(deltaE_vals)):
    
    e = deltaE_vals[i] # total energy
    n = 12 # number of intervals we want to divide
    n_turn = 1 # nth turning point we want to choose
    deltaE = e - parameters[2]
    
    #Trial initial Condition s.t. one initial condition is on the LHS of the UPO and the 
    #other one is on the RHS of the UPO
    
    state0_2 = [-0.1 , -math.sqrt(2*e+0.1**2-0.5*0.1**4),0.0,0.0]
    state0_3 = [0.11 , -math.sqrt(2*e+0.11**2-0.5*0.11**4),0.0,0.0]
    
    po_fam_file = open("x0_turningpoint_deltaE%s_uncoupled.dat" %(deltaE),'a+')
    [x0po_1, T_1,energyPO_1] = tp.turningPoint(
        state0_2, state0_3, uncoupled.get_coord_uncoupled, \
        uncoupled.guess_coords_uncoupled, uncoupled.ham2dof_uncoupled, \
        uncoupled.half_period_uncoupled, uncoupled.variational_eqns_uncoupled, \
        uncoupled.pot_energy_uncoupled, uncoupled.plot_iter_orbit_uncoupled, \
        parameters, e, n, n_turn, show_itrsteps_plots, po_fam_file) 
    po_fam_file.close()



#%% Load periodic orbit data from ascii files
    
x0po = np.zeros((4,len(deltaE_vals))) #each column is a different initial condition

for i in range(len(deltaE_vals)):
    deltaE = deltaE_vals[i]

    po_fam_file = open("x0_turningpoint_deltaE%s_uncoupled.dat" %(deltaE),'a+')
    print('Loading the periodic orbit family from data file',po_fam_file.name,'\n') 
    x0podata = np.loadtxt(po_fam_file.name)
    po_fam_file.close()
    x0po[:,i] = x0podata[-1,0:4] 


#%% Plotting the Family

TSPAN = [0,30]
plt.close('all')
axis_fs = 15
RelTol = 3.e-10
AbsTol = 1.e-10

f = lambda t,x : uncoupled.ham2dof_uncoupled(t,x,parameters) 

ax = plt.gca(projection='3d')

for i in range(len(deltaE_vals)):
    
    soln = solve_ivp(f, TSPAN, x0po[:,i], method='RK45', dense_output=True, \
                     events = lambda t,x : uncoupled.half_period_uncoupled(t,x,parameters), \
                     rtol=RelTol, atol=AbsTol)
    te = soln.t_events[0]
    tt = [0,te[2]]
    t,x,phi_t1,PHI = tp.state_transit_matrix(tt, x0po[:,i], parameters, \
                                        uncoupled.variational_eqns_uncoupled)
    
    
    ax.plot(x[:,0],x[:,1],x[:,3],'-', color=linecolor[i], \
            label='$\Delta E$ = %.2f'%(deltaE_vals[i]))
    ax.scatter(x[0,0],x[0,1],x[0,3], s=10, marker='*')
    ax.plot(x[:,0], x[:,1], zs=0, zdir='z')


resX = 100
xVec = np.linspace(-4,4,resX)
yVec = np.linspace(-4,4,resX)
xMat, yMat = np.meshgrid(xVec, yVec)
cset1 = ax.contour(
        xMat, yMat, tp.get_pot_surf_proj(
            xVec, yVec, uncoupled.pot_energy_uncoupled, parameters), \
        [0.01,0.1,1,2,4], zdir='z', offset=0, \
        linewidths = 1.0, cmap=cm.viridis, \
        alpha = 0.8)

ax.scatter(eqPt[0], eqPt[1], s = 50, c = 'r', marker = 'X')
ax.set_xlabel('$x$', fontsize=axis_fs)
ax.set_ylabel('$y$', fontsize=axis_fs)
ax.set_zlabel('$p_y$', fontsize=axis_fs)

legend = ax.legend(loc='upper left')
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_zlim(-4, 4)

plt.grid()

if show_final_plot:
    plt.show()

if save_final_plot:  
    plt.savefig('./tests/plots/tp_uncoupled_upos.pdf', format='pdf', \
                bbox_inches='tight')


