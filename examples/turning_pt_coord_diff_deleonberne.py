# -*- coding: utf-8 -*-
# """
# Created on Wed Sep 11 17:57:34 2019

# @author: Wenyang Lyu and Shibabrat Naik
# """

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import math
from scipy import optimize

import matplotlib as mpl
from matplotlib import cm
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'

# This needs testing for installation 
import sys
import importlib
sys.path.insert(0, './src/')

import turning_point_coord_difference
importlib.reload(turning_point_coord_difference)
import turning_point_coord_difference as tpcd

import deleonberne_hamiltonian
importlib.reload(deleonberne_hamiltonian)
import deleonberne_hamiltonian as deleonberne
# This needs testing for installation


#%% Setting up parameters and global variables
save_final_plot = True
show_final_plot = False
show_itrsteps_plots = False # show iteration of the UPOs in plots
N = 4         # dimension of phase space
MASS_A = 8.0
MASS_B = 8.0 # De Leon, Marston (1989)
EPSILON_S = 1.0
D_X = 10.0
ALPHA = 1.00
LAMBDA = 1.5
parameters = np.array([MASS_A, MASS_B, EPSILON_S, D_X, LAMBDA, ALPHA])
eqNum = 1 
eqPt = tpcd.get_eq_pts(eqNum, deleonberne.init_guess_eqpt_deleonberne, \
                        deleonberne.grad_pot_deleonberne, parameters)



#%%
#E_vals = [1.1, 2.00, 3.00, 5.00]
#linecolor = ['b','r','g','m','c']
E_vals = [1.1, 2.00]
linecolor = ['b','r']

n = 4 # number of intervals we want to divide.
n_turn = 1 # nth turning point we want to choose.
    
for i in range(len(E_vals)):
    
    e = E_vals[i] # total energy.
    deltaE = e - parameters[2]
    
    #Trial initial Condition s.t. one initial condition is on the LHS of the UPO 
    #and the other one is on the RHS of the UPO
    
    f1 = lambda x: deleonberne.get_coord_deleonberne(x,0.06,e,parameters)
    x0_2 = optimize.newton(f1,-0.15)
    state0_2 = [x0_2,0.06,0.0,0.0]

    f2 = lambda x: deleonberne.get_coord_deleonberne(x,-0.05,e,parameters)
    x0_3 = optimize.newton(f2,-0.15)
    state0_3 = [x0_3,-0.05,0.0,0.0]
    
    with open("x0_tpcd_deltaE%s_deleonberne.dat" %(deltaE),'a+') as po_fam_file:
        [x0po_1, T_1,energyPO_1] = tpcd.turningPoint_configdiff(
            state0_2, state0_3, deleonberne.get_coord_deleonberne, \
            deleonberne.pot_energy_deleonberne, deleonberne.variational_eqns_deleonberne, \
            deleonberne.configdiff_deleonberne, \
            deleonberne.ham2dof_deleonberne, \
            deleonberne.half_period_deleonberne, \
            deleonberne.guess_coords_deleonberne, \
            deleonberne.plot_iter_orbit_deleonberne, \
            parameters, e, n, n_turn, show_itrsteps_plots, po_fam_file) 


#%% Load periodic orbit data from ascii files
    
x0po = np.zeros((4,len(E_vals))) #each column is a different initial condition

for i in range(len(E_vals)):
    
    e = E_vals[i]
    deltaE = e - parameters[2]

    with open("x0_tpcd_deltaE%s_deleonberne.dat" %(deltaE),'a+') as po_fam_file:
        print('Loading the periodic orbit family from data file',po_fam_file.name,'\n') 
        x0podata = np.loadtxt(po_fam_file.name)
        x0po[:,i] = x0podata[-1,0:4] 


#%% Plotting the Family
    
TSPAN = [0,30]
plt.close('all')
axis_fs = 15
RelTol = 3.e-10
AbsTol = 1.e-10

f = lambda t,x : deleonberne.ham2dof_deleonberne(t,x,parameters) 

ax = plt.gca(projection='3d')

for i in range(len(E_vals)):
    
    e = E_vals[i]
    deltaE = e - parameters[2]
    
    soln = solve_ivp(f, TSPAN, x0po[:,i], method='RK45', dense_output=True, \
                     events = lambda t,x : deleonberne.half_period_deleonberne(t,x,parameters), \
                     rtol=RelTol, atol=AbsTol)
    
    te = soln.t_events[0]
    tt = [0,te[2]]
    t,x,phi_t1,PHI = tpcd.state_transit_matrix(tt, x0po[:,i], parameters, \
                                        deleonberne.variational_eqns_deleonberne)
    
    ax.plot(x[:,0],x[:,1],x[:,2],'-',color=linecolor[i], \
            label='$\Delta E$ = %.2f'%(deltaE))
    ax.scatter(x[0,0],x[0,1],x[0,2],s=10,marker='*')
    ax.plot(x[:,0], x[:,1], zs=0, zdir='z')
    # ax.plot(x[:,0], x[:,1], zs=0, zdir='z') # 2D projection of the UPO

    
resX = 100
xVec = np.linspace(-1,1,resX)
yVec = np.linspace(-2,2,resX)
xMat, yMat = np.meshgrid(xVec, yVec)
cset1 = ax.contour(xMat, yMat, \
                    tpcd.get_pot_surf_proj(
                       xVec, yVec, deleonberne.pot_energy_deleonberne, \
                       parameters), \
                    [0.01,0.1,1,2,4], zdir='z', offset=0, \
                    linewidths = 1.0, cmap=cm.viridis, \
                    alpha = 0.8)

ax.scatter(eqPt[0], eqPt[1], s = 50, c = 'r', marker = 'X')
ax.set_xlabel('$x$', fontsize=axis_fs)
ax.set_ylabel('$y$', fontsize=axis_fs)
ax.set_zlabel('$p_x$', fontsize=axis_fs)

ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_zlim(-4, 4)
legend = ax.legend(loc='upper left')

plt.grid()

if show_final_plot:
    plt.show()

if save_final_plot:  
    plt.savefig('./tests/plots/tpcd_deleonberne_upos.pdf', format='pdf', \
                        bbox_inches='tight')

