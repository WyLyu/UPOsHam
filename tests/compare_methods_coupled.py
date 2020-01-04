# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 16:45:58 2019

@author: Wenyang Lyu and Shibabrat Naik
"""

import numpy as np

from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from matplotlib import cm
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'

# Make sure uposham is installed for the following imports to work
import uposham.differential_correction as diffcorr
import uposham.coupled_quartic_hamiltonian as coupled

import os
path_to_data = os.path.join(os.path.dirname(__file__), '../data/')


#%% Setting up parameters and global variables

save_final_plot = True
show_final_plot = True

N = 4          # dimension of phase space
MASS_A = 1.00
MASS_B = 1.00
OMEGA = 1.00
EPSILON_S = 0.0 #Energy of the saddle
ALPHA = 1.00
BETA = 1.00
EPSILON = 1e-1
parameters = np.array([MASS_A, MASS_B, EPSILON_S, ALPHA, BETA, OMEGA, EPSILON])

eqNum = 1 
# eqPt = differential_correction.get_eq_pts(eqNum, init_guess_eqpt_coupled, \
#                                        grad_pot_coupled, parameters)
eqPt = diffcorr.get_eq_pts(eqNum, coupled.init_guess_eqpt_coupled, \
                            coupled.grad_pot_coupled, parameters)

#check energy of the saddle eq pt
eSaddle = diffcorr.get_total_energy([eqPt[0],eqPt[1],0,0], \
                                    coupled.pot_energy_coupled, parameters)

#%% Load Data


deltaE = 0.10
eSaddle = 0.0 # energy of the saddle

x0po = np.zeros((3,4)) # number of methods x number of coordinates

diff_corr_fam_file = path_to_data + "x0_diffcorr_deltaE%s_coupled.dat" %(deltaE)
turning_point_fam_file = path_to_data + "x0_turningpoint_deltaE%s_coupled.dat" %(deltaE)
tpcd_fam_file = path_to_data + "x0_tpcd_deltaE%s_coupled.dat" %(deltaE)

with open(diff_corr_fam_file ,'a+') as po_fam_file:
    print('Loading the periodic orbit family from data file',po_fam_file.name,'\n') 
    x0podata = np.loadtxt(po_fam_file.name)
    x0po[0,:] = x0podata[0:4]


with open(turning_point_fam_file ,'a+') as po_fam_file:
    print('Loading the periodic orbit family from data file',po_fam_file.name,'\n') 
    x0podata = np.loadtxt(po_fam_file.name)
    x0po[1,:] = x0podata[-1,0:4]


with open(tpcd_fam_file ,'a+') as po_fam_file:
    print('Loading the periodic orbit family from data file',po_fam_file.name,'\n') 
    x0podata = np.loadtxt(po_fam_file.name)
    x0po[2,:] = x0podata[-1,0:4]



#%% Integrate the Hamilton's equations w.r.t the initial conditions for the full period T and plot the UPOs 

TSPAN = [0,30]
plt.close('all')
axis_fs = 15
RelTol = 3.e-10
AbsTol = 1.e-10


f = lambda t,x: coupled.ham2dof_coupled(t,x,parameters)
soln = solve_ivp(f, TSPAN, x0po[0,0:4],method='RK45',dense_output=True, \
                 events = lambda t,x : coupled.half_period_coupled(t,x,parameters), \
                 rtol=RelTol, atol=AbsTol)
te = soln.t_events[0]
tt = [0,te[2]]
t,x_diffcorr,phi_t1,PHI = diffcorr.state_transit_matrix(tt, x0po[0,0:4], parameters, \
                                                        coupled.variational_eqns_coupled)



f = lambda t,x: coupled.ham2dof_coupled(t,x,parameters)
soln = solve_ivp(f, TSPAN, x0po[1,0:4], method='RK45',dense_output=True, \
                 events = lambda t,x : coupled.half_period_coupled(t,x,parameters), \
                 rtol=RelTol, atol=AbsTol)
te = soln.t_events[0]
tt = [0,te[2]]
t,x_turningpoint,phi_t1,PHI = diffcorr.state_transit_matrix(tt, x0po[1,0:4], parameters, \
                                                            coupled.variational_eqns_coupled)


f= lambda t,x: coupled.ham2dof_coupled(t,x,parameters)
soln = solve_ivp(f, TSPAN, x0po[2,0:4], method='RK45', dense_output=True, \
                 events = lambda t,x : coupled.half_period_coupled(t,x,parameters), \
                 rtol=RelTol, atol=AbsTol)
te = soln.t_events[0]
tt = [0,te[2]]
t,x_tpcd,phi_t1,PHI = diffcorr.state_transit_matrix(tt, x0po[2,0:4], parameters, \
                                                    coupled.variational_eqns_coupled)



fig = plt.figure(figsize = (10,10))
ax = plt.gca(projection='3d')
ax.plot(x_tpcd[:,0],x_tpcd[:,1],x_tpcd[:,2],':',label='$\Delta E$ = 0.1, using tpcd')
ax.scatter(x_tpcd[0,0],x_tpcd[0,1],x_tpcd[0,2],s=20,marker='*')
ax.plot(x_tpcd[:,0], x_tpcd[:,1], zs=0, zdir='z')

ax.plot(x_diffcorr[:,0],x_diffcorr[:,1],x_diffcorr[:,2],'-',label='$\Delta E$ = 0.1, using dcnc')
ax.scatter(x_diffcorr[0,0],x_diffcorr[0,1],x_diffcorr[0,2],s=20,marker='*')
ax.plot(x_diffcorr[:,0], x_diffcorr[:,1], zs=0, zdir='z')

ax.plot(x_turningpoint[:,0],x_turningpoint[:,1],x_turningpoint[:,2],'-.',label='$\Delta E$ = 0.1, using tp')
ax.scatter(x_turningpoint[0,0],x_turningpoint[0,1],x_turningpoint[0,2],s=20,marker='*')
ax.plot(x_turningpoint[:,0], x_turningpoint[:,1], zs=0, zdir='z')


resX = 100
xVec = np.linspace(-4,4,resX)
yVec = np.linspace(-4,4,resX)
xMat, yMat = np.meshgrid(xVec, yVec)
#cset1 = ax.contour(xMat, yMat, uncoupled_tpcd.get_pot_surf_proj(xVec, yVec,parameters), [0.001,0.1,1,2,4],
#                       linewidths = 1.0, cmap=cm.viridis, alpha = 0.8)
cset2 = ax.contour(xMat, yMat, \
                   diffcorr.get_pot_surf_proj(xVec, yVec, coupled.pot_energy_coupled, \
                                              parameters), [deltaE], zdir='z', offset=0, \
                   linewidths = 1.0, cmap=cm.viridis, alpha = 0.8)
ax.scatter(eqPt[0], eqPt[1], s = 100, c = 'r', marker = 'X')
ax.set_xlabel('$x$', fontsize=axis_fs)
ax.set_ylabel('$y$', fontsize=axis_fs)
ax.set_zlabel('$p_y$', fontsize=axis_fs)
legend = ax.legend(loc='upper left')

ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_zlim(-0.05, 0.05)
plt.grid()


if show_final_plot:
    plt.show()

if save_final_plot:  
    #    plt.savefig('./tests/plots/comparison_coupled.pdf', format='pdf', \
#                bbox_inches='tight')
    plt.savefig('./tests/plots/comparison_coupled.png', dpi = 300, \
                bbox_inches='tight')
































