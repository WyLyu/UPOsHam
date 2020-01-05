# -*- coding: utf-8 -*-
# """
# Created on Thu Oct 25 17:19:30 2019

# @author: Wenyang Lyu and Shibabrat Naik

# Script to compute unstable periodic orbits at specified energies for the 
# coupled quartic Hamiltonian using differential correction
# """

import numpy as np
from scipy.integrate import solve_ivp
import time

import matplotlib as mpl
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'

# This needs testing for installation 
# import sys
# sys.path.insert(0, './src/')
# import differential_correction as diffcorr
# import coupled_quartic_hamiltonian as coupled
import uposham.differential_correction as diffcorr
import uposham.coupled_quartic_hamiltonian as coupled

import os
path_to_data = os.path.join(os.path.dirname(__file__), '../data/')
path_to_saveplot = os.path.join(os.path.dirname(__file__), '../tests/plots/')
# This needs testing for installation 

#%% Setting up parameters for the method and the Hamiltonian system

save_final_plot = True
show_final_plot = False
N = 4               # dimension of phase space
omega=1.00
EPSILON_S = 0.0     # Energy of the saddle
alpha = 1.00
beta = 1.00
epsilon= 1e-1
parameters = np.array([1, omega, EPSILON_S, alpha, beta, omega, epsilon])
eqNum = 1
#deltaE_vals = [0.01, 0.1, 1.00, 2.0, 4.0]
#linecolor = ['b','r','g','m','c']
deltaE_vals = [0.1, 1.00]
linecolor = ['b','r']


eqPt = diffcorr.get_eq_pts(eqNum, coupled.init_guess_eqpt_coupled, \
                            coupled.grad_pot_coupled, parameters)

#energy of the saddle eq pt
eSaddle = diffcorr.get_total_energy([eqPt[0],eqPt[1],0,0], \
                                    coupled.pot_energy_coupled, parameters) 
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
with open("x0_diffcorr_fam_eqPt%s_coupled.dat" %eqNum,'a+') as po_fam_file:
    [po_x0Fam,po_tpFam] = diffcorr.get_po_fam(eqNum, Ax1, Ax2, nFam, \
                                            po_fam_file, \
                                            coupled.init_guess_eqpt_coupled, \
                                            coupled.grad_pot_coupled, \
                                            coupled.jacobian_coupled, \
                                            coupled.guess_lin_coupled, \
                                            coupled.diffcorr_setup_coupled, \
                                            coupled.conv_coord_coupled, \
                                            coupled.diffcorr_acc_corr_coupled, \
                                            coupled.ham2dof_coupled, \
                                            coupled.half_period_coupled, \
                                            coupled.pot_energy_coupled, \
                                            coupled.variational_eqns_coupled, \
                                            coupled.plot_iter_orbit_coupled, \
                                            parameters)  

poFamRuntime = time.time() - t
x0podata = np.concatenate((po_x0Fam, po_tpFam),axis=1)



#%%
# begins with a family of periodic orbits and steps until crossing the
# initial condition with target energy 

for i in range(len(deltaE_vals)):
    deltaE = deltaE_vals[i]
    
    with open("x0_diffcorr_fam_eqPt%s_coupled.dat" %eqNum ,'a+') as po_fam_file:
        eTarget = eSaddle + deltaE 
        print('Loading the periodic orbit family from data file',po_fam_file.name,'\n') 
        x0podata = np.loadtxt(po_fam_file.name)
    
    
    with open("x0po_T_energyPO_eqPt%s_brac%s_coupled.dat" %(eqNum,deltaE),'a+') as po_brac_file:
        t = time.time()
        x0poTarget,TTarget = diffcorr.po_bracket_energy(eTarget, x0podata, po_brac_file, \
                                                    coupled.diffcorr_setup_coupled, \
                                                    coupled.conv_coord_coupled, \
                                                    coupled.diffcorr_acc_corr_coupled, \
                                                    coupled.ham2dof_coupled, \
                                                    coupled.half_period_coupled, \
                                                    coupled.pot_energy_coupled, \
                                                    coupled.variational_eqns_coupled, \
                                                    coupled.plot_iter_orbit_coupled, \
                                                    parameters)
        poTarE_runtime = time.time()-t 
        with open("model_parameters_eqPt%s_DelE%s_coupled.dat" %(eqNum,deltaE),'a+') as model_parameters_file:
            np.savetxt(model_parameters_file.name, parameters,fmt='%1.16e')
    
    
    #%
    # target specific periodic orbit
    # Target PO of specific energy with high precision does not work for the
    # model 
    
    with open("x0_diffcorr_deltaE%s_coupled.dat" %(deltaE),'a+') as po_target_file:
                    
        [x0po, T,energyPO] = diffcorr.po_target_energy(x0poTarget,eTarget, \
                                                    po_target_file, \
                                                    coupled.diffcorr_setup_coupled, \
                                                    coupled.conv_coord_coupled, \
                                                    coupled.diffcorr_acc_corr_coupled, \
                                                    coupled.ham2dof_coupled, \
                                                    coupled.half_period_coupled, \
                                                    coupled.pot_energy_coupled, \
                                                    coupled.variational_eqns_coupled, \
                                                    coupled.plot_iter_orbit_coupled, \
                                                    parameters)



#%% Load periodic orbit data from ascii files
    
x0po = np.zeros((4,len(deltaE_vals)))

for i in range(len(deltaE_vals)):
    deltaE = deltaE_vals[i]

    with open("x0_diffcorr_deltaE%s_coupled.dat" %(deltaE),'a+') as po_fam_file:
        print('Loading the periodic orbit family from data file',po_fam_file.name,'\n') 
        x0podata = np.loadtxt(po_fam_file.name)
        x0po[:,i] = x0podata[0:4]


#%% Plotting the unstable periodic orbits at the specified energies

TSPAN = [0,30]
RelTol = 3.e-10
AbsTol = 1.e-10

plt.close('all')
figH = plt.figure(figsize=(7,7))
ax = figH.gca(projection='3d')
axis_fs = 15


for i in range(len(deltaE_vals)):
    deltaE = deltaE_vals[i]
    f= lambda t,x: coupled.ham2dof_coupled(t,x,parameters)
    soln = solve_ivp(f, TSPAN, x0po[:,i],method='RK45',dense_output=True, \
                     events = lambda t,x : coupled.half_period_coupled(t,x,parameters), \
                     rtol=RelTol, atol=AbsTol)
    
    te = soln.t_events[0]
    tt = [0,te[1]] 
    t,x,phi_t1,PHI = diffcorr.state_transit_matrix(tt,x0po[:,i], \
                                            parameters, \
                                            coupled.variational_eqns_coupled)


    

    ax.plot(x[:,0],x[:,1],x[:,3],'-',color = linecolor[i], \
            label = '$\Delta E$ = %.2f'%(deltaE))
    ax.plot(x[:,0],x[:,1],-x[:,3],'-',color = linecolor[i])
    ax.scatter(x[0,0],x[0,1],x[0,3],s=10,marker='*')
    ax.scatter(x[0,0],x[0,1],-x[0,3],s=10,marker='o')


# Plotting equipotential lines
resX = 100
xVec = np.linspace(-4,4,resX)
yVec = np.linspace(-4,4,resX)
xMat, yMat = np.meshgrid(xVec, yVec)
cset1 = ax.contour(xMat, yMat, diffcorr.get_pot_surf_proj(xVec, yVec, \
                                coupled.pot_energy_coupled, \
                                parameters), [0.01,0.1,1,2,4], \
                                zdir='z', offset=0, 
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
    # plt.savefig(path_to_saveplot + 'diff_corr_coupled_upos.pdf', \
    #             format='pdf', bbox_inches='tight')
    plt.savefig(path_to_saveplot + 'diff_corr_coupled_upos.png', \
                dpi = 300, bbox_inches='tight')


