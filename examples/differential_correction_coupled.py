# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 17:19:30 2019

@author: Wenyang Lyu and Shibabrat Naik

Script to compute unstable periodic orbits at specified energies for the 
coupled quartic Hamiltonian using differential correction
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time

import matplotlib as mpl
from matplotlib import cm
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'

import sys
sys.path.insert(0, '../src/')
import differential_correction as diffcorr
import coupled_quartic_ham

#%% Setting up parameters and global variables

N = 4               # dimension of phase space
omega=1.00
EPSILON_S = 0.0     # Energy of the saddle
alpha = 1.00
beta = 1.00
epsilon= 1e-1
parameters = np.array([1, omega, EPSILON_S, alpha, beta, omega, epsilon])
eqNum = 1
#model = 'coupled'
#eqPt = diffcorr.get_eq_pts(eqNum,model, parameters)
eqPt = diffcorr.get_eq_pts(eqNum, init_guess_eqpt_coupled, \
                                       grad_pot_coupled, parameters)

#energy of the saddle eq pt
eSaddle = diffcorr.get_total_energy([eqPt[0],eqPt[1],0,0], pot_energy_coupled, \
                                                parameters) 
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
po_fam_file = open("x0_diffcorr_fam_eqPt%s_coupled.dat" %eqNum,'a+')
[po_x0Fam,po_tpFam] = diffcorr.get_POFam(eqNum, Ax1, Ax2, nFam, \
                                                    po_fam_file, init_guess_eqpt_coupled, \
                                                    grad_pot_coupled, jacobian_coupled, \
                                                    guess_lin_coupled, diffcorr_setup_coupled, \
                                                    conv_coord_coupled, \
                                                    diffcorr_acc_corr_coupled, ham2dof_coupled, \
                                                    half_period_coupled, pot_energy_coupled, \
                                                    varEqns_coupled, plot_iter_orbit_coupled, \
                                                    parameters)  

poFamRuntime = time.time() - t
x0podata = np.concatenate((po_x0Fam, po_tpFam),axis=1)
po_fam_file.close()



#%%
# begins with a family of periodic orbits and steps until crossing the
# initial condition with target energy 
# fileName = 'x0po_T_energy_case1_L41.dat'
# fileName = 'x0po_T.dat'

#deltaE_vals = [0.01, 0.1, 1.00, 2.0, 4.0]
#linecolor = ['b','r','g','m','c']
deltaE_vals = [0.1, 1.00]
linecolor = ['b','r']

for i in range(len(deltaE_vals)):
    deltaE = deltaE_vals[i]
    
    po_fam_file = open("x0_diffcorr_fam_eqPt%s_coupled.dat" %eqNum ,'a+')
    eTarget = eSaddle + deltaE 
    print('Loading the periodic orbit family from data file',po_fam_file.name,'\n') 
    x0podata = np.loadtxt(po_fam_file.name)
    po_fam_file.close()
    
    
    #%
    po_brac_file = open("x0po_T_energyPO_eqPt%s_brac%s_coupled.dat" %(eqNum,deltaE),'a+')
    t = time.time()
    # [x0poTarget,TTarget] = bracket_POEnergy_bp(eTarget, x0podata, po_brac_file)
    x0poTarget,TTarget = diffcorr.poBracketEnergy(eTarget, x0podata, po_brac_file, \
                                                              diffcorr_setup_coupled, \
                                                              conv_coord_coupled, \
                                                              diffcorr_acc_corr_coupled, \
                                                              ham2dof_coupled, \
                                                              half_period_coupled, \
                                                              pot_energy_coupled, varEqns_coupled, \
                                                              plot_iter_orbit_coupled, \
                                                              parameters)
    poTarE_runtime = time.time()-t
    model_parameters_file = open("model_parameters_eqPt%s_DelE%s_coupled.dat" %(eqNum,deltaE),'a+')
    np.savetxt(model_parameters_file.name, parameters,fmt='%1.16e')
    model_parameters_file.close()
    po_brac_file.close()
    
    
    #%
    # target specific periodic orbit
    # Target PO of specific energy with high precision does not work for the
    # model 
    
    po_target_file = open("x0_diffcorr_deltaE%s_coupled.dat" %(deltaE),'a+')
                    
    [x0po, T,energyPO] = diffcorr.poTargetEnergy(x0poTarget,eTarget, \
                                                            po_target_file, \
                                                            diffcorr_setup_coupled, \
                                                            conv_coord_coupled, \
                                                            diffcorr_acc_corr_coupled, \
                                                            ham2dof_coupled, half_period_coupled, \
                                                            pot_energy_coupled, varEqns_coupled, \
                                                            plot_iter_orbit_coupled, \
                                                            parameters)
    
    po_target_file.close()


#%% Load periodic orbit data from ascii files
    
x0po = np.zeros((4,len(deltaE_vals)))

for i in range(len(deltaE_vals)):
    deltaE = deltaE_vals[i]

    po_fam_file = open("x0_diffcorr_deltaE%s_coupled.dat" %(deltaE),'a+')
    print('Loading the periodic orbit family from data file',po_fam_file.name,'\n') 
    x0podata = np.loadtxt(po_fam_file.name)
    po_fam_file.close()
    x0po[:,i] = x0podata[0:4]


#%% Plotting the unstable periodic orbits at the specified energies

TSPAN = [0,30]
RelTol = 3.e-10
AbsTol = 1.e-10

plt.close('all')
axis_fs = 15


for i in range(len(deltaE_vals)):
    deltaE = deltaE_vals[i]
    f= lambda t,x: ham2dof_coupled(t,x,parameters)
    soln = solve_ivp(f, TSPAN, x0po[:,i],method='RK45',dense_output=True, \
                     events = lambda t,x : half_period_coupled(t,x,parameters), \
                     rtol=RelTol, atol=AbsTol)
    
    te = soln.t_events[0]
    tt = [0,te[1]] 
    t,x,phi_t1,PHI = diffcorr.stateTransitMat(tt,x0po[:,i],parameters,varEqns_coupled)
    ax = plt.gca(projection='3d')
    ax.plot(x[:,0],x[:,1],x[:,3],'-',color = linecolor[i], label = '$\Delta E$ = %.2f'%(deltaE))
    ax.plot(x[:,0],x[:,1],-x[:,3],'-',color = linecolor[i])
    ax.scatter(x[0,0],x[0,1],x[0,3],s=20,marker='*')
    ax.scatter(x[0,0],x[0,1],-x[0,3],s=20,marker='o')
    ax.plot(x[:,0], x[:,1], zs=0, zdir='z')


ax = plt.gca(projection='3d')
resX = 100
xVec = np.linspace(-4,4,resX)
yVec = np.linspace(-4,4,resX)
xMat, yMat = np.meshgrid(xVec, yVec)
cset1 = ax.contour(xMat, yMat, \
                   diffcorr.get_pot_surf_proj(xVec, yVec, pot_energy_coupled, \
                                                          parameters), [0.01,0.1,1,2,4], \
                                                          zdir='z', offset=0, 
                                                          linewidths = 1.0, cmap=cm.viridis, \
                                                          alpha = 0.8)

ax.scatter(eqPt[0], eqPt[1], s = 200, c = 'r', marker = 'X')
ax.set_xlabel('$x$', fontsize=axis_fs)
ax.set_ylabel('$y$', fontsize=axis_fs)
ax.set_zlabel('$p_y$', fontsize=axis_fs)
#ax.set_title('$\Delta E$ = %1.e,%1.e,%1.e,%1.e,%1.e' \
#                %(energyPO_1,energyPO_2,energyPO_3,energyPO_4,energyPO_5) ,fontsize=axis_fs)

legend = ax.legend(loc='upper left')
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_zlim(-4, 4)

plt.grid()
plt.show()

plt.savefig('diffcorr_POfam_coupled.pdf',format='pdf',bbox_inches='tight')




