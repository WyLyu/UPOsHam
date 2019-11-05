# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 10:02:48 2019

@author: Wenyang Lyu and Shibabrat Naik

Compute unstable peridoic orbits at different energies using turning point method
"""

# For the DeLeon-Berne problem
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

import turning_point
importlib.reload(turning_point)
import turning_point as tp

import deleonberne_hamiltonian
importlib.reload(deleonberne_hamiltonian)
import deleonberne_hamiltonian as deleonberne
# This needs testing for installation

#%% Setting up parameters and global variables
"""
Initial Condition for the true periodic orbit
H = T + V where T is the kinetic energy and V is the potential energy. In our example, 
the potential energy V = -0.5*alpha*x**2+0.25*beta*x**4+0.5*omega*y**2.
If we fix x, then y = +- math.sqrt((V +0.5*alpha*x**2-0.25*beta*x**4)/(0.5*omega) ) so that
the point starts at the potential energy surface V.
"""
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
eqPt = tp.get_eq_pts(eqNum, deleonberne.init_guess_eqpt_deleonberne, \
                    deleonberne.grad_pot_deleonberne, parameters)


#%% 
#E_vals = [1.1, 2.00, 3.00, 5.00]
#linecolor = ['b','r','g','m','c']
E_vals = [1.1, 2.00]
linecolor = ['b','r']

n = 4 # number of intervals we want to divide
n_turn = 1 # nth turning point we want to choose.
    
for i in range(len(E_vals)):
    
    e = E_vals[i] # total energy
    deltaE = e - parameters[2]
    
    """
    Trial initial Condition s.t. one initial condition is on the LHS of the UPO and the 
    other one is on the RHS of the UPO
    """
    f1 = lambda x: deleonberne.get_coord_deleonberne(x,0.06,e,parameters)
    x0_2 = optimize.newton(f1,-0.15)
    state0_2 = [x0_2,0.06,0.0,0.0]

    f2 = lambda x: deleonberne.get_coord_deleonberne(x,-0.05,e,parameters)
    x0_3 = optimize.newton(f2,-0.15)
    state0_3 = [x0_3,-0.05,0.0,0.0]
    
    po_fam_file = open("x0_turningpoint_deltaE%s_deleonberne.dat"%(deltaE),'a+')
    
    [x0po_1, T_1,energyPO_1] = tp.turningPoint( 
        state0_2, state0_3, deleonberne.get_coord_deleonberne, \
        deleonberne.guess_coords_deleonberne, deleonberne.ham2dof_deleonberne, \
        deleonberne.half_period_deleonberne, deleonberne.varEqns_deleonberne, \
        deleonberne.pot_energy_deleonberne, \
        deleonberne.plot_iter_orbit_deleonberne, 
        parameters, e, n, n_turn, show_itrsteps_plots, po_fam_file) 
    
    po_fam_file.close()

#%% Load periodic orbit data from ascii files
    
x0po = np.zeros((4,len(E_vals))) #each column is a different initial condition

for i in range(len(E_vals)):
    
    e = E_vals[i]
    deltaE = e - parameters[2]

    po_fam_file = open("x0_turningpoint_deltaE%s_deleonberne.dat"%(deltaE),'a+')
    print('Loading the periodic orbit family from data file',po_fam_file.name,'\n') 
    x0podata = np.loadtxt(po_fam_file.name)
    po_fam_file.close()
    x0po[:,i] = x0podata[-1,0:4] 
   
    
#%% Plotting the family

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
    t,x,phi_t1,PHI = tp.stateTransitMat(tt, x0po[:,i], parameters, \
                                        deleonberne.varEqns_deleonberne)
    
    ax.plot(x[:,0],x[:,1],x[:,2],'-',color=linecolor[i], \
            label='$\Delta E$ = %.2f'%(deltaE))
    ax.scatter(x[0,0],x[0,1],x[0,2],s=20,marker='*')
    ax.plot(x[:,0], x[:,1], zs=0, zdir='z')

    
resX = 100
xVec = np.linspace(-1,1,resX)
yVec = np.linspace(-2,2,resX)
xMat, yMat = np.meshgrid(xVec, yVec)
cset1 = ax.contour(xMat, yMat, tp.get_pot_surf_proj(xVec, yVec, \
                    deleonberne.pot_energy_deleonberne, parameters), \
                    [0.01,0.1,1,2,4], zdir='z', offset=0, \
                    linewidths = 1.0, cmap=cm.viridis, \
                    alpha = 0.8)

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
# plt.show()

if show_final_plot:
    plt.show()

if save_final_plot:  
    plt.savefig('./tests/plots/tp_deleonberne_upos.pdf', format='pdf', \
                bbox_inches='tight')




#%%
# e is the total energy, n is the number of intervals we want to divide, n_turn is the nth turning point we want to choose.
#e=1.1
#n=12
#n_turn = 1
#deltaE = e-parameters[2] #In this case deltaE = 0.1
#"""Trial initial Condition s.t. one initial condition is on the top of the UPO and the other one is on the bottom of the UPO"""
#f1 = lambda x: tp.get_coordinate(model,x,0.06,e,parameters)
#x0_2 = optimize.newton(f1,-0.15)
#state0_2 = [x0_2,0.06,0.0,0.0]
#f2 = lambda x: tp.get_coordinate(model,x,-0.05,e,parameters)
#x0_3 = optimize.newton(f2,-0.15)
#state0_3 = [x0_3, -0.05,0.0,0.0]
#
#po_fam_file = open("1111x0_turningpoint_deltaE%s_deleonberne.dat" %(0.1),'a+')
#[x0po_1, T_1,energyPO_1] = tp.turningPoint(model,state0_3,state0_2 ,parameters,e,n,n_turn,po_fam_file)  
#
#po_fam_file.close()
#
#
##%%
## e is the total energy, n is the number of intervals we want to divide, n_turn is the nth turning point we want to choose.
#e=2.0
#n=12
#n_turn = 1
#deltaE = e-parameters[2]
#"""Trial initial Condition s.t. one initial condition is on the LHS of the UPO and the other one is on the RHS of the UPO"""
#f1 = lambda x: tp.get_coordinate(model,x,0.06,e,parameters)
#x0_2 = optimize.newton(f1,-0.15)
#state0_2 = [x0_2,0.06,0.0,0.0]
#f2 = lambda x: tp.get_coordinate(model,x,-0.05,e,parameters)
#x0_3 = optimize.newton(f2,-0.15)
#state0_3 = [x0_3, -0.05,0.0,0.0]
#
#po_fam_file = open("1111x0_turningpoint_deltaE%s_deleonberne.dat" %(deltaE),'a+')
#[x0po_2, T_2,energyPO_2] = tp.turningPoint(model,state0_3,state0_2 ,parameters,e,n,n_turn,po_fam_file)  
#
#po_fam_file.close()
##%%
## e is the total energy, n is the number of intervals we want to divide, n_turn is the nth turning point we want to choose.
#e=3.0
#n=12
#n_turn = 1
#deltaE = e-parameters[2]
#"""Trial initial Condition s.t. one initial condition is on the LHS of the UPO and the other one is on the RHS of the UPO"""
#f1 = lambda x: tp.get_coordinate(model,x,0.06,e,parameters)
#x0_2 = optimize.newton(f1,-0.15)
#state0_2 = [x0_2,0.06,0.0,0.0]
#f2 = lambda x: tp.get_coordinate(model,x,-0.05,e,parameters)
#x0_3 = optimize.newton(f2,-0.15)
#state0_3 = [x0_3, -0.05,0.0,0.0]
#
#po_fam_file = open("1111x0_turningpoint_deltaE%s_deleonberne.dat" %(deltaE),'a+')
#[x0po_3, T_3,energyPO_3] = tp.turningPoint(model,state0_3,state0_2 ,parameters,e,n,n_turn,po_fam_file)  
#
#po_fam_file.close()
##%%
## e is the total energy, n is the number of intervals we want to divide, n_turn is the nth turning point we want to choose.
#e=5.0
#n=12
#n_turn = 1
#deltaE = e-parameters[2]
#"""Trial initial Condition s.t. one initial condition is on the LHS of the UPO and the other one is on the RHS of the UPO"""
#f1 = lambda x: tp.get_coordinate(model,x,0.06,e,parameters)
#x0_2 = optimize.newton(f1,-0.15)
#state0_2 = [x0_2,0.06,0.0,0.0]
#f2 = lambda x: tp.get_coordinate(model,x,-0.05,e,parameters)
#x0_3 = optimize.newton(f2,-0.15)
#state0_3 = [x0_3, -0.05,0.0,0.0]
#
#po_fam_file = open("1111x0_turningpoint_deltaE%s_deleonberne.dat" %(deltaE),'a+')
#[x0po_4, T_4,energyPO_4] = tp.turningPoint(model,state0_3,state0_2 ,parameters,e,n,n_turn,po_fam_file)  
#
#po_fam_file.close()

#%% Load Data
#deltaE = 0.10
#po_fam_file = open("1111x0_turningpoint_deltaE%s_deleonberne.dat" %(deltaE),'a+')
#print('Loading the periodic orbit family from data file',po_fam_file.name,'\n') 
#x0podata = np.loadtxt(po_fam_file.name)
#po_fam_file.close()
#x0po_1 = x0podata
#
#deltaE = 1.0
#po_fam_file = open("1111x0_turningpoint_deltaE%s_deleonberne.dat" %(deltaE),'a+')
#print('Loading the periodic orbit family from data file',po_fam_file.name,'\n') 
#x0podata = np.loadtxt(po_fam_file.name)
#po_fam_file.close()
#x0po_2 = x0podata
#
#deltaE = 2.0
#po_fam_file = open("1111x0_turningpoint_deltaE%s_deleonberne.dat" %(deltaE),'a+')
#print('Loading the periodic orbit family from data file',po_fam_file.name,'\n') 
#x0podata = np.loadtxt(po_fam_file.name)
#po_fam_file.close()
#x0po_3 = x0podata
#
#deltaE = 4.0
#po_fam_file = open("1111x0_turningpoint_deltaE%s_deleonberne.dat" %(deltaE),'a+')
#print('Loading the periodic orbit family from data file',po_fam_file.name,'\n') 
#x0podata = np.loadtxt(po_fam_file.name)
#po_fam_file.close()
#x0po_4 = x0podata

#%% Plotting the Family
#TSPAN = [0,30]
#plt.close('all')
#axis_fs = 15
#RelTol = 3.e-10
#AbsTol = 1.e-10
#f = lambda t,x : tp.Ham2dof(model,t,x,parameters) 
#soln = solve_ivp(f, TSPAN, x0po_1[-1,0:4],method='RK45',dense_output=True, events = lambda t,x : tp.half_period(t,x,model),rtol=RelTol, atol=AbsTol)
#te = soln.t_events[0]
#tt = [0,te[1]]
#t,x,phi_t1,PHI = tp.stateTransitMat(tt,x0po_1[-1,0:4],parameters,model)
#ax = plt.gca(projection='3d')
#ax.plot(x[:,0],x[:,1],x[:,2],'-',color='b',label='$\Delta E$ = 0.1')
#ax.plot(x[:,0],x[:,1],-x[:,2],'-',color='b')
#ax.scatter(x[0,0],x[0,1],x[0,2],s=20,marker='*')
#ax.scatter(x[0,0],x[0,1],-x[0,2],s=20,marker='o')
#ax.plot(x[:,0], x[:,1], zs=0, zdir='z')
#
#
#f = lambda t,x : tp.Ham2dof(model,t,x,parameters) 
#soln = solve_ivp(f, TSPAN, x0po_2[-1,0:4],method='RK45',dense_output=True, events = lambda t,x : tp.half_period(t,x,model),rtol=RelTol, atol=AbsTol)
#te = soln.t_events[0]
#tt = [0,te[1]]
#t,x,phi_t1,PHI = tp.stateTransitMat(tt,x0po_2[-1,0:4],parameters,model)
#ax = plt.gca(projection='3d')
#ax.plot(x[:,0],x[:,1],x[:,2],'-',color='r',label='$\Delta E$ = 1.0')
#ax.plot(x[:,0],x[:,1],-x[:,2],'-',color='r')
#ax.scatter(x[0,0],x[0,1],x[0,2],s=20,marker='*')
#ax.scatter(x[0,0],x[0,1],-x[0,2],s=20,marker='o')
#ax.plot(x[:,0], x[:,1], zs=0, zdir='z')
#
#f = lambda t,x : tp.Ham2dof(model,t,x,parameters) 
#soln = solve_ivp(f, TSPAN, x0po_3[-1,0:4],method='RK45',dense_output=True, events = lambda t,x : tp.half_period(t,x,model),rtol=RelTol, atol=AbsTol)
#te = soln.t_events[0]
#tt = [0,te[1]]
#t,x,phi_t1,PHI = tp.stateTransitMat(tt,x0po_3[-1,0:4],parameters,model)
#ax = plt.gca(projection='3d')
#ax.plot(x[:,0],x[:,1],x[:,2],'-',color='g',label='$\Delta E$ = 2.0')
#ax.plot(x[:,0],x[:,1],-x[:,2],'-',color='g')
#ax.scatter(x[0,0],x[0,1],x[0,2],s=20,marker='*')
#ax.scatter(x[0,0],x[0,1],-x[0,2],s=20,marker='o')
#ax.plot(x[:,0], x[:,1], zs=0, zdir='z')
#
#
#f = lambda t,x : tp.Ham2dof(model,t,x,parameters) 
#soln = solve_ivp(f, TSPAN, x0po_4[-1,0:4],method='RK45',dense_output=True, events = lambda t,x : tp.half_period(t,x,model),rtol=RelTol, atol=AbsTol)
#te = soln.t_events[0]
#tt = [0,te[1]]
#t,x,phi_t1,PHI = tp.stateTransitMat(tt,x0po_4[-1,0:4],parameters,model)
#ax = plt.gca(projection='3d')
#ax.plot(x[:,0],x[:,1],x[:,2],'-',color='m',label='$\Delta E$ = 4.0')
#ax.plot(x[:,0],x[:,1],-x[:,2],'-',color='m')
#ax.scatter(x[0,0],x[0,1],x[0,2],s=20,marker='*')
#ax.scatter(x[0,0],x[0,1],-x[0,2],s=20,marker='o')
#ax.plot(x[:,0], x[:,1], zs=0, zdir='z')
#
#ax = plt.gca(projection='3d')
#resX = 100
#xVec = np.linspace(-1,2,resX)
#yVec = np.linspace(-2,2,resX)
#xMat, yMat = np.meshgrid(xVec, yVec)
#cset1 = ax.contour(xMat, yMat, tp.get_pot_surf_proj(model,xVec, yVec,parameters), [1.1,2,3,5],zdir='z', offset=0,
#                       linewidths = 1.0, cmap=cm.viridis, alpha = 0.8)
#
#ax.scatter(eqPt[0], eqPt[1], s = 200, c = 'r', marker = 'X')
#ax.set_xlabel('$x$', fontsize=axis_fs)
#ax.set_ylabel('$y$', fontsize=axis_fs)
#ax.set_zlabel('$p_x$', fontsize=axis_fs)
##ax.set_title('$\Delta E$ = %1.e,%1.e,%1.e,%1.e,%1.e' %(energyPO_1[-1]-parameters[2],energyPO_2[-1]-parameters[2],energyPO_3[-1]-parameters[2],energyPO_4[-1]-parameters[2],energyPO_5[-1]-parameters[2]) ,fontsize=axis_fs)
#ax.set_xlim(-1.5, 1.5)
#ax.set_ylim(-1.5, 1.5)
#ax.set_zlim(-4, 4)
#legend = ax.legend(loc='upper left')
#
#plt.grid()
#plt.show()
#
#plt.savefig('turningpoint_POfam_deleonberne.pdf',format='pdf',bbox_inches='tight')

