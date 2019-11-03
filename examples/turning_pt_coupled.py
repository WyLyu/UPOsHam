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

import matplotlib as mpl
from matplotlib import cm
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'

# This needs testing for installation 
import sys
sys.path.insert(0, './src/')
import turning_point as tp
import coupled_quartic_hamiltonian as coupled
# This needs testing for installation


#%% Setting up parameters and global variables

save_final_plot = True
show_final_plot = False
show_itrsteps_plots = False # show iteration of the UPOs in plots
N = 4          # dimension of phase space
omega=1.00
EPSILON_S = 0.0 #Energy of the saddle
alpha = 1.00
beta = 1.00
epsilon= 1e-1
parameters = np.array([1,omega, EPSILON_S, alpha, beta,omega,epsilon])
eqNum = 1  
#model = 'coupled'
#eqPt = tp.get_eq_pts(eqNum,model, parameters)
eqPt = tp.get_eq_pts(eqNum, coupled.init_guess_eqpt_coupled, \
                            coupled.grad_pot_coupled, parameters)

#%%
#deltaE_vals = [0.01, 0.1, 1.00, 2.0, 4.0]
#xLeft = [0.0,-0.05,0.01,0.01,0.18]
#xRight = [0.05,0.10,0.18,0.18,0.18]
#linecolor = ['b','r','g','m','c']
deltaE_vals = [0.1, 1.00]
xLeft = [0.0,0.01]
xRight = [0.05,0.11]
linecolor = ['b','r']

"""
e is the total energy, 
n is the number of intervals we want to divide, 
n_turn is the nth turning point we want to choose.
"""
for i in range(len(deltaE_vals)):
    
    e = deltaE_vals[i]
    n = 12
    n_turn = 1
    deltaE = e - parameters[2]
    
    """
    Trial initial Condition s.t. one initial condition is on the LHS of the UPO and the 
    other one is on the RHS of the UPO
    """
    x = xLeft[i]
    f2 = lambda y: coupled.get_coord_coupled(x, y, e, parameters)
    yanalytic = math.sqrt(2/(parameters[1]+parameters[6]))*(-math.sqrt( e + \
                         0.5*parameters[3]* x**2- 0.25*parameters[4]*x**4 - 0.5*parameters[6]*x**2 + \
                         (parameters[6]*x)**2/(2*(parameters[1] + parameters[6]) )) + \
                            parameters[6]/(math.sqrt(2*(parameters[1]+parameters[6])) )*x ) #coupled
    state0_2 = [x,optimize.newton(f2,yanalytic),0.0,0.0]
    
    x = xRight[i]
    f3 = lambda y: coupled.get_coord_coupled(x, y, e, parameters)
    yanalytic = math.sqrt(2/(parameters[1]+parameters[6]))*(-math.sqrt( e + \
                         0.5*parameters[3]* x**2- 0.25*parameters[4]*x**4 -0.5*parameters[6]*x**2 + \
                         (parameters[6]*x)**2/(2*(parameters[1] + parameters[6]) )) + \
                            parameters[6]/(math.sqrt(2*(parameters[1]+parameters[6])) )*x ) #coupled
    state0_3 = [x, optimize.newton(f3,yanalytic),0.0,0.0]
    
    
    po_fam_file = open("x0_turningpoint_deltaE%s_coupled.txt" %(deltaE),'a+')
    [x0po_1, T_1,energyPO_1] = tp.turningPoint(state0_2, state0_3, \
                                        coupled.get_coord_coupled, \
                                        coupled.guess_coords_coupled, \
                                        coupled.ham2dof_coupled, \
                                        coupled.half_period_coupled, \
                                        coupled.varEqns_coupled, \
                                        coupled.pot_energy_coupled, \
                                        coupled.plot_iter_orbit_coupled, \
                                        parameters, \
                                        e, n, n_turn, show_itrsteps_plots, \
                                        po_fam_file) 
    po_fam_file.close()



#[x0po_1, T_1,energyPO_1] = tp.turningPoint(model,state0_2,state0_3 ,parameters,e,n,n_turn,po_fam_file)  

#%% Load periodic orbit data from ascii files
    
x0po = np.zeros((4,len(deltaE_vals))) #each column is a different initial condition

for i in range(len(deltaE_vals)):
    deltaE = deltaE_vals[i]

    po_fam_file = open("x0_turningpoint_deltaE%s_coupled.txt" %(deltaE),'a+')
    print('Loading the periodic orbit family from data file',po_fam_file.name,'\n') 
    x0podata = np.loadtxt(po_fam_file.name)
    po_fam_file.close()
    x0po[:,i] = x0podata[-1,0:4] 


#%%

#%% Plotting the Family

TSPAN = [0,30]
plt.close('all')
axis_fs = 15
RelTol = 3.e-10
AbsTol = 1.e-10

f = lambda t,x : coupled.ham2dof_coupled(t,x,parameters) 

ax = plt.gca(projection='3d')

for i in range(len(deltaE_vals)):
    
    soln = solve_ivp(f, TSPAN, x0po[:,i], method='RK45', dense_output=True, \
                    events = lambda t,x : coupled.half_period_coupled(t,x,parameters), \
                    rtol=RelTol, atol=AbsTol)
    te = soln.t_events[0]
    tt = [0,te[2]]
    t,x,phi_t1,PHI = tp.stateTransitMat(tt, x0po[:,i], parameters, \
                    coupled.varEqns_coupled)
    
    
    ax.plot(x[:,0],x[:,1],x[:,3],'-', color=linecolor[i], \
            label='$\Delta E$ = %.2f'%(deltaE_vals[i]))
    ax.scatter(x[0,0],x[0,1],x[0,3], s=20, marker='*')
    ax.plot(x[:,0], x[:,1], zs=0, zdir='z')


resX = 100
xVec = np.linspace(-4,4,resX)
yVec = np.linspace(-4,4,resX)
xMat, yMat = np.meshgrid(xVec, yVec)
cset1 = ax.contour(xMat, yMat, 
                   tp.get_pot_surf_proj(xVec, yVec, \
                        coupled.pot_energy_coupled, parameters), \
                        [0.01,0.1,1,2,4], zdir='z', offset=0, \
                        linewidths = 1.0, cmap=cm.viridis, \
                        alpha = 0.8)

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


if show_final_plot:
    plt.show()

if save_final_plot:  
    plt.savefig('./tests/plots/tp_coupled_upos.pdf', format='pdf', \
                bbox_inches='tight')



#%%
#e=0.1
#n=12
#n_turn = 1
#deltaE = e-parameters[2]
#"""Trial initial Condition s.t. one initial condition is on the LHS of the UPO and the other one is on the RHS of the UPO"""
#x=-0.05
#f2 = lambda y: tp.get_coordinate(model,0.0,y,e,parameters)
#yanalytic = math.sqrt(2/(parameters[1]+parameters[6]))*(-math.sqrt( e +0.5*parameters[3]* x**2- 0.25*parameters[4]*x**4 -0.5*parameters[6]* x**2 + (parameters[6]*x)**2/(2*(parameters[1] +parameters[6]) )) +parameters[6]/(math.sqrt(2*(parameters[1]+parameters[6])) )*x ) #coupled
#state0_2 = [0.0,optimize.newton(f2,yanalytic),0.0,0.0]
#
#x=0.05
#f3 = lambda y: tp.get_coordinate(model,x,y,e,parameters)
#yanalytic = math.sqrt(2/(parameters[1]+parameters[6]))*(-math.sqrt( e +0.5*parameters[3]* x**2- 0.25*parameters[4]*x**4 -0.5*parameters[6]* x**2 + (parameters[6]*x)**2/(2*(parameters[1] +parameters[6]) )) +parameters[6]/(math.sqrt(2*(parameters[1]+parameters[6])) )*x ) #coupled
#state0_3 = [x, optimize.newton(f3,yanalytic),0.0,0.0]
#
#
#po_fam_file = open("1111x0_turningpoint_deltaE%s_coupled.txt" %(deltaE),'a+')
#[x0po_2, T_2,energyPO_2] = tp.turningPoint(model,state0_2,state0_3 ,parameters,e,n,n_turn,po_fam_file)  
#po_fam_file.close()
#
##%%
#e=1.0
#n=12
#n_turn = 1
#deltaE = e-parameters[2]
#"""Trial initial Condition s.t. one initial condition is on the LHS of the UPO and the other one is on the RHS of the UPO"""
#x=0.01
#f2 = lambda y: tp.get_coordinate(model,0.0,y,e,parameters)
#yanalytic = math.sqrt(2/(parameters[1]+parameters[6]))*(-math.sqrt( e +0.5*parameters[3]* x**2- 0.25*parameters[4]*x**4 -0.5*parameters[6]* x**2 + (parameters[6]*x)**2/(2*(parameters[1] +parameters[6]) )) +parameters[6]/(math.sqrt(2*(parameters[1]+parameters[6])) )*x ) #coupled
#state0_2 = [0.0,optimize.newton(f2,yanalytic),0.0,0.0]
#
#x=0.11
#f3 = lambda y: tp.get_coordinate(model,x,y,e,parameters)
#yanalytic = math.sqrt(2/(parameters[1]+parameters[6]))*(-math.sqrt( e +0.5*parameters[3]* x**2- 0.25*parameters[4]*x**4 -0.5*parameters[6]* x**2 + (parameters[6]*x)**2/(2*(parameters[1] +parameters[6]) )) +parameters[6]/(math.sqrt(2*(parameters[1]+parameters[6])) )*x ) #coupled
#state0_3 = [x, optimize.newton(f3,yanalytic),0.0,0.0]
#
#
#po_fam_file = open("1111x0_turningpoint_deltaE%s_coupled.txt" %(deltaE),'a+')
#[x0po_3, T_3,energyPO_3] = tp.turningPoint(model,state0_2,state0_3 ,parameters,e,n,n_turn,po_fam_file)  
#po_fam_file.close()
#
##%%
#e=2.0
#n=12
#n_turn = 1
#deltaE = e-parameters[2]
#"""Trial initial Condition s.t. one initial condition is on the LHS of the UPO and the other one is on the RHS of the UPO"""
#x=0.05
#f2 = lambda y: tp.get_coordinate(model,0.0,y,e,parameters)
#yanalytic = math.sqrt(2/(parameters[1]+parameters[6]))*(-math.sqrt( e +0.5*parameters[3]* x**2- 0.25*parameters[4]*x**4 -0.5*parameters[6]* x**2 + (parameters[6]*x)**2/(2*(parameters[1] +parameters[6]) )) +parameters[6]/(math.sqrt(2*(parameters[1]+parameters[6])) )*x ) #coupled
#state0_2 = [0.0,optimize.newton(f2,yanalytic),0.0,0.0]
#
#x=0.15
#f3 = lambda y: tp.get_coordinate(model,x,y,e,parameters)
#yanalytic = math.sqrt(2/(parameters[1]+parameters[6]))*(-math.sqrt( e +0.5*parameters[3]* x**2- 0.25*parameters[4]*x**4 -0.5*parameters[6]* x**2 + (parameters[6]*x)**2/(2*(parameters[1] +parameters[6]) )) +parameters[6]/(math.sqrt(2*(parameters[1]+parameters[6])) )*x ) #coupled
#state0_3 = [x, optimize.newton(f3,yanalytic),0.0,0.0]
#
#
#po_fam_file = open("1111x0_turningpoint_deltaE%s_coupled.txt" %(deltaE),'a+')
#[x0po_4, T_4,energyPO_4] = tp.turningPoint(model,state0_2,state0_3 ,parameters,e,n,n_turn,po_fam_file)  
#po_fam_file.close()
#
##%%
#e=4.0
#n=12
#n_turn = 1
#deltaE = e-parameters[2]
#"""Trial initial Condition s.t. one initial condition is on the LHS of the UPO and the other one is on the RHS of the UPO"""
#x=0.08
#f2 = lambda y: tp.get_coordinate(model,0.0,y,e,parameters)
#yanalytic = math.sqrt(2/(parameters[1]+parameters[6]))*(-math.sqrt( e +0.5*parameters[3]* x**2- 0.25*parameters[4]*x**4 -0.5*parameters[6]* x**2 + (parameters[6]*x)**2/(2*(parameters[1] +parameters[6]) )) +parameters[6]/(math.sqrt(2*(parameters[1]+parameters[6])) )*x ) #coupled
#state0_2 = [0.0,optimize.newton(f2,yanalytic),0.0,0.0]
#
#x=0.18
#f3 = lambda y: tp.get_coordinate(model,x,y,e,parameters)
#yanalytic = math.sqrt(2/(parameters[1]+parameters[6]))*(-math.sqrt( e +0.5*parameters[3]* x**2- 0.25*parameters[4]*x**4 -0.5*parameters[6]* x**2 + (parameters[6]*x)**2/(2*(parameters[1] +parameters[6]) )) +parameters[6]/(math.sqrt(2*(parameters[1]+parameters[6])) )*x ) #coupled
#state0_3 = [x, optimize.newton(f3,yanalytic),0.0,0.0]
#
#
#po_fam_file = open("1111x0_turningpoint_deltaE%s_coupled.txt" %(deltaE),'a+')
#[x0po_5, T_5,energyPO_5] = tp.turningPoint(model,state0_2,state0_3 ,parameters,e,n,n_turn,po_fam_file)  
#po_fam_file.close()


#%% Load Data
#deltaE = 0.010
#po_fam_file = open("1111x0_turningpoint_deltaE%s_coupled.txt" %(deltaE),'a+')
#print('Loading the periodic orbit family from data file',po_fam_file.name,'\n') 
#x0podata = np.loadtxt(po_fam_file.name)
#po_fam_file.close()
#x0po_1 = x0podata
#
#deltaE = 0.10
#po_fam_file = open("1111x0_turningpoint_deltaE%s_coupled.txt" %(deltaE),'a+')
#print('Loading the periodic orbit family from data file',po_fam_file.name,'\n') 
#x0podata = np.loadtxt(po_fam_file.name)
#po_fam_file.close()
#x0po_2 = x0podata
#
#deltaE = 1.0
#po_fam_file = open("1111x0_turningpoint_deltaE%s_coupled.txt" %(deltaE),'a+')
#print('Loading the periodic orbit family from data file',po_fam_file.name,'\n') 
#x0podata = np.loadtxt(po_fam_file.name)
#po_fam_file.close()
#x0po_3 = x0podata
#
#deltaE = 2.0
#po_fam_file = open("1111x0_turningpoint_deltaE%s_coupled.txt" %(deltaE),'a+')
#print('Loading the periodic orbit family from data file',po_fam_file.name,'\n') 
#x0podata = np.loadtxt(po_fam_file.name)
#po_fam_file.close()
#x0po_4 = x0podata
#
#deltaE = 4.0
#po_fam_file = open("1111x0_turningpoint_deltaE%s_coupled.txt" %(deltaE),'a+')
#print('Loading the periodic orbit family from data file',po_fam_file.name,'\n') 
#x0podata = np.loadtxt(po_fam_file.name)
#po_fam_file.close()
#x0po_5 = x0podata




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
#ax.plot(x[:,0],x[:,1],x[:,3],'-',color='b',label='$\Delta E$ = 0.01')
#ax.plot(x[:,0],x[:,1],-x[:,3],'-',color='b')
#ax.scatter(x[0,0],x[0,1],x[0,3],s=20,marker='*')
#ax.scatter(x[0,0],x[0,1],-x[0,3],s=20,marker='o')
#ax.plot(x[:,0], x[:,1], zs=0, zdir='z')
#
#f = lambda t,x : tp.Ham2dof(model,t,x,parameters) 
#soln = solve_ivp(f, TSPAN, x0po_2[-1,0:4],method='RK45',dense_output=True, events = lambda t,x : tp.half_period(t,x,model),rtol=RelTol, atol=AbsTol)
#te = soln.t_events[0]
#tt = [0,te[1]]
#t,x,phi_t1,PHI = tp.stateTransitMat(tt,x0po_2[-1,0:4],parameters,model)
#ax = plt.gca(projection='3d')
#ax.plot(x[:,0],x[:,1],x[:,3],'-',color='r',label='$\Delta E$ = 0.1')
#ax.plot(x[:,0],x[:,1],-x[:,3],'-',color='r')
#ax.scatter(x[0,0],x[0,1],x[0,3],s=20,marker='*')
#ax.scatter(x[0,0],x[0,1],-x[0,3],s=20,marker='o')
#ax.plot(x[:,0], x[:,1], zs=0, zdir='z')
#
#f = lambda t,x : tp.Ham2dof(model,t,x,parameters) 
#soln = solve_ivp(f, TSPAN, x0po_3[-1,0:4],method='RK45',dense_output=True, events = lambda t,x : tp.half_period(t,x,model),rtol=RelTol, atol=AbsTol)
#te = soln.t_events[0]
#tt = [0,te[1]]
#t,x,phi_t1,PHI = tp.stateTransitMat(tt,x0po_3[-1,0:4],parameters,model)
#ax = plt.gca(projection='3d')
#ax.plot(x[:,0],x[:,1],x[:,3],'-',color='g',label='$\Delta E$ = 1.0')
#ax.plot(x[:,0],x[:,1],-x[:,3],'-',color='g')
#ax.scatter(x[0,0],x[0,1],x[0,3],s=20,marker='*')
#ax.scatter(x[0,0],x[0,1],-x[0,3],s=20,marker='o')
#ax.plot(x[:,0], x[:,1], zs=0, zdir='z')
#
#
#f = lambda t,x : tp.Ham2dof(model,t,x,parameters) 
#soln = solve_ivp(f, TSPAN, x0po_4[-1,0:4],method='RK45',dense_output=True, events = lambda t,x : tp.half_period(t,x,model),rtol=RelTol, atol=AbsTol)
#te = soln.t_events[0]
#tt = [0,te[1]]
#t,x,phi_t1,PHI = tp.stateTransitMat(tt,x0po_4[-1,0:4],parameters,model)
#ax = plt.gca(projection='3d')
#ax.plot(x[:,0],x[:,1],x[:,3],'-',color='m',label='$\Delta E$ = 2.0')
#ax.plot(x[:,0],x[:,1],-x[:,3],'-',color='m')
#ax.scatter(x[0,0],x[0,1],x[0,3],s=20,marker='*')
#ax.scatter(x[0,0],x[0,1],-x[0,3],s=20,marker='o')
#ax.plot(x[:,0], x[:,1], zs=0, zdir='z')
#
#
#f = lambda t,x : tp.Ham2dof(model,t,x,parameters) 
#soln = solve_ivp(f, TSPAN, x0po_5[-1,0:4],method='RK45',dense_output=True, events = lambda t,x : tp.half_period(t,x,model),rtol=RelTol, atol=AbsTol)
#te = soln.t_events[0]
#tt = [0,te[1]]
#t,x,phi_t1,PHI = tp.stateTransitMat(tt,x0po_5[-1,0:4],parameters,model)
#ax = plt.gca(projection='3d')
#ax.plot(x[:,0],x[:,1],x[:,3],'-',color='c',label='$\Delta E$ = 4.0')
#ax.plot(x[:,0],x[:,1],-x[:,3],'-',color='c')
#ax.scatter(x[0,0],x[0,1],x[0,3],s=20,marker='*')
#ax.scatter(x[0,0],x[0,1],-x[0,3],s=20,marker='o')
#ax.plot(x[:,0], x[:,1], zs=0, zdir='z')
#
#
#ax = plt.gca(projection='3d')
#resX = 100
#xVec = np.linspace(-4,4,resX)
#yVec = np.linspace(-4,4,resX)
#xMat, yMat = np.meshgrid(xVec, yVec)
#cset1 = ax.contour(xMat, yMat, tp.get_pot_surf_proj(model,xVec, yVec,parameters), [0.01,0.1,1,2,4],zdir='z', offset=0,
#                       linewidths = 1.0, cmap=cm.viridis, alpha = 0.8)
#
#ax.scatter(eqPt[0], eqPt[1], s = 200, c = 'r', marker = 'X')
#ax.set_xlabel('$x$', fontsize=axis_fs)
#ax.set_ylabel('$y$', fontsize=axis_fs)
#ax.set_zlabel('$p_y$', fontsize=axis_fs)
##ax.set_title('$\Delta E$ = %1.e,%1.e,%1.e,%1.e,%1.e' %(energyPO_1[-1],energyPO_2[-1],energyPO_3[-1],energyPO_4[-1],energyPO_5[-1]) ,fontsize=axis_fs)
#legend = ax.legend(loc='best')
#ax.set_xlim(-4, 4)
#ax.set_ylim(-4, 4)
#ax.set_zlim(-4, 4)
#
#plt.grid()
#plt.show()
#
#plt.savefig('turningpoint_POfam_coupled.pdf',format='pdf',bbox_inches='tight')



