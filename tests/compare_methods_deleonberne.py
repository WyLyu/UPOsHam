# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 16:53:45 2019


@author: Wenyang Lyu and Shibabrat Naik
"""

import numpy as np
import matplotlib.pyplot as plt
#from scipy.integrate import odeint,quad,trapz,solve_ivp
from scipy.integrate import solve_ivp
#import math
#from IPython.display import Image # for Notebook
#from IPython.core.display import HTML
from mpl_toolkits.mplot3d import Axes3D
#from numpy import linalg as LA
#import scipy.linalg as linalg
#from scipy.optimize import fsolve
#import time
#from functools import partial
import sys
sys.path.append('./src/')
import turning_point_coord_difference ### import module xxx where xxx is the name of the python file xxx.py 
import differential_correction


#from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from matplotlib import cm
#from pylab import rcParams
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'

#from scipy import optimize


#% Begin problem specific functions
def init_guess_eqpt_deleonberne(eqNum, par):
    """
    Returns configuration space coordinates of the equilibrium points according to the index:
    Saddle (EQNUM=1)
    Centre (EQNUM=2,3)
    """
    
    if eqNum == 1:
        x0 = [0, 0]
    elif eqNum == 2:
        x0 = [0, 1/np.sqrt(2)]  # EQNUM = 2, center-center
    elif eqNum == 3:
        x0 = [0, -1/np.sqrt(2)] # EQNUM = 3, center-center
    
    return x0


def grad_pot_deleonberne(x, par):
    """Returns the gradient of the potential energy function V(x,y)
    """ 
    
    dVdx = -2*par[3]*par[4]*np.exp(-par[4]*x[0])*(np.exp(-par[4]*x[0]) - 1) - \
        4*par[5]*par[4]*x[1]**2*(x[1]**2 - 1)*np.exp(-par[5]*par[4]*x[0])
    dVdy = 8*x[1]*(2*x[1]**2 - 1)*np.exp(-par[5]*par[4]*x[0])
    
    F = [-dVdx, -dVdy]
    
    return F

def pot_energy_deleonberne(x, y, par):
    """Returns the potential energy function V(x,y)
    """
    
    return par[3]*( 1 - np.exp(-par[4]*x) )**2 + 4*y**2*(y**2 - 1)*np.exp(-par[5]*par[4]*x) + par[2]


def varEqns_deleonberne(t,PHI,par):
    """    
    Returns the state transition matrix , PHI(t,t0), where Df(t) is the Jacobian of the 
    Hamiltonian vector field
    
    d PHI(t, t0)
    ------------ =  Df(t) * PHI(t, t0)
        dt
    
    """
    
    phi = PHI[0:16]
    phimatrix  = np.reshape(PHI[0:16],(4,4))
    x,y,px,py = PHI[16:20]
    
    
    # The first order derivative of the potential energy.
    dVdx = - 2*par[3]*par[4]*np.exp(-par[4]*x)*(np.exp(-par[4]*x) - 1) - 4*par[5]*par[4]*y**2*(y**2 - 1)*np.exp(-par[5]*par[4]*x) 
    dVdy = 8*y*(2*y**2 - 1)*np.exp(-par[5]*par[4]*x)

    # The second order derivative of the potential energy. 
    d2Vdx2 = - ( 2*par[3]*par[4]**2*( np.exp(-par[4]*x) - 2.0*np.exp(-2*par[4]*x) ) - 4*(par[5]*par[4])**2*x**2*(y**2 - 1)*np.exp(-par[5]*par[4]*x) )
        
    d2Vdy2 = 8*(6*y**2 - 1)*np.exp( -par[4]*par[5]*x )

    d2Vdydx = -8*y*par[4]*par[5]*np.exp( -par[4]*par[5]*x )*(2*y**2 - 1)
        
    
    
    d2Vdxdy = d2Vdydx    

    Df    = np.array([[  0,     0,    par[0],    0],
              [0,     0,    0,    par[1]],
              [-d2Vdx2,  -d2Vdydx,   0,    0],
              [-d2Vdxdy, -d2Vdy2,    0,    0]])

    
    phidot = np.matmul(Df, phimatrix) # variational equation

    PHIdot        = np.zeros(20)
    PHIdot[0:16]  = np.reshape(phidot,(1,16)) 
    PHIdot[16]    = px/par[0]
    PHIdot[17]    = py/par[1]
    PHIdot[18]    = -dVdx 
    PHIdot[19]    = -dVdy
    
    return list(PHIdot)


def ham2dof_deleonberne(t, x, par):
    """ 
    Returns the Hamiltonian vector field (Hamilton's equations of motion) 
    """
    
    xDot = np.zeros(4)
    
    dVdx = -2*par[3]*par[4]*np.exp(-par[4]*x[0])*(np.exp(-par[4]*x[0]) - 1) - \
        4*par[5]*par[4]*x[1]**2*(x[1]**2 - 1)*np.exp(-par[5]*par[4]*x[0])
    dVdy = 8*x[1]*(2*x[1]**2 - 1)*np.exp(-par[5]*par[4]*x[0])
        
    xDot[0] = x[2]/par[0]
    xDot[1] = x[3]/par[1]
    xDot[2] = -dVdx 
    xDot[3] = -dVdy
    
    return list(xDot)  



def half_period_deleonberne(t,x,par):
    """
    Return the turning point where we want to stop the integration                           
    
    pxDot = x[0]
    pyDot = x[1]
    xDot = x[2]
    yDot = x[3]
    """
    
    terminal = True
    # The zero can be approached from either direction
    direction = 0 #0: all directions of crossing
    
    return x[2]
        

#% End problem specific functions


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
#model = 'deleonberne'
#eqPt = turning_point_coord_difference.get_eq_pts(eqNum, model,parameters)

#eSaddle = turning_point_coord_difference.get_total_energy(model,[eqPt[0],eqPt[1],0,0], parameters) #energy of the saddle eq pt

eqPt = differential_correction.get_eq_pts(eqNum, init_guess_eqpt_deleonberne, \
                                       grad_pot_deleonberne, parameters)

#energy of the saddle eq pt
eSaddle = differential_correction.get_total_energy([eqPt[0],eqPt[1],0,0], pot_energy_deleonberne, \
                                                parameters)


#%% Load Data
deltaE = 1.0
eSaddle = 1.0 # energy of the saddle

data_path = "./data/"
#po_fam_file = open("1111x0_tpcd_deltaE%s_deleonberne.txt" %(deltaE),'a+')
po_fam_file = "x0_tpcd_deltaE%s_deleonberne.txt" %(deltaE)
print('Loading the periodic orbit family from data file',po_fam_file,'\n') 
#x0podata = np.loadtxt(po_fam_file.name)
#po_fam_file.close()

x0podata = np.loadtxt(data_path + po_fam_file)

x0po_1_tpcd = x0podata


#po_fam_file = open("1111x0_turningpoint_deltaE%s_deleonberne.txt" %(deltaE),'a+')
po_fam_file = "x0_turningpoint_deltaE%s_deleonberne.txt" %(deltaE)
print('Loading the periodic orbit family from data file',po_fam_file,'\n') 
#x0podata = np.loadtxt(po_fam_file.name)
#po_fam_file.close()
x0podata = np.loadtxt(data_path + po_fam_file)

x0po_1_turningpoint = x0podata


#po_fam_file = open("1111x0_diffcorr_deltaE%s_deleonberne.txt" %(deltaE),'a+')
po_fam_file = "x0_diffcorr_deltaE%s_deleonberne.txt" %(deltaE)

print('Loading the periodic orbit family from data file',po_fam_file,'\n') 
#x0podata = np.loadtxt(po_fam_file.name)
#po_fam_file.close()
x0podata = np.loadtxt(data_path + po_fam_file)

x0po_1_diffcorr = x0podata[0:4]


#%%
TSPAN = [0,30]
plt.close('all')
axis_fs = 15
RelTol = 3.e-10
AbsTol = 1.e-10

#f = lambda t,x : turning_point_coord_difference.Ham2dof(model,t,x,parameters)  
#soln = solve_ivp(f, TSPAN, x0po_1_tpcd[-1,0:4],method='RK45',dense_output=True, events = lambda t,x : turning_point_coord_difference.half_period(t,x,model),rtol=RelTol, atol=AbsTol)
f= lambda t,x: ham2dof_deleonberne(t,x,parameters)
soln = solve_ivp(f, TSPAN, x0po_1_tpcd[-1,0:4],method='RK45',dense_output=True, \
                 events = lambda t,x : half_period_deleonberne(t,x,parameters), \
                 rtol=RelTol, atol=AbsTol)
te = soln.t_events[0]
tt = [0,te[2]]
#t,x,phi_t1,PHI = turning_point_coord_difference.stateTransitMat(tt,x0po_1_tpcd[-1,0:4],parameters,model)
t,x,phi_t1,PHI = differential_correction.stateTransitMat(tt, x0po_1_tpcd[-1,0:4], parameters, \
                                                      varEqns_deleonberne)

ax = plt.gca(projection='3d')
ax.plot(x[:,0],x[:,1],x[:,2],':',label='$\Delta E$ = 0.1, using tpcd')
ax.scatter(x[0,0],x[0,1],x[0,2],s=20,marker='*')
ax.plot(x[:,0], x[:,1], zs=0, zdir='z')

#f = lambda t,x : turning_point_coord_difference.Ham2dof(model,t,x,parameters)  
#soln = solve_ivp(f, TSPAN, x0po_1_diffcorr,method='RK45',dense_output=True, events = lambda t,x : turning_point_coord_difference.half_period(t,x,model),rtol=RelTol, atol=AbsTol)
f= lambda t,x: ham2dof_deleonberne(t,x,parameters)
soln = solve_ivp(f, TSPAN, x0po_1_diffcorr,method='RK45',dense_output=True, \
                 events = lambda t,x : half_period_deleonberne(t,x,parameters), \
                 rtol=RelTol, atol=AbsTol)
te = soln.t_events[0]
tt = [0,te[2]]
#t,x,phi_t1,PHI = turning_point_coord_difference.stateTransitMat(tt,x0po_1_diffcorr,parameters,model)
t,x,phi_t1,PHI = differential_correction.stateTransitMat(tt, x0po_1_diffcorr, parameters, \
                                                      varEqns_deleonberne)

#ax = plt.gca(projection='3d')
ax.plot(x[:,0],x[:,1],x[:,2],'-',label='$\Delta E$ = 0.1, using dcnc')
ax.scatter(x[0,0],x[0,1],x[0,2],s=20,marker='*')
ax.plot(x[:,0], x[:,1], zs=0, zdir='z')

#f = lambda t,x : turning_point_coord_difference.Ham2dof(model,t,x,parameters)  
#soln = solve_ivp(f, TSPAN, x0po_1_turningpoint[-1,0:4],method='RK45',dense_output=True, events = lambda t,x : turning_point_coord_difference.half_period(t,x,model),rtol=RelTol, atol=AbsTol)
f= lambda t,x: ham2dof_deleonberne(t,x,parameters)
soln = solve_ivp(f, TSPAN, x0po_1_turningpoint[-1,0:4],method='RK45',dense_output=True, \
                 events = lambda t,x : half_period_deleonberne(t,x,parameters), \
                 rtol=RelTol, atol=AbsTol)
te = soln.t_events[0]
tt = [0,te[2]]
#t,x,phi_t1,PHI = turning_point_coord_difference.stateTransitMat(tt,x0po_1_turningpoint[-1,0:4],parameters,model)
t,x,phi_t1,PHI = differential_correction.stateTransitMat(tt, x0po_1_turningpoint[-1,0:4], parameters, \
                                                      varEqns_deleonberne)

#ax = plt.gca(projection='3d')
ax.plot(x[:,0],x[:,1],x[:,2],'-.',label='$\Delta E$ = 0.1, using tp')
ax.scatter(x[0,0],x[0,1],x[0,2],s=20,marker='*')
ax.plot(x[:,0], x[:,1], zs=0, zdir='z')



ax = plt.gca(projection='3d')
resX = 100
xVec = np.linspace(-1,1,resX)
yVec = np.linspace(-2,2,resX)
xMat, yMat = np.meshgrid(xVec, yVec)
#cset1 = ax.contour(xMat, yMat, uncoupled_tpcd.get_pot_surf_proj(xVec, yVec,parameters), [0.001,0.1,1,2,4],
#                       linewidths = 1.0, cmap=cm.viridis, alpha = 0.8)
cset2 = ax.contour(xMat, yMat, \
                   differential_correction.get_pot_surf_proj(xVec, yVec, pot_energy_deleonberne, \
                                                          parameters), 2.0, zdir='z', offset=0,
                       linewidths = 1.0, cmap=cm.viridis, alpha = 0.8)
ax.scatter(eqPt[0], eqPt[1], s = 100, c = 'r', marker = 'X')
ax.set_xlabel('$x$', fontsize=axis_fs)
ax.set_ylabel('$y$', fontsize=axis_fs)
ax.set_zlabel('$p_x$', fontsize=axis_fs)
legend = ax.legend(loc='upper left')

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-4, 4)
plt.grid()
plt.show()

plt.savefig('comparison_deleonberne.pdf',format='pdf',bbox_inches='tight')

