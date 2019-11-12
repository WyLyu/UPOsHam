# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 10:02:48 2019
@author: Wenyang Lyu and Shibabrat Naik
Script to define expressions for the coupled quartic Hamiltonian
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time
import math
from scipy import optimize


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
    """ Returns the gradient of the potential energy function V(x,y) """
     
    dVdx = -2*par[3]*par[4]*np.exp(-par[4]*x[0])*(np.exp(-par[4]*x[0]) - 1) - \
        4*par[5]*par[4]*x[1]**2*(x[1]**2 - 1)*np.exp(-par[5]*par[4]*x[0])
    dVdy = 8*x[1]*(2*x[1]**2 - 1)*np.exp(-par[5]*par[4]*x[0])
    
    F = [-dVdx, -dVdy]
    
    return F

def pot_energy_deleonberne(x, y, par):
    """ Returns the potential energy function V(x,y) """
    
    return par[3]*( 1 - np.exp(-par[4]*x) )**2 + 4*y**2*(y**2 - 1)*np.exp(-par[5]*par[4]*x) + par[2]


def eigvector_deleonberne(par):
    """ Returns the correction factor to the eigenvectors for the linear guess """

    correcx = 1
    correcy = 0
    
    return correcx, correcy


def guess_lin_deleonberne(eqPt, Ax, par):
    """ Returns an initial guess for the unstable periodic orbit """
    
    correcx, correcy = eigvector_deleonberne(par)
    

    return [eqPt[0] - Ax*correcx,eqPt[1] + Ax*correcy,0,0]


def jacobian_deleonberne(eqPt, par):
    """ Returns Jacobian of the Hamiltonian vector field """
    
    x,y,px,py = eqPt[0:4]
    # The first order derivative of the Hamiltonian.
    dVdx = - 2*par[3]*par[4]*np.exp(-par[4]*x)*(np.exp(-par[4]*x) - 1) - 4*par[5]*par[4]*y**2*(y**2 - 1)*np.exp(-par[5]*par[4]*x) 
    dVdy = 8*y*(2*y**2 - 1)*np.exp(-par[5]*par[4]*x)

    # The following is the Jacobian matrix 
    d2Vdx2 = - ( 2*par[3]*par[4]**2*( np.exp(-par[4]*x) - 2.0*np.exp(-2*par[4]*x) ) - 4*(par[5]*par[4])**2*x**2*(y**2 - 1)*np.exp(-par[5]*par[4]*x) )
        
    d2Vdy2 = 8*(6*y**2 - 1)*np.exp( -par[4]*par[5]*x )

    d2Vdydx = -8*y*par[4]*par[5]*np.exp( -par[4]*par[5]*x )*(2*y**2 - 1)

    d2Vdxdy = d2Vdydx    

    Df = np.array([[  0,     0,    par[0],    0],
                   [0,     0,    0,    par[1]],
                   [-d2Vdx2,  -d2Vdydx,   0,    0],
                   [-d2Vdxdy, -d2Vdy2,    0,    0]])
    
    return Df


def variational_eqns_deleonberne(t,PHI,par):
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


def diffcorr_setup_deleonberne():
    """ 
    Returns settings for differential correction method 
        
    Settings include choosing coordinates for event criteria, convergence criteria, and 
    correction (see references for details on how to choose these coordinates).
    """
    
    dydot1 = 1
    correcty0= 0
    MAXdydot1 = 1.e-10
    drdot1 = dydot1
    correctr0 = correcty0
    MAXdrdot1 = MAXdydot1
    
    return [drdot1, correctr0, MAXdrdot1]


def conv_coord_deleonberne(x1, y1, dxdot1, dydot1):
    """
    Returns the variable we want to keep fixed during differential correction.
    
    dxdot1----fix x, dydot1----fix y.
    """
    return dydot1


def get_coord_deleonberne(x,y, E, par):
    """ 
    Returns the initial position of x/y-coordinate on the potential energy 
    surface(PES) for a specific energy E.
    """

    return par[3]*( 1 - math.e**(-par[4]*x) )**2 + \
                4*y**2*(y**2 - 1)*math.e**(-par[5]*par[4]*x) + par[2] - E

def diffcorr_acc_corr_deleonberne(coords, phi_t1, x0, par):
    """ 
    Returns the new guess initial condition of the unstable periodic orbit after applying 
    small correction to the guess. 
        
    Correcting x or y coordinate depends on the problem and needs to chosen by inspecting the 
    geometry of the bottleneck in the potential energy surface.
    """
    
    x1, y1, dxdot1, dydot1 = coords
    
    dVdx = -2*par[3]*par[4]*np.exp(-par[4]*x1)*(np.exp(-par[4]*x1) - 1) - 4*par[5]*par[4]*y1**2*(y1**2 - 1)*np.exp(-par[5]*par[4]*x1)
    dVdy = 8*y1*(2*y1**2 - 1)*np.exp(-par[5]*par[4]*x1)
    vxdot1 = -dVdx
    vydot1 = -dVdy
    #correction to the initial y0
    correcty0 = 1/(phi_t1[3,1] - phi_t1[2,1]*vydot1*(1/vxdot1))*dydot1
    x0[1] = x0[1] - correcty0

    return x0


def configdiff_deleonberne(guess1, guess2, ham2dof_model, half_period_model, n_turn, par):
    """
    Returns the difference of x(or y) coordinates between the guess initial conditions 
    and the ith turning points

    either difference in x coordintes(x_diff1, x_diff2) or difference in 
    y coordinates(y_diff1, y_diff2) is returned as the result.
    """
    
    TSPAN = [0,40]
    RelTol = 3.e-10
    AbsTol = 1.e-10 
    
    f1 = lambda t,x: ham2dof_model(t,x,par) 
    soln1 = solve_ivp(f1, TSPAN, guess1, method='RK45', dense_output=True, \
                      events = lambda t,x: half_period_model(t, x, par), rtol=RelTol, atol=AbsTol)
    te1 = soln1.t_events[0]
    t1 = [0,te1[n_turn]]#[0,te1[1]]
    turn1 = soln1.sol(t1)
    x_turn1 = turn1[0,-1] 
    y_turn1 = turn1[1,-1]
    x_diff1 = guess1[0] - x_turn1
    y_diff1 = guess1[1] - y_turn1
    
    f2 = lambda t,x: ham2dof_model(t,x,par) 
    soln2 = solve_ivp(f2, TSPAN, guess2,method='RK45', dense_output=True, \
                      events = lambda t,x: half_period_model(t, x, par), rtol=RelTol, atol=AbsTol)
    te2 = soln2.t_events[0]
    t2 = [0,te2[n_turn]]#[0,te2[1]]
    turn2 = soln2.sol(t2)
    x_turn2 = turn2[0,-1] 
    y_turn2 = turn2[1,-1] 
    x_diff2 = guess2[0] - x_turn2
    y_diff2 = guess2[1] - y_turn2
    

    print("Initial guess1%s, initial guess2%s, y_diff1 is %s, y_diff2 is%s " %(guess1, \
                                                                               guess2, y_diff1, \
                                                                               y_diff2))
        
    return y_diff1, y_diff2


def guess_coords_deleonberne(guess1, guess2, i, n, e, get_coord_model, par):
    """
    Returns x and y (configuration space) coordinates as guess for the next iteration of the 
    turning point based on confifuration difference method
    """
    
    h = (guess2[1] - guess1[1])*i/n # h is defined for dividing the interval
    print("h is ",h)
    yguess = guess1[1]+h
    f = lambda x: get_coord_model(x,yguess,e,par)
    xguess = optimize.newton(f,-0.2)   # to find the x coordinate for a given y
    
    return xguess, yguess


def plot_iter_orbit_deleonberne(x, ax, e, par):
    """ 
    Plots the orbit in the 3D space of (x,y,p_y) coordinates with the initial and 
    final points marked 
    """

    label_fs = 10
    axis_fs = 15 # fontsize for publications 
    
    ax.plot(x[:,0],x[:,1],x[:,2],'-')
    ax.plot(x[:,0],x[:,1],-x[:,2],'--')
    ax.scatter(x[0,0],x[0,1],x[0,2],s=20,marker='*')
    ax.scatter(x[-1,0],x[-1,1],x[-1,2],s=20,marker='o')
    ax.set_xlabel(r'$x$', fontsize=axis_fs)
    ax.set_ylabel(r'$y$', fontsize=axis_fs)
    ax.set_zlabel(r'$p_x$', fontsize=axis_fs)

    return



def ham2dof_deleonberne(t, x, par):
    """ Returns the Hamiltonian vector field (Hamilton's equations of motion) """
    
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
    Returns the coordinate for the half-period event for the unstable periodic orbit                          
    
    xDot = x[0]
    yDot = x[1]
    pxDot = x[2]
    pyDot = x[3]
    """
    
    terminal = True
    # The zero can be approached from either direction
    direction = 0 #0: all directions of crossing
    
    return x[2]

