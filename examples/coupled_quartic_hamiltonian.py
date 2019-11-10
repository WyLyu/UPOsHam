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
 

# import matplotlib as mpl
# from matplotlib import cm
# mpl.rcParams['mathtext.fontset'] = 'cm'
# mpl.rcParams['mathtext.rm'] = 'serif'


#% Begin problem specific functions
def init_guess_eqpt_coupled(eqNum, par):
    """
    Returns configuration space coordinates of the equilibrium points according to the index:
    Saddle (EQNUM=1)
    Centre (EQNUM=2,3)
    """
    
    if eqNum == 1:
        x0 = [0, 0]
    elif eqNum == 2:
        x0 = [np.sqrt(par[3]-par[6]/par[4]),0] 
    elif eqNum == 3:
        x0 = [-np.sqrt(par[3]-par[6]/par[4]),0] 
    
    return x0

def grad_pot_coupled(x, par):
    """ Returns the gradient of the potential energy function V(x,y) """
     
    dVdx = (-par[3]+par[6])*x[0]+par[4]*(x[0])**3-par[6]*x[1]
    dVdy = (par[5]+par[6])*x[1]-par[6]*x[0]
    
    F = [-dVdx, -dVdy]
    
    return F

def pot_energy_coupled(x, y, par):
    """ Returns the potential energy function V(x,y) """
    
    return -0.5*par[3]*x**2+0.25*par[4]*x**4 +0.5*par[5]*y**2+0.5*par[6]*(x-y)**2


#%
def eigvector_coupled(par):
    """ Returns the correction factor to the eigenvectors for the linear guess """
    
    evaluelamb = np.sqrt(-0.5*(par[3]-par[6]-par[1]*(par[1]+par[6]) - np.sqrt(par[1]**4 + \
                               2*par[1]**3*par[6] + par[1]**2*(par[6]**2+2*par[3]-2*par[6]) + \
                               par[1]*( 2*par[6]**2 + 2*par[3]*par[6]) +(par[3]- par[6])**2)))
#    correcx = par[6]/(-evaluelamb**2 -par[3]+par[6])
#    correcy = 1
    #
    #
    #eqPt = 1
    #eqPt = get_eq_pts_coupled(eqNum, par)
    #evalue, evector = np.linalg.eig(jacobian_coupled([eqPt[0],eqPt[1],0,0],par))
    #evector = RemoveInfinitesimals(evector[:,2])
    #correcx = (evector[0]*1j).real
    #correcy = (evector[1]*1j).real
    correcx = (par[1]*par[6])/(-evaluelamb**2 - par[3] + par[6])
    correcy = par[1]
    
    return correcx, correcy


def guess_lin_coupled(eqPt, Ax, par):
    """ Returns an initial guess for the unstable periodic orbit """ 
    
    correcx, correcy = eigvector_coupled(par)
    

    return [eqPt[0]-Ax*correcx,eqPt[1]-Ax*correcy,0,0]


def jacobian_coupled(eqPt, par):
    """ Returns Jacobian of the Hamiltonian vector field """
    
    x,y,px,py = eqPt[0:4]
    
    # The first order derivative of the Hamiltonian.
    dVdx = (-par[3]+par[6])*x+par[4]*x**3-par[6]*y
    dVdy = (par[5]+par[6])*y-par[6]*x

    # The following is the Jacobian matrix 
    d2Vdx2 = -par[3]+par[6]+par[4]*3*x**2
        
    d2Vdy2 = par[5]+par[6]

    d2Vdydx = -par[6]
        
    d2Vdxdy = d2Vdydx    

    Df = np.array([[  0,     0,    par[0],    0],
                   [0,     0,    0,    par[1]],
                   [-d2Vdx2,  -d2Vdydx,   0,    0],
                   [-d2Vdxdy, -d2Vdy2,    0,    0]])
    
    return Df


def varEqns_coupled(t,PHI,par):
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
    dVdx = (-par[3]+par[6])*x+par[4]*x**3-par[6]*y
    dVdy = (par[5]+par[6])*y-par[6]*x

    # The second order derivative of the potential energy. 
    d2Vdx2 = -par[3]+par[6]+par[4]*3*x**2
        
    d2Vdy2 = par[5]+par[6]

    d2Vdydx = -par[6]

    
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


def diffcorr_setup_coupled():
    """ 
    Returns settings for differential correction method 
        
    Settings include choosing coordinates for event criteria, convergence criteria, and 
    correction (see references for details on how to choose these coordinates).
    """
    
    dxdot1 = 1
    correctx0 = 0
    MAXdxdot1 = 1.e-10
    drdot1 = dxdot1
    correctr0 = correctx0
    MAXdrdot1 = MAXdxdot1
    
    return [drdot1, correctr0, MAXdrdot1]


def conv_coord_coupled(x1, y1, dxdot1, dydot1):
    return dxdot1


def get_coord_coupled(x, y, E, par):
    """ 
    Returns the initial position of x/y-coordinate on the potential energy 
    surface(PES) for a specific energy E.
    """
#    if model == 'uncoupled':
#        return -0.5*par[3]*x**2+0.25*par[4]*x**4 +0.5*par[5]*y**2-V
#    elif model =='coupled':
    return -0.5*par[3]*x**2+0.25*par[4]*x**4 +0.5*par[5]*y**2+0.5*par[6]*(x-y)**2 - E
#    elif model== 'deleonberne':
#        return par[3]*( 1 - math.e**(-par[4]*x) )**2 + 4*y**2*(y**2 - 1)*math.e**(-par[5]*par[4]*x) + par[2]-V
#    else:
#        print("The model you are chosen does not exist, enter the function for finding coordinates on the PES for given x or y and V")


def diffcorr_acc_corr_coupled(coords, phi_t1, x0, par):
    """ 
    Returns the new guess initial condition of the unstable periodic orbit after applying 
    small correction to the guess. 
        
    Correcting x or y coordinate depends on the problem and needs to chosen by inspecting the 
    geometry of the bottleneck in the potential energy surface.
    """
    
    x1, y1, dxdot1, dydot1 = coords
    
    dVdx = (-par[3]+par[6])*x1+par[4]*(x1)**3-par[6]*y1
    dVdy = (par[5]+par[6])*y1-par[6]*x1
    vxdot1 = -dVdx
    vydot1 = -dVdy
    
    #correction to the initial x0
    correctx0 = dxdot1/(phi_t1[2,0] - phi_t1[3,0]*(vxdot1/vydot1))	
    x0[0] = x0[0] - correctx0
    
    return x0


def configdiff_coupled(guess1, guess2, ham2dof_model, half_period_model, n_turn, par):
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
    

    print("Initial guess1%s, initial guess2 %s, x_diff1 is %s, x_diff2 is %s " %(guess1, \
                                                                                 guess2, x_diff1, \
                                                                                 x_diff2))
    
    return x_diff1, x_diff2 #,y_diff1, y_diff2


def guess_coords_coupled(guess1, guess2, i, n, e, get_coord_model, par):
    """
    Returns x and y (configuration space) coordinates as guess for the next iteration of the 
    turning point based on confifuration difference method
    """
    
    
    h = (guess2[0] - guess1[0])*i/n
    print("h is ",h)
    xguess = guess1[0]+h
    f = lambda y: get_coord_model(xguess,y,e,par)
    yanalytic = math.sqrt(2/(par[1]+par[6]))*(-math.sqrt( e +0.5*par[3]* xguess**2 - \
                         0.25*par[4]*xguess**4 -0.5*par[6]* xguess**2 + \
                         (par[6]*xguess)**2/(2*(par[1] +par[6]) )) + 
                        par[6]/(math.sqrt(2*(par[1]+par[6])) )*xguess ) #coupled
    yguess = optimize.newton(f,yanalytic) 
    
    return xguess, yguess


def plot_iter_orbit_coupled(x, ax, e, par):
    """ 
    Plots the orbit in the 3D space of (x,y,p_y) coordinates with the initial and 
    final points marked 
    """
    
#    label_fs = 10
    axis_fs = 15 # fontsize for publications 
    
    ax.plot(x[:,0],x[:,1],x[:,3],'-')
    ax.plot(x[:,0],x[:,1],-x[:,3],'--')
    ax.scatter(x[0,0],x[0,1],x[0,3],s=20,marker='*')
    ax.scatter(x[-1,0],x[-1,1],x[-1,3],s=20,marker='o')
    ax.set_xlabel(r'$x$', fontsize=axis_fs)
    ax.set_ylabel(r'$y$', fontsize=axis_fs)
    ax.set_zlabel(r'$p_y$', fontsize=axis_fs)
    ax.set_title(r'$\Delta E$ = %e' %(np.mean(e) - par[2]) ,fontsize=axis_fs)
    #par(3) is the energy of the saddle
    ax.set_xlim(-0.1, 0.1)
    
    return 


def ham2dof_coupled(t, x, par):
    """ Returns the Hamiltonian vector field (Hamilton's equations of motion) """
    
    xDot = np.zeros(4)
    
    dVdx = (-par[3]+par[6])*x[0]+par[4]*(x[0])**3-par[6]*x[1]
    dVdy = (par[5]+par[6])*x[1]-par[6]*x[0]
        
    xDot[0] = x[2]/par[0]
    xDot[1] = x[3]/par[1]
    xDot[2] = -dVdx 
    xDot[3] = -dVdy
    
    return list(xDot)    

def half_period_coupled(t, x, par):
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
    
    return x[3]


#% End problem specific functions

#% Begin problem specific functions
# def init_guess_eqpt_coupled(eqNum, par):
#     """
#     Returns configuration space coordinates of the equilibrium points according to the index:
#     Saddle (EQNUM=1)
#     Centre (EQNUM=2,3)
#     """ 
    
#     if eqNum == 1:
#         x0 = [0, 0]
#     elif eqNum == 2:
#         x0 = [np.sqrt(par[3]-par[6]/par[4]),0] 
#     elif eqNum == 3:
#         x0 = [-np.sqrt(par[3]-par[6]/par[4]),0] 
    
#     return x0

# def grad_pot_coupled(x, par):
#     """
#     Returns the gradient of the potential energy function V(x,y)
#     """ 
    
#     dVdx = (-par[3]+par[6])*x[0]+par[4]*(x[0])**3-par[6]*x[1]
#     dVdy = (par[5]+par[6])*x[1]-par[6]*x[0]
            
#     F = [-dVdx, -dVdy]
    
#     return F

# def pot_energy_coupled(x, y, par):
#     """Returns the potential energy function V(x,y)
#     """
    
#     return -0.5*par[3]*x**2+0.25*par[4]*x**4 +0.5*par[5]*y**2+0.5*par[6]*(x-y)**2


# def get_coord_coupled(x, y, E, par):
#     """ 
#     Returns the initial position of x/y-coordinate on the potential energy 
#     surface(PES) for a specific energy E.
#     """
    
# #    if model == 'uncoupled':
# #        return -0.5*par[3]*x**2+0.25*par[4]*x**4 +0.5*par[5]*y**2-V
# #    elif model =='coupled':
#     return -0.5*par[3]*x**2+0.25*par[4]*x**4 +0.5*par[5]*y**2+0.5*par[6]*(x-y)**2 - E
# #    elif model== 'deleonberne':
# #        return par[3]*( 1 - math.e**(-par[4]*x) )**2 + 4*y**2*(y**2 - 1)*math.e**(-par[5]*par[4]*x) + par[2]-V
# #    else:
# #        print("The model you are chosen does not exist, enter the function for finding coordinates on the PES for given x or y and V")


# def varEqns_coupled(t,PHI,par):
#     """    
#     Returns the state transition matrix , PHI(t,t0), where Df(t) is the Jacobian of the 
#     Hamiltonian vector field
    
#     d PHI(t, t0)
#     ------------ =  Df(t) * PHI(t, t0)
#         dt
    
#     """
    
#     phi = PHI[0:16]
#     phimatrix  = np.reshape(PHI[0:16],(4,4))
#     x,y,px,py = PHI[16:20]
    
    
#     # The first order derivative of the potential energy.
#     dVdx = (-par[3]+par[6])*x+par[4]*x**3-par[6]*y
#     dVdy = (par[5]+par[6])*y-par[6]*x

#     # The second order derivative of the potential energy. 
#     d2Vdx2 = -par[3]+par[6]+par[4]*3*x**2
        
#     d2Vdy2 = par[5]+par[6]

#     d2Vdydx = -par[6]

    
#     d2Vdxdy = d2Vdydx    

#     Df    = np.array([[  0,     0,    par[0],    0],
#               [0,     0,    0,    par[1]],
#               [-d2Vdx2,  -d2Vdydx,   0,    0],
#               [-d2Vdxdy, -d2Vdy2,    0,    0]])

    
#     phidot = np.matmul(Df, phimatrix) # variational equation

#     PHIdot        = np.zeros(20)
#     PHIdot[0:16]  = np.reshape(phidot,(1,16)) 
#     PHIdot[16]    = px/par[0]
#     PHIdot[17]    = py/par[1]
#     PHIdot[18]    = -dVdx 
#     PHIdot[19]    = -dVdy
    
#     return list(PHIdot)


# def ham2dof_coupled(t, x, par):
#     """ Returns the Hamiltonian vector field (Hamilton's equations of motion)
#     """
    
#     xDot = np.zeros(4)
    
#     dVdx = (-par[3]+par[6])*x[0]+par[4]*(x[0])**3-par[6]*x[1]
#     dVdy = (par[5]+par[6])*x[1]-par[6]*x[0]
        
#     xDot[0] = x[2]/par[0]
#     xDot[1] = x[3]/par[1]
#     xDot[2] = -dVdx 
#     xDot[3] = -dVdy
    
#     return list(xDot)    


# def half_period_coupled(t, x, par):
#     """
#     Return the turning point where we want to stop the integration                           
    
#     pxDot = x[0]
#     pyDot = x[1]
#     xDot = x[2]
#     yDot = x[3]
#     """
    
#     terminal = True
#     # The zero can be approached from either direction
#     direction = 0 #0: all directions of crossing
    
#     return x[3]


# def guess_coords_coupled(guess1, guess2, i, n, e, get_coord_model, par):
#     """
#     Returns x and y (configuration space) coordinates as guess for the next iteration of the 
#     turning point based on confifuration difference method
#     """
    
    
#     h = (guess2[0] - guess1[0])*i/n
#     print("h is ",h)
#     xguess = guess1[0]+h
#     f = lambda y: get_coord_model(xguess,y,e,par)
#     yanalytic = math.sqrt(2/(par[1]+par[6]))*(-math.sqrt( e +0.5*par[3]* xguess**2 - \
#                          0.25*par[4]*xguess**4 -0.5*par[6]* xguess**2 + \
#                          (par[6]*xguess)**2/(2*(par[1] +par[6]) )) + 
#                         par[6]/(math.sqrt(2*(par[1]+par[6])) )*xguess ) #coupled
#     yguess = optimize.newton(f,yanalytic) 
    
#     return xguess, yguess
    
# def plot_iter_orbit_coupled(x, ax, e, par):
#     """ 
#     Plots the orbit in the 3D space of (x,y,p_y) coordinates with the initial and 
#     final points marked 
#     """
    
#     label_fs = 10
#     axis_fs = 15 # fontsize for publications 
    
#     ax.plot(x[:,0],x[:,1],x[:,3],'-')
#     ax.plot(x[:,0],x[:,1],-x[:,3],'--')
#     ax.scatter(x[0,0],x[0,1],x[0,3],s=20,marker='*')
#     ax.scatter(x[-1,0],x[-1,1],x[-1,3],s=20,marker='o')
#     ax.set_xlabel(r'$x$', fontsize=axis_fs)
#     ax.set_ylabel(r'$y$', fontsize=axis_fs)
#     ax.set_zlabel(r'$p_y$', fontsize=axis_fs)
#     ax.set_title(r'$\Delta E$ = %e' %(np.mean(e) - par[2]) ,fontsize=axis_fs)
#     #par(3) is the energy of the saddle
#     ax.set_xlim(-0.1, 0.1)
    
#     return 
#% End problem specific functions


#% Begin problem specific functions
# def init_guess_eqpt_coupled(eqNum, par):
#     """
#     Returns configuration space coordinates of the equilibrium points according to the index:
#     Saddle (EQNUM=1)
#     Centre (EQNUM=2,3)
#     """    
#     if eqNum == 1:
#         x0 = [0, 0]
#     elif eqNum == 2:
#         x0 = [np.sqrt(par[3]-par[6]/par[4]),0] 
#     elif eqNum == 3:
#         x0 = [-np.sqrt(par[3]-par[6]/par[4]),0] 
    
#     return x0

# def grad_pot_coupled(x, par):
#     """
#     Returns the gradient of the potential energy function V(x,y)
#     """
    
#     dVdx = (-par[3]+par[6])*x[0]+par[4]*(x[0])**3-par[6]*x[1]
#     dVdy = (par[5]+par[6])*x[1]-par[6]*x[0]
            
#     F = [-dVdx, -dVdy]
    
#     return F

# def pot_energy_coupled(x, y, par):
#     """Returns the potential energy function V(x,y)
#     """
#     return -0.5*par[3]*x**2+0.25*par[4]*x**4 +0.5*par[5]*y**2+0.5*par[6]*(x-y)**2


# def varEqns_coupled(t,PHI,par):
#     """    
#     Returns the state transition matrix , PHI(t,t0), where Df(t) is the Jacobian of the 
#     Hamiltonian vector field
    
#     d PHI(t, t0)
#     ------------ =  Df(t) * PHI(t, t0)
#         dt
    
#     """
    
#     phi = PHI[0:16]
#     phimatrix  = np.reshape(PHI[0:16],(4,4))
#     x,y,px,py = PHI[16:20]
    
    
#     # The first order derivative of the potential energy.
#     dVdx = (-par[3]+par[6])*x+par[4]*x**3-par[6]*y
#     dVdy = (par[5]+par[6])*y-par[6]*x

#     # The second order derivative of the potential energy. 
#     d2Vdx2 = -par[3]+par[6]+par[4]*3*x**2
        
#     d2Vdy2 = par[5]+par[6]

#     d2Vdydx = -par[6]

    
#     d2Vdxdy = d2Vdydx    

#     Df    = np.array([[  0,     0,    par[0],    0],
#               [0,     0,    0,    par[1]],
#               [-d2Vdx2,  -d2Vdydx,   0,    0],
#               [-d2Vdxdy, -d2Vdy2,    0,    0]])

    
#     phidot = np.matmul(Df, phimatrix) # variational equation

#     PHIdot        = np.zeros(20)
#     PHIdot[0:16]  = np.reshape(phidot,(1,16)) 
#     PHIdot[16]    = px/par[0]
#     PHIdot[17]    = py/par[1]
#     PHIdot[18]    = -dVdx 
#     PHIdot[19]    = -dVdy
    
#     return list(PHIdot)


# def ham2dof_coupled(t, x, par):
#     """ Returns the Hamiltonian vector field (Hamilton's equations of motion)
#     """
    
#     xDot = np.zeros(4)
    
#     dVdx = (-par[3]+par[6])*x[0]+par[4]*(x[0])**3-par[6]*x[1]
#     dVdy = (par[5]+par[6])*x[1]-par[6]*x[0]
        
#     xDot[0] = x[2]/par[0]
#     xDot[1] = x[3]/par[1]
#     xDot[2] = -dVdx 
#     xDot[3] = -dVdy
    
#     return list(xDot)    

# def half_period_coupled(t, x, par):
#     """
#     Return the turning point where we want to stop the integration                           
    
#     pxDot = x[0]
#     pyDot = x[1]
#     xDot = x[2]
#     yDot = x[3]
#     """
    
#     terminal = True
#     # The zero can be approached from either direction
#     direction = 0 #0: all directions of crossing
    
#     return x[3]


    
# def plot_iter_orbit_coupled(x, ax, e, par):
#     """ 
#     Plots the orbit in the 3D space of (x,y,p_y) coordinates with the initial and 
#     final points marked 
#     """
    
#     label_fs = 10
#     axis_fs = 15 # fontsize for publications 
    
#     ax.plot(x[:,0],x[:,1],x[:,3],'-')
#     ax.plot(x[:,0],x[:,1],-x[:,3],'--')
#     ax.scatter(x[0,0],x[0,1],x[0,3],s=20,marker='*')
#     ax.scatter(x[-1,0],x[-1,1],x[-1,3],s=20,marker='o')
#     ax.set_xlabel(r'$x$', fontsize=axis_fs)
#     ax.set_ylabel(r'$y$', fontsize=axis_fs)
#     ax.set_zlabel(r'$p_y$', fontsize=axis_fs)
#     ax.set_title(r'$\Delta E$ = %e' %(np.mean(e) - par[2]) ,fontsize=axis_fs)
#     #par(3) is the energy of the saddle
#     ax.set_xlim(-0.1, 0.1)
    
#     return 

#% End problem specific functions