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













