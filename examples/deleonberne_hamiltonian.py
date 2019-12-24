# -*- coding: utf-8 -*-
# """
# Created on Tue Jul 30 10:02:48 2019

# @author: Wenyang Lyu and Shibabrat Naik

# Script to define expressions for the De Leon-Berne Hamiltonian
# """

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time
import math
from scipy import optimize



#% Begin problem specific functions
def init_guess_eqpt_deleonberne(eqNum, parameters):
    """
    Returns guess for solving configuration space coordinates of the equilibrium points.  

    For numerical solution of the equilibrium points, this function returns the guess that be inferred from the potential energy surface. 

    Parameters
    ----------
    eqNum : int
        = 1 for saddle and = 2,3 for centre equilibrium points

    parameters : float (list)
        model parameters

    Returns
    -------
    x0 : float (list of size 2)
        configuration space coordinates of the guess: [x, y] 

    """
    
    if eqNum == 1:
        x0 = [0, 0]
    elif eqNum == 2:
        x0 = [0, 1/np.sqrt(2)]  # EQNUM = 2, center-center
    elif eqNum == 3:
        x0 = [0, -1/np.sqrt(2)] # EQNUM = 3, center-center
    
    return x0


def grad_pot_deleonberne(x, parameters):
    """ Returns the negative of the gradient of the potential energy function 
    
    Parameters
    ----------
    x : float (list of size 2) 
        configuration space coordinates: [x, y]

    parameters : float (list)
        model parameters

    Returns
    -------
    F : float (list of size 2)
        configuration space coordinates of the guess: [x, y]

    """
     
    dVdx = -2*parameters[3]*parameters[4]*np.exp(-parameters[4]*x[0])*(np.exp(-parameters[4]*x[0]) - 1) - \
        4*parameters[5]*parameters[4]*x[1]**2*(x[1]**2 - 1)*np.exp(-parameters[5]*parameters[4]*x[0])
    dVdy = 8*x[1]*(2*x[1]**2 - 1)*np.exp(-parameters[5]*parameters[4]*x[0])
    
    F = [-dVdx, -dVdy]
    
    return F

def pot_energy_deleonberne(x, y, parameters):
    """ Returns the potential energy at the configuration space coordinates 
    

    Parameters
    ----------
    x : float
        configuration space coordinate

    y : float
        configuration space coordinate

    parameters : float (list)
        model parameters

    Returns
    -------
    float 
        potential energy of the configuration
    
    """
    
    return parameters[3]*( 1 - np.exp(-parameters[4]*x) )**2 + 4*y**2*(y**2 - 1)*np.exp(-parameters[5]*parameters[4]*x) + parameters[2]


def eigvector_deleonberne(parameters):
    """ Returns the flag for the correction factor to the eigenvectors for the linear guess of the unstable periodic orbit.
    
    Parameters
    ----------
    parameters : float (list)
        model parameters

    Returns
    -------
    correcx : 1 or 0 
        flag to set the x-component of the eigenvector

    correcy : 1 or 0
        flag to use the y-component of the eigenvector 
    
    """
    correcx = 1
    correcy = 0
    
    return correcx, correcy


def guess_lin_deleonberne(eqPt, Ax, parameters):
    """ Returns an initial guess as list of coordinates in the phase space 
    
    This guess is based on the linearization at the saddle equilibrium point and is used for starting the differential correction iteration 
    
    Parameters
    ----------
    eqPt : float (list of size 2)
        configuration space coordinates of the equilibrium point

    Ax : float
        small amplitude displacement from the equilibrium point to initialize guess for the differential correction

    parameters : float (list)
        model parameters

    Returns
    -------
    float (list of size 4)
        phase space coordinates of the initial guess in the phase space
         
    """    
    correcx, correcy = eigvector_deleonberne(parameters)
    
    return [eqPt[0] - Ax*correcx,eqPt[1] + Ax*correcy,0,0]


def jacobian_deleonberne(eqPt, parameters):
    """ Returns Jacobian of the Hamiltonian vector field 
    
    Parameters
    ----------
    eqPt : float (list of size 4)
        phase space coordinates of the equilibrium point

    parameters : float (list)
        model parameters


    Returns
    -------
    Df : 2d numpy array
        Jacobian matrix 

    
    """
    
    x,y,px,py = eqPt[0:4]
    # The first order derivative of the Hamiltonian.
    dVdx = - 2*parameters[3]*parameters[4]*np.exp(-parameters[4]*x)*(np.exp(-parameters[4]*x) - 1) - 4*parameters[5]*parameters[4]*y**2*(y**2 - 1)*np.exp(-parameters[5]*parameters[4]*x) 
    dVdy = 8*y*(2*y**2 - 1)*np.exp(-parameters[5]*parameters[4]*x)

    # The following is the Jacobian matrix 
    d2Vdx2 = - ( 2*parameters[3]*parameters[4]**2*( np.exp(-parameters[4]*x) - 2.0*np.exp(-2*parameters[4]*x) ) - 4*(parameters[5]*parameters[4])**2*x**2*(y**2 - 1)*np.exp(-parameters[5]*parameters[4]*x) )
        
    d2Vdy2 = 8*(6*y**2 - 1)*np.exp( -parameters[4]*parameters[5]*x )

    d2Vdydx = -8*y*parameters[4]*parameters[5]*np.exp( -parameters[4]*parameters[5]*x )*(2*y**2 - 1)

    d2Vdxdy = d2Vdydx    

    Df = np.array([[  0,     0,    parameters[0],    0],
                   [0,     0,    0,    parameters[1]],
                   [-d2Vdx2,  -d2Vdydx,   0,    0],
                   [-d2Vdxdy, -d2Vdy2,    0,    0]])
    
    return Df


def variational_eqns_deleonberne(t,PHI,parameters):
    """    
    Returns the state transition matrix, PHI(t,t0), where Df(t) is the Jacobian of the Hamiltonian vector field
    
    d PHI(t, t0)/dt =  Df(t) * PHI(t, t0)

    Parameters
    ----------
    t : float 
        solution time

    PHI : 1d numpy array
        state transition matrix and the phase space coordinates at initial time in the form of a vector

    parameters : float (list)
        model parameters 

    Returns
    -------
    PHIdot : float (list of size 20)
        right hand side for solving the state transition matrix 

    """
    
    phi = PHI[0:16]
    phimatrix  = np.reshape(PHI[0:16],(4,4))
    x,y,px,py = PHI[16:20]
    
    # The first order derivative of the potential energy.
    dVdx = - 2*parameters[3]*parameters[4]*np.exp(-parameters[4]*x)*(np.exp(-parameters[4]*x) - 1) \
        - 4*parameters[5]*parameters[4]*y**2*(y**2 - 1)*np.exp(-parameters[5]*parameters[4]*x) 
    dVdy = 8*y*(2*y**2 - 1)*np.exp(-parameters[5]*parameters[4]*x)

    # The second order derivative of the potential energy. 
    d2Vdx2 = - ( 2*parameters[3]*parameters[4]**2*( np.exp(-parameters[4]*x) \
        - 2.0*np.exp(-2*parameters[4]*x) ) \
        - 4*(parameters[5]*parameters[4])**2*x**2*(y**2 - 1)*np.exp(-parameters[5]*parameters[4]*x) )
        
    d2Vdy2 = 8*(6*y**2 - 1)*np.exp( -parameters[4]*parameters[5]*x )

    d2Vdydx = -8*y*parameters[4]*parameters[5]*np.exp( -parameters[4]*parameters[5]*x )*(2*y**2 - 1)
        
    
    d2Vdxdy = d2Vdydx    

    Df    = np.array([[  0,     0,    parameters[0],    0],
              [0,     0,    0,    parameters[1]],
              [-d2Vdx2,  -d2Vdydx,   0,    0],
              [-d2Vdxdy, -d2Vdy2,    0,    0]])

    
    phidot = np.matmul(Df, phimatrix) # variational equation

    PHIdot        = np.zeros(20)
    PHIdot[0:16]  = np.reshape(phidot,(1,16)) 
    PHIdot[16]    = px/parameters[0]
    PHIdot[17]    = py/parameters[1]
    PHIdot[18]    = -dVdx 
    PHIdot[19]    = -dVdy
    
    return list(PHIdot)


def diffcorr_setup_deleonberne():
    """ 
    Returns iteration conditions for differential correction.

    See references for details on how to set up the conditions and how to choose the coordinates in the iteration procedure.

    Parameters
    ----------
    Empty - None

    Returns
    -------
    drdot1 : 1 or 0
        flag to select the phase space coordinate for event criteria in stopping integration of the periodic orbit

    correctr0 : 1 or 0
        flag to select which configuration coordinate to apply correction

    MAXdrdot1 : float
        tolerance to satisfy for convergence of the method
        
    """
    
    dydot1 = 1
    correcty0 = 0
    MAXdydot1 = 1.e-10
    drdot1 = dydot1
    correctr0 = correcty0
    MAXdrdot1 = MAXdydot1
    
    return [drdot1, correctr0, MAXdrdot1]


def conv_coord_deleonberne(x1, y1, dxdot1, dydot1):
    """
    Returns the variable we want to keep fixed during differential correction.
    
    dxdot1 -> fix x, dydot1 -> fix y.

    Parameters
    ----------
    x1 : float
        value of phase space coordinate, x

    y1 : float
        value of phase space coordinate, y

    dxdot1 : float
        value of phase space coordinate, dxdot

    dydot1 : float 
        value of phase space coordinate, dydot

    Returns
    -------
        value of one of the phase space coordinates

    """
    return dydot1


def get_coord_deleonberne(x,y, E, parameters):
    """ 
    Function that returns the potential energy for a given total energy with one of the configuration space coordinate being fixed

    Used to solve for coordinates on the isopotential contours using numerical solvers

    Parameters
    ----------
    x : float
        configuration space coordinate

    y : float
        configuration space coordinate

    E : float
        total energy

    parameters :float (list)
        model parameters

    Returns
    -------
        float
        Potential energy
    """

    return parameters[3]*( 1 - math.e**(-parameters[4]*x) )**2 + \
                4*y**2*(y**2 - 1)*math.e**(-parameters[5]*parameters[4]*x) + parameters[2] - E

def diffcorr_acc_corr_deleonberne(coords, phi_t1, x0, parameters):
    """ 
    Returns the updated guess for the initial condition after applying 
    small correction based on the leading order terms. 
        
    Correcting x or y coordinate of the guess depends on the system and needs to be chosen by inspecting the geometry of the bottleneck in the potential energy surface.

    Parameters
    ----------
    coords : float (list of size 4)
        phase space coordinates in the order of position and momentum

    phi_t1 : 2d numpy array
        state transition matrix evaluated at the time t1 which is used to derive the correction terms

    x0 : float 
        coordinate of the initial condition before the correction

    parameters : float (list)
        model parameters

    Returns
    -------
    x0 : float 
        coordinate of the initial condition after the correction
    """
    
    x1, y1, dxdot1, dydot1 = coords
    
    dVdx = -2*parameters[3]*parameters[4]*np.exp(-parameters[4]*x1)*(np.exp(-parameters[4]*x1) - 1) \
        - 4*parameters[5]*parameters[4]*y1**2*(y1**2 - 1)*np.exp(-parameters[5]*parameters[4]*x1)
    dVdy = 8*y1*(2*y1**2 - 1)*np.exp(-parameters[5]*parameters[4]*x1)
    vxdot1 = -dVdx
    vydot1 = -dVdy

    #correction to the initial y0
    correcty0 = 1/(phi_t1[3,1] - phi_t1[2,1]*vydot1*(1/vxdot1))*dydot1
    x0[1] = x0[1] - correcty0

    return x0


def configdiff_deleonberne(guess1, guess2, ham2dof_model,\
                            half_period_model, n_turn, parameters):
    """
    Returns the difference of x(or y) coordinates of the guess initial condition and the ith turning point

    Used by turning point based on configuration difference method and passed as an argument by the user. Depending on the orientation of a system's bottleneck in the potential energy surface, this function should return either difference in x coordinates or difference in y coordinates is returned as the result.

    Parameters
    ----------
    guess1 : float (list of size 4)
        initial condition # 1

    guess2 : float (list of size 4)
        initial condition # 2 

    ham2dof_model : function name
        function that returns the Hamiltonian vector field  
        
    half_period_model : function name
        function to catch the half period event during integration
    
    n_turn : int 
        index of the number of turn as a trajectory comes close to an equipotential contour
    
    parameters : float (list)
        model parameters 

    Returns
    -------
    (x_diff1, x_diff2) or (y_diff1, y_diff2) : float (list of size 2)
        difference in the configuration space coordinates, either x or y depending on the orientation of the bottleneck.

    """
    
    TSPAN = [0,40]
    RelTol = 3.e-10
    AbsTol = 1.e-10 
    
    f1 = lambda t,x: ham2dof_model(t,x,parameters) 
    soln1 = solve_ivp(f1, TSPAN, guess1, method='RK45', dense_output=True, \
                      events = lambda t,x: half_period_model(t, x, parameters), rtol=RelTol, atol=AbsTol)
    te1 = soln1.t_events[0]
    t1 = [0,te1[n_turn]]
    turn1 = soln1.sol(t1)
    x_turn1 = turn1[0,-1] 
    y_turn1 = turn1[1,-1]
    x_diff1 = guess1[0] - x_turn1
    y_diff1 = guess1[1] - y_turn1
    
    f2 = lambda t,x: ham2dof_model(t,x,parameters) 
    soln2 = solve_ivp(f2, TSPAN, guess2,method='RK45', dense_output=True, \
                      events = lambda t,x: half_period_model(t, x, parameters), rtol=RelTol, atol=AbsTol)
    te2 = soln2.t_events[0]
    t2 = [0,te2[n_turn]]#[0,te2[1]]
    turn2 = soln2.sol(t2)
    x_turn2 = turn2[0,-1] 
    y_turn2 = turn2[1,-1] 
    x_diff2 = guess2[0] - x_turn2
    y_diff2 = guess2[1] - y_turn2
    

    print("Initial guess1 %.6f, initial guess2 %.6f, \
            y_diff1 is %.6f, y_diff2 is %.6f" %(guess1, guess2, y_diff1, y_diff2))
        
    return y_diff1, y_diff2


def guess_coords_deleonberne(guess1, guess2, i, n, e, \
                            get_coord_model,parameters):
    """
    Returns x and y (configuration space) coordinates as guess for the next iteration of the turning point based on confifuration difference method

    Function to be used by the turning point based on configuration difference method and passed as an argument.

    Parameters
    ----------
    guess1 : float (list of size 4)
        initial condition # 1

    guess2 : float (list of size 4)
        initial condition # 2 

    i : int
        index of the number of partitions of the interval between the two guess coordinates

    n : int
        total number of partitions of the interval between the two guess coordinates 
    
    e : float
        total energy

    get_coord_model : function name
        function that returns the potential energy for a given total energy with one of the configuration space coordinate being fixed

    parameters : float (list)
        model parameters

    Returns
    -------
    xguess, yguess : float 
        configuration space coordinates of the next guess of the initial condition

    """
    
    h = (guess2[1] - guess1[1])*i/n # h is defined for dividing the interval
    print("h is %.6f",h)
    yguess = guess1[1] + h
    f = lambda x: get_coord_model(x,yguess,e,parameters)
    xguess = optimize.newton(f,-0.2) # to find the x coordinate for a given y
    
    return xguess, yguess


def plot_iter_orbit_deleonberne(x, ax, e, parameters):
    """ 
    Plots the orbit in the 3D space of (x,y,p_x) coordinates with the initial and final points marked with star and circle. 

    Parameters
    ----------
    x : 2d numpy array
        trajectory with time ordering along rows and coordinates along columns

    ax : figure object
        3D matplotlib axes

    e : float
        total energy

    parameters :float (list)
        model parameters 
    
    Returns
    -------
    Empty - None

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


def ham2dof_deleonberne(t, x, parameters):
    """ 
    Returns the Hamiltonian vector field (Hamilton's equations of motion) 
    
    Used for passing to ode solvers for integrating initial conditions over a time interval.

    Parameters
    ----------
    t : float
        time instant

    x : float (list of size 4)
        phase space coordinates at time instant t

    parameters : float (list)
        model parameters

    Returns
    -------
    xDot : float (list of size 4)
        right hand side of the vector field evaluated at the phase space coordinates, x, at time instant, t

    """
    
    xDot = np.zeros(4)
    
    dVdx = -2*parameters[3]*parameters[4]*np.exp(-parameters[4]*x[0])*(np.exp(-parameters[4]*x[0]) - 1) - \
        4*parameters[5]*parameters[4]*x[1]**2*(x[1]**2 - 1)*np.exp(-parameters[5]*parameters[4]*x[0])
    dVdy = 8*x[1]*(2*x[1]**2 - 1)*np.exp(-parameters[5]*parameters[4]*x[0])
        
    xDot[0] = x[2]/parameters[0]
    xDot[1] = x[3]/parameters[1]
    xDot[2] = -dVdx 
    xDot[3] = -dVdy
    
    return list(xDot)  


def half_period_deleonberne(t,x,parameters):
    """
    Returns the event function 
    
    Zero of this function is used as the event criteria to stop integration along a trajectory. For symmetric periodic orbits this acts as the half period event when the momentum coordinate is zero.

    Parameters
    ----------
    t : float
        time instant

    x : float (list of size 4)
        phase space coordinates at time instant t

    parameters : float (list)
        model parameters

    Returns
    -------
    float 
        event function evaluated at the phase space coordinate, x, and time instant, t.

    """
    
    return x[2]

        
half_period_deleonberne.terminal = True # terminate the integration.
half_period_deleonberne.direction = 0 # zero of the event function can be approached from either direction and will trigger the terminate

