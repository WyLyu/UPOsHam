# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 17:49:31 2019

@author: Wenyang Lyu and Shibabrat Naik
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import math
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import fsolve
from scipy import optimize
import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'

    
def get_eq_pts(eqNum, init_guess_eqpt_model, grad_pot_model, par):
    """
    Returns configuration space coordinates of the equilibrium points.

    get_eq_pts(eqNum, init_guess_eqpt_model, grad_pot_model, par) solves the
    coordinates of the equilibrium point for a Hamiltonian of the form kinetic
    energy (KE) + potential energy (PE).

    Parameters
    ----------
    eqNum : int
        1 is the saddle-center equilibrium point
        2 or 3 is the center-center equilibrium point

    init_guess_eqpt_model : function name
        function that returns the initial guess for the equilibrium point

    grad_pot_model : function name
        function that defines the vector of potential gradient

    par : float (list)
        model parameters

    Returns
    -------
    float (list)
        configuration space coordinates

    """
    #fix the equilibrium point numbering convention here and make a
    #starting guess at the solution
    x0 = init_guess_eqpt_model(eqNum, par)
    
    # F(xEq) = 0 at the equilibrium point, solve using in-built function
    F = lambda x: grad_pot_model(x, par)
    
    eqPt = fsolve(F, x0, fprime = None) # Call solver
    
    return eqPt




#%
def get_total_energy(orbit, pot_energy_model, parameters):
    """
    Returns total energy (value of Hamiltonian) of a phase space point on an orbit

    get_total_energy(orbit, pot_energy_model, parameters) returns the total energy for a
    Hamiltonian of the form KE + PE.

    Parameters
    ----------
    orbit : float (list)
        phase space coordinates (x,y,px,py) of a point on an orbit

    pot_energy_model : function name
        function that returns the potential energy of Hamiltonian

    parameters : float (list)
        model parameters

    Returns
    -------
    scalar
        total energy (value of Hamiltonian)

    """
    x  = orbit[0]
    y  = orbit[1]
    px = orbit[2]
    py = orbit[3]
    
    return (1.0/(2*parameters[0]))*(px**2.0) + (1.0/(2*parameters[1]))*(py**2.0) + \
            pot_energy_model(x, y, parameters)   


#%%
def get_pot_surf_proj(xVec, yVec, pot_energy_model, par):            
    """
    Returns projection of the potential energy (PE) surface on the configuration space

    Parameters
    ----------
    xVec, yVec : 1d numpy arrays
        x,y-coordinates that discretizes the x, y domain of the configuration space

    pot_energy_model : function name
        function that returns the potential energy of Hamiltonian

    parameters : float (list)
        model parameters

    Returns
    -------
    2d numpy array
        values of the PE

    """
    
    resX = np.size(xVec)
    resY = np.size(xVec)
    surfProj = np.zeros([resX, resY])
    for i in range(len(xVec)):
        for j in range(len(yVec)):
            surfProj[i,j] = pot_energy_model(xVec[j], yVec[i], par)

    return surfProj 
    
    
#%
def stateTransitMat(tf,x0,par,varEqns_model,fixed_step=0): 
    """
    Returns state transition matrix, the trajectory, and the solution of the
    variational equations over a length of time

    In particular, for periodic solutions of % period tf=T, one can obtain
    the monodromy matrix, PHI(0,T).
    
    Parameters
    ----------
    tf : float
        final time for the integration

    x0 : float
        initial condition

    par : float (list)
        model parameters

    varEqns_model : function name
        function that returns the variational equations of the dynamical system

    Returns
    -------
    t : 1d numpy array
        solution time

    x : 2d numpy array
        solution of the phase space coordinates

    phi_tf : 2d numpy array
        state transition matrix at the final time, tf

    PHI : 1d numpy array,
        solution of phase space coordinates and corresponding tangent space coordinates

    """

    N = len(x0)  # N=4 
    RelTol=3e-14
    AbsTol=1e-14  
    tf = tf[-1]
    if fixed_step == 0:
        TSPAN = [ 0 , tf ] 
    else:
        TSPAN = np.linspace(0, tf, fixed_step)
    PHI_0 = np.zeros(N+N**2)
    PHI_0[0:N**2] = np.reshape(np.identity(N),(N**2)) #initial condition for state transition matrix
    PHI_0[N**2:N+N**2] = x0                           # initial condition for trajectory

    
    f = lambda t,PHI: varEqns_model(t,PHI,par) # Use partial in order to pass parameters to function
    soln = solve_ivp(f, TSPAN, list(PHI_0), method='RK45', dense_output=True, \
                     events = None, rtol=RelTol, atol=AbsTol)
    t = soln.t
    PHI = soln.y
    PHI = PHI.transpose()
    x = PHI[:,N**2:N+N**2]		   # trajectory from time 0 to tf
    phi_tf = np.reshape(PHI[len(t)-1,0:N**2],(N,N)) # state transition matrix, PHI(O,tf)

    
    return t,x,phi_tf,PHI



#%%
def turningPoint_configdiff(begin1,begin2, get_coord_model, pot_energy_model, varEqns_model, \
                            configdiff_model, ham2dof_model, half_period_model, \
                            guess_coords_model, plot_iter_orbit_model, par, \
                            e, n, n_turn, show_itrsteps_plots, po_fam_file):
    """
    turningPoint computes the periodic orbit of target energy using turning point based on configuration difference method. 
    
    Given 2 inital conditions begin1, begin2, the periodic orbit is assumed to exist between begin1, begin2 such that
    trajectories with inital conditions begin1, begin2 are turning in different directions,
    which gives different signs(+ or -) for configuration difference 

    Parameters
    ----------
    begin1 : function name
        guess initial condition 1 for the unstable periodic orbit

    begin2 : function name
        guess initial condition 2 for the unstable periodic orbit

    get_coord_model : function name
        function that returns the phase space coordinate for a given x/y value and total energy E
        
    guess_coord_model : function name
        function that returns the coordinates as guess for the next iteration of the 
    turning point 
        
    ham2dof_model : function name
        function that returns the Hamiltonian vector field at an input phase space coordinate
        and time

    half_period_model : function name
        function that returns the event criteria in terms of the coordinate that is set to zero
        for half-period of the unstable periodic orbit

    pot_energy_model : function name
        function that returns the potential energy of Hamiltonian

    varEqns_model : function name
        function that returns the variational equations of the dynamical system

    plot_iter_orbit_model : function name
        function to plot the computed orbit in the 3D phase space of 2 position and 1 momentum
        coordinate
    
    par: float (list)
        model parameters

    e: float 
        total energy of the system

    n: int
        number of intervals that is divided bewteen the 2 initial guesses begin1 and begin2
        
    n_turn: int
        nth turning point that is used to define the dot product

    show_itrsteps_plots: logical
        flag (True or False) to show iteration of the UPOs in plots

    po_fam_file : function name
        file name to save the members in the family of the unstable periodic orbits      
    Returns
    -------
    x0po : 1d numpy array 
        Initial condition of the target unstable periodic orbit
        
    T : float
        Time period of the target unstable periodic orbit
        
    energyPO : float
        Energy of the target unstable periodic orbit. 
    """
    axis_fs = 15
    
    guess1 = begin1
    guess2 = begin2
    MAXiter = 30
    dum = np.zeros(((n+1)*MAXiter ,7))
    result = np.zeros(((n+1),4))  # record data for each iteration
    result2 = np.zeros(((n+1)*MAXiter ,4))
    np.set_printoptions(precision=17,suppress=True)
    x0po = np.zeros((MAXiter ,4))
    i_turn = np.zeros((MAXiter ,1))
    T = np.zeros((MAXiter ,1))
    energyPO = np.zeros((MAXiter ,1))
    i_iter = 0
    iter_diff =0  # for counting the correct index

    while i_iter< MAXiter and n_turn < 5:
        for i in range(0,n+1):
            # the x difference between guess1 and each guess is recorded in "result" matrix
            
            
            xguess, yguess = guess_coords_model(guess1, guess2, i, n, e, get_coord_model, par)
            
            guess = [xguess,yguess,0, 0]
            coordinate_diff1, coordinate_diff2 = configdiff_model(guess1, guess, ham2dof_model, \
                                                                  half_period_model, \
                                                                  n_turn, par)
            result[i,0] = np.sign(coordinate_diff1)
            result[i,1] = guess[0]
            result[i,2] = guess[1]
            result[i,3] = np.sign(coordinate_diff2)
        for i in range(1,n+1):
            if np.sign(result[i,0]) != np.sign(result[i,3]) and \
                np.sign(result[i-1,0]) == np.sign(result[i-1,3]):
                i_turn[i_iter] = i


        # if the follwing condition holds, we can zoom in to a smaller interval and 
        # continue our procedure
        if i_turn[i_iter] > 0:
            index = int(i_turn[i_iter])
            guesspo  = [result[index-1,1],result[index-1,2],0,0]
            print("Our guess for the periodic orbit is",guesspo)
            x0po[i_iter,:] = guesspo[:]
            TSPAN = [0,30]
            RelTol = 3.e-10
            AbsTol = 1.e-10
            f = lambda t,x: ham2dof_model(t,x,par) 
            soln = solve_ivp(f, TSPAN, guesspo,method='RK45',dense_output=True, \
                             events = lambda t,x: half_period_model(t,x,par), \
                             rtol=RelTol, atol=AbsTol)
            te = soln.t_events[0]
            tt = [0,te[1]]
            
            t,x,phi_t1,PHI = stateTransitMat(tt, guesspo, par, varEqns_model)
            
            T[i_iter]= tt[-1]*2
            print("period is%s " %T[i_iter])
            energy = np.zeros(len(x))
            #print(len(t))
            for j in range(len(t)):
                energy[j] = get_total_energy(x[j,:], pot_energy_model, par)
            energyPO[i_iter]= np.mean(energy)
            
            if show_itrsteps_plots: # show iteration of the UPOs in plots
                ax = plt.gca(projection='3d')
                plot_iter_orbit_model(x, ax, e, par)
            
                plt.grid()
                plt.show()

            guess2 = np.array([result[index,1], result[index,2],0,0])
            guess1 = np.array([result[index-1,1], result[index-1,2],0,0])
            
            iter_diff =0
        # If the if condition does not hold, it indicates that the interval we picked for 
        # performing 'configration difference' is wrong and it needs to be changed.
        else:
            # return to the previous iteration that dot product works
            #iteration------i------i+1---------------i+2----------------i+3---------------------i+4
            # succ------succ-------------unsucc(return to i+1,n_turn+1)
            #   if  --------succ--------         
            #   else -----unsucc(return to i+1, n_turn+2)
            #   unscc---------------unsucc------------   unsucc(return to i, n_turn+3)        
            #
            #
            #
            # we take a larger interval so that it contains the true value of the initial 
            # condition and avoids to reach the limitation of the configration difference
            iter_diff = iter_diff +1
            if iter_diff > 1:
                # return to the iteration that is before the previous 
                print("Warning: the result after this iteration may not be accurate, \
                      try to increase the number of intervals or use other ways ")
                break
            n_turn = n_turn+1
            print("nth turningpoint we pick is ", n_turn)
            index = int((n+1)*(i_iter-iter_diff)+i_turn[i_iter-iter_diff])
            print("index is ", index)
            xguess2=result2[index+iter_diff,1]
            yguess2 =result2[index+iter_diff,2]
            xguess1=result2[index-1-iter_diff,1]
            yguess1 = result2[index-1-iter_diff,2]
            guess2 = np.array([xguess2, yguess2,0,0])
            guess1 = np.array([xguess1, yguess1,0,0])
                    

        
        print(result)
        print("nth turningpoint we pick is ", n_turn)
        i_iter= i_iter+1
        print(i_iter)

    
    
    dum = np.concatenate((x0po,T, energyPO),axis=1)
    np.savetxt(po_fam_file.name,dum,fmt='%1.16e')
    return x0po, T,energyPO



    



