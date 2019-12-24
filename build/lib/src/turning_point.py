# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 17:34:06 2019

@author: Wenyang Lyu and Shibabrat Naik
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import math
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import fsolve
import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'


#%%

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
def state_transit_matrix(tf,x0,par,variational_eqns_model,fixed_step=0): 
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

    variational_eqns_model : function name
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

    
    f = lambda t,PHI: variational_eqns_model(t,PHI,par) # Use partial in order to pass parameters to function
    soln = solve_ivp(f, TSPAN, list(PHI_0), method='RK45', dense_output=True, \
                     events = None, rtol=RelTol, atol=AbsTol)
    t = soln.t
    PHI = soln.y
    PHI = PHI.transpose()
    x = PHI[:,N**2:N+N**2]		   # trajectory from time 0 to tf
    phi_tf = np.reshape(PHI[len(t)-1,0:N**2],(N,N)) # state transition matrix, PHI(O,tf)

    
    return t,x,phi_tf,PHI


#%%
def dotproduct(guess1, guess2,n_turn, ham2dof_model, half_period_model, \
                varEqns_model, par):
    """
    Returns x,y coordinates of the turning points for guess initial conditions guess1, guess2 and the defined product product for the 2 turning points
    
    Uses turning point method(defined a dot product form before the "actual" turning point.)
    
    Parameters
    ----------
    guess1 : 1d numpy array 
        guess initial condition 1 for the unstable periodic orbit

    guess2 : 1d numpy array 
        guess initial condition 2 for the unstable periodic orbit

    n_turn : int
        nth turning point that is used to define the dot product

    ham2dof_model : function name
        function that returns the Hamiltonian vector field at an input phase space coordinate
        and time
                    
    variational_eqns_model : function name
        function that returns the variational equations of the dynamical system

    par : float (list)
        model parameters

    Returns
    -------
    x_turn1 : float
        x coordinate of the turning point with initional condition guess1

    x_turn2 : float
        x coordinate of the turning point with initional condition guess2

    y_turn1 : float
        y coordinate of the turning point with initional condition guess1

    y_turn2 : float
        y coordinate of the turning point with initional condition guess2

    dotproduct : float
        value of the dot product

    """

    TSPAN = [0,40]
    RelTol = 3.e-10
    AbsTol = 1.e-10 
    f1 = lambda t,x: ham2dof_model(t,x,par) 
    soln1 = solve_ivp(f1, TSPAN, guess1, method='RK45', dense_output=True, \
                      events = lambda t,x: half_period_model(t,x,par),rtol=RelTol, atol=AbsTol)
    te1 = soln1.t_events[0]
    t1 = [0,te1[n_turn]]#[0,te1[1]]
    turn1 = soln1.sol(t1)
    x_turn1 = turn1[0,-1] 
    y_turn1 = turn1[1,-1] 
    t,xx1,phi_t1,PHI = state_transit_matrix(t1,guess1,par,variational_eqns_model)
    x1 = xx1[:,0]
    y1 = xx1[:,1]
    p1 = xx1[:,2:]
    p_perpendicular_1 = math.sqrt(np.dot(p1[-3,:],p1[-3,:]))*p1[-2,:] - np.dot(p1[-2,:],p1[-3,:])*p1[-3,:]
    f2 = lambda t,x: ham2dof_model(t,x,par)  
    soln2 = solve_ivp(f2, TSPAN, guess2,method='RK45',dense_output=True, \
                      events = lambda t,x: half_period_model(t,x,par),rtol=RelTol, atol=AbsTol)
    te2 = soln2.t_events[0]
    t2 = [0,te2[n_turn]]#[0,te2[1]]
    turn2 = soln1.sol(t2)
    x_turn2 = turn2[0,-1] 
    y_turn2 = turn2[1,-1] 
    t,xx2,phi_t1,PHI = state_transit_matrix(t2,guess2,par,variational_eqns_model)
    x2 = xx2[:,0]
    y2 = xx2[:,1]
    p2 = xx2[:,2:]
    p_perpendicular_2 = math.sqrt(np.dot(p2[-3,:],p2[-3,:]))*p2[-2,:] - np.dot(p2[-2,:],p2[-3,:])*p2[-3,:]
    dotproduct = np.dot(p_perpendicular_1,p_perpendicular_2)
    print("Initial guess1%s, initial guess2%s, dot product is%s" %(guess1,guess2,dotproduct))
    
    return x_turn1,x_turn2,y_turn1,y_turn2, dotproduct


#%%
def turningPoint(begin1, begin2, get_coord_model, guess_coords_model, ham2dof_model, \
                 half_period_model, variational_eqns_model, pot_energy_model, plot_iter_orbit_model, par, \
                 e, n, n_turn, show_itrsteps_plots, po_fam_file):
    """
    turningPoint computes the periodic orbit of target energy using turning point method.
    
    Given 2 inital conditions begin1, begin2, the periodic orbit is assumed to exist between begin1, begin2 such that
    trajectories with inital conditions begin1, begin2 are turning in different directions,
    which results in a negative value of the dot product

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

    variational_eqns_model : function name
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
    result = np.zeros(((n+1),3))  # record data for each iteration
    result2 = np.zeros(((n+1)*MAXiter ,3)) # record all data for every iteration
    x0po = np.zeros((MAXiter ,4))
    i_turn = np.zeros((MAXiter ,1))
    T = np.zeros((MAXiter ,1))
    energyPO = np.zeros((MAXiter ,1))
    iter = 0
    iter_diff =0  # for counting the correct index
    
    while iter < MAXiter and n_turn < 10:

        for i in range(0,n+1):
            
            xguess, yguess = guess_coords_model(guess1, guess2, i, n, e, get_coord_model, par)
            
            guess = [xguess,yguess,0, 0]
            x_turn1,x_turn2,y_turn1,y_turn2, dotpro = dotproduct(guess1, guess, n_turn, \
                                                                 ham2dof_model, \
                                                                 half_period_model, \
                                                                 variational_eqns_model, \
                                                                 par)
            result[i,0] = dotpro
            result[i,1] = guess[0]
            result[i,2] = guess[1]
            result2[(n+1)*iter+i,:] = result[i,:]
            
        i_turn_iter = 0
        for i in range(0,n+1):
            #  we record the sign change for each pair of inital conditions
            # i_turn_iter is the coordinate which sign changes from positive to negative
            # we only want the first i_turn_iter terms to have positve sign and the rest of n-i_turn_iter+1 terms to have negative signs to avoid error
            #
            if np.sign(result[i,0]) <0 and np.sign(result[i-1,0]) >0:
                i_turn[iter] = i
                i_turn_iter = int(i_turn[iter])
                check = np.sign(result[:,0])
                check_same= sum(check[0:i_turn_iter])
                check_diff= sum(check[i_turn_iter:])
                print(check_same == i_turn[iter])
                print(check_diff == -n+i_turn[iter]-1)
            

        if check_same == i_turn[iter] and check_diff == -n+i_turn[iter]-1 and i_turn_iter>0:
            # if the follwing condition holds, we can zoom in to a smaller interval and continue our procedure
            index = int(i_turn[iter])
            guesspo  = [result[index-1,1],result[index-1,2],0,0]
            print("Our guess of the inital condition is", guesspo)
            x0po[iter,:] = guesspo[:]
            TSPAN = [0,10]
            RelTol = 3.e-10
            AbsTol = 1.e-10
            f = lambda t,x: ham2dof_model(t,x,par) 
            soln = solve_ivp(f, TSPAN, guesspo,method='RK45', dense_output=True, \
                             events = lambda t,x: half_period_model(t,x,par),rtol=RelTol, atol=AbsTol)
            te = soln.t_events[0]
            tt = [0,te[1]]
            t,x,phi_t1,PHI = state_transit_matrix(tt,guesspo,par,variational_eqns_model)
            T[iter] = tt[-1]*2
            print("period is%s " %T[iter])
            energy = np.zeros(len(x))
            #print(len(t))
            for j in range(len(t)):
                energy[j] = get_total_energy(x[j,:], pot_energy_model, par)
            energyPO[iter] = np.mean(energy)
            
            if show_itrsteps_plots: # show iteration of the UPOs in plots
                ax = plt.gca(projection='3d')
                plot_iter_orbit_model(x, ax, e, par)
                plt.grid()
                plt.show()


            
            guess2 = np.array([result[index,1], result[index,2],0,0])
            guess1 = np.array([result[index-1,1], result[index-1,2],0,0])
            iter_diff =0
        # If the if condition does not hold, it indicates that the interval we picked for performing 'dot product' is wrong and it needs to be changed.
        else:
            # return to the previous iteration that dot product works
            #iteration------i------i+1---------------i+2----------------i+3-----------------------i+4
            #            succ------succ-------------unsucc(return to i+1,n_turn+1)
            #                                              if  --------succ--------         
            #                                              else -----unsucc(return to i+1, n_turn+2)
            #                                       unscc---------------unsucc------------   unsucc(return to i, n_turn+3)        
            #
            #
            #
            # we take a larger interval so that it contains the true value of the initial condition and avoids to reach the limitation of the dot product
            
            iter_diff = iter_diff +1
            #for k in range(iter):
            if iter_diff> 1:
                #iter_diff = iter_diff+1   # return to the iteration that is before the previous 
                print("Warning: the result after this iteration may not be accurate, try to increase the number of intervals or use other ways ")
                break
            n_turn = n_turn+1
            print("the nth turning point we pick is ", n_turn)
            index = int((n+1)*(iter-iter_diff)+i_turn[iter-iter_diff])
            print("index is ", index)
            xguess2=result2[index+iter_diff,1]
            yguess2 =result2[index+iter_diff,2]
            xguess1=result2[index-1-iter_diff,1]
            yguess1 = result2[index-1-iter_diff,2]
            guess2 = np.array([xguess2, yguess2,0,0])
            guess1 = np.array([xguess1, yguess1,0,0])
                    

        
        print(result)
        print("the nth turning point we pick is ", n_turn)
        iter = iter +1

    end = MAXiter

    for i in range(MAXiter):
        if x0po[i,0] ==0 and x0po[i-1,0] !=0:
            end = i

    x0po = x0po[0:end,:]
    T = T[0:end]
    energyPO = energyPO[0:end]   
    
    dum =np.concatenate((x0po,T, energyPO),axis=1)
    np.savetxt(po_fam_file.name,dum,fmt='%1.16e')
    return x0po, T,energyPO


