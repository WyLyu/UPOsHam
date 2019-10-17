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
#    F = lambda x: func_vec_field_eq_pt(model,x,par)
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
                            e, n, n_turn, po_fam_file):
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
    #y_PES = get_y(begin1[0], e,par)
    #y_turning = begin1[1]
    #toler = math.sqrt((y_PES-y_turning)**2)
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
    #while toler < 1e-6 or i_iter< MAXiter:
    while i_iter< MAXiter and n_turn < 5:
        #y_PES = -get_y(guess1[0], e,par)
        #y_turning = guess1[1]
        #toler = math.sqrt((y_PES-y_turning)**2)
        for i in range(0,n+1):
            # the x difference between guess1 and each guess is recorded in "result" matrix
            
#            if model == 'uncoupled':
#                h = (guess2[0] - guess1[0])*i/n
#                print("h is ",h)
#                xguess = guess1[0]+h
#                f = lambda y: get_coord_model(xguess,y,e,par)
#                yanalytic = math.sqrt((e +0.5*par[3]*xguess**2-0.25*par[4]*xguess**4)/(0.5*par[1])) #uncoupled
#                yguess = optimize.newton(f,yanalytic)   # to find the x coordinate for a given y
#            elif model == 'coupled':
#                h = (guess2[0] - guess1[0])*i/n
#                print("h is ",h)
#                xguess = guess1[0]+h
#                f = lambda y: get_coord_model(xguess,y,e,par)
#                yanalytic = math.sqrt(2/(par[1]+par[6]))*(-math.sqrt( e +0.5*par[3]* xguess**2 - \
#                                     0.25*par[4]*xguess**4 -0.5*par[6]* xguess**2 + \
#                                     (par[6]*xguess)**2/(2*(par[1] +par[6]) )) + 
#                                    par[6]/(math.sqrt(2*(par[1]+par[6])) )*xguess ) #coupled
#                yguess = optimize.newton(f,yanalytic) 
#            elif model == 'deleonberne':
#                h = (guess2[1] - guess1[1])*i/n # h is defined for dividing the interval
#                print("h is ",h)
#                yguess = guess1[1]+h
#                f = lambda x: get_coord_model(x,yguess,e,par)
#                xguess = optimize.newton(f,-0.2)   # to find the x coordinate for a given y 
#            else:
#                print("The model you are chosen does not exist")
#                break
            
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
            
            ax = plt.gca(projection='3d')
            plot_iter_orbit_model(x, ax, e, par)
            
#            if model== 'uncoupled':
#                ax.plot(x[:,0],x[:,1],x[:,3],'-')
#                ax.plot(x[:,0],x[:,1],-x[:,3],'--')
#                ax.scatter(x[0,0],x[0,1],x[0,3],s=20,marker='*')
#                ax.scatter(x[-1,0],x[-1,1],x[-1,3],s=20,marker='o')
#            elif model== 'coupled':
#                ax.plot(x[:,0],x[:,1],x[:,3],'-')
#                ax.plot(x[:,0],x[:,1],-x[:,3],'--')
#                ax.scatter(x[0,0],x[0,1],x[0,3],s=20,marker='*')
#                ax.scatter(x[-1,0],x[-1,1],x[-1,3],s=20,marker='o')
#            elif model == 'deleonberne':
#                ax.plot(x[:,0],x[:,1],x[:,2],'-')
#                ax.plot(x[:,0],x[:,1],-x[:,2],'--')
#                ax.scatter(x[0,0],x[0,1],x[0,2],s=20,marker='*')
#                ax.scatter(x[-1,0],x[-1,1],x[-1,2],s=20,marker='o')
#            else:
#                print("The model you are chosen does not exist ") 
#                
#            ax.set_xlabel('$x$', fontsize=axis_fs)
#            ax.set_ylabel('$y$', fontsize=axis_fs)
#            ax.set_zlabel('$p_y$', fontsize=axis_fs)
#            ax.set_title('$\Delta E$ = %e' %(np.mean(energy) - par[2] ), fontsize=axis_fs)
#            ax.set_xlim(-1, 1)
#            ax.set_ylim(-1, 1)
            #x_turn= x[-1,0]  # x coordinate of turning point
            #y_turn= x[-1,1] # y coordinate of turning point
            #y_PES = -get_y(x_turn,e,par)
            #toler = math.sqrt((y_PES-y_turn)**2)
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
            #index = int((n+1)*(re_iter-1)+i_turn[re_iter-1])
            index = int((n+1)*(i_iter-iter_diff)+i_turn[i_iter-iter_diff])
            print("index is ", index)
            xguess2=result2[index+iter_diff,1]
            yguess2 =result2[index+iter_diff,2]
            xguess1=result2[index-1-iter_diff,1]
            yguess1 = result2[index-1-iter_diff,2]
            guess2 = np.array([xguess2, yguess2,0,0])
            guess1 = np.array([xguess1, yguess1,0,0])
                    

        
        #print("tolerance is ", toler)
        print(result)
        print("nth turningpoint we pick is ", n_turn)
        i_iter= i_iter+1
        print(i_iter)

    
    
    dum = np.concatenate((x0po,T, energyPO),axis=1)
    np.savetxt(po_fam_file.name,dum,fmt='%1.16e')
    return x0po, T,energyPO



    

#%% get potential energy
#def get_potential_energy(model,x,y,par):
#    """ enter the definition of the potential energy function  
#    """      
#    
#    if model == 'uncoupled':
#        pot_energy =  -0.5*par[3]*x**2+0.25*par[4]*x**4 +0.5*par[5]*y**2
#    elif model == 'coupled':
#        pot_energy =  -0.5*par[3]*x**2+0.25*par[4]*x**4 +0.5*par[5]*y**2+0.5*par[6]*(x-y)**2
#    elif model== 'deleonberne':
#        pot_energy = par[3]*( 1 - math.e**(-par[4]*x) )**2 + 4*y**2*(y**2 - 1)*math.e**(-par[5]*par[4]*x) + par[2]
#    else:
#        print("The model you are chosen does not exist, enter the definition of the potential ernergy")
#                
#    return pot_energy

#%
#def func_vec_field_eq_pt(model,x,par):
#    """ vecor field(same as pxdot, pydot), used to find the equilibrium points of the system.
#    """
#    if model == 'uncoupled':
#        dVdx = -par[3]*x[0]+par[4]*(x[0])**3
#        dVdy = par[5]*x[1]
#    elif model == 'coupled':
#        dVdx = (-par[3]+par[6])*x[0]+par[4]*(x[0])**3-par[6]*x[1]
#        dVdy = (par[5]+par[6])*x[1]-par[6]*x[0]
#    elif model == 'deleonberne':
#        dVdx = -2*par[3]*par[4]*math.e**(-par[4]*x[0])*(math.e**(-par[4]*x[0]) - 1) - 4*par[5]*par[4]*x[1]**2*(x[1]**2 - 1)*math.e**(-par[5]*par[4]*x[0])
#        dVdy = 8*x[1]*(2*x[1]**2 - 1)*math.e**(-par[5]*par[4]*x[0])
#    else:
#        print("The model you are chosen does not exist, enter the definition of Hamilton's equations")
#    F = [-dVdx, -dVdy]
#    return F


#%
#def get_eq_pts(eqNum,model, par):
#    """
#    GET_EQ_PTS_BP solves the saddle center equilibrium point for a system with
#    KE + PE. 
#     H = 1/2*px^2+omega/2*py^2 -(1/2)*alpha*x^2+1/4*beta*x^4+ omega/2y^2 with alpha > 0
#    --------------------------------------------------------------------------
#       Uncouled potential energy surface notations:
#    
#               Well (stable, EQNUM = 2)    
#    
#                   Saddle (EQNUM=1)
#    
#               Well (stable, EQNUM = 3)    
#    
#    --------------------------------------------------------------------------
#       
#    
#    """
#    #fix the equilibrium point numbering convention here and make a
#    #starting guess at the solution
#    if 	eqNum == 1:
#        
#        x0 = [0, 0]                  # EQNUM = 1, saddle  
#    elif 	eqNum == 2: 
#        if model == 'uncoupled':
#            eqPt = [+math.sqrt(par[3]/par[4]),0]    # EQNUM = 2, stable
#            return eqPt
#        elif model == 'coupled':
#            eqPt = [+math.sqrt(par[3]-par[6]/par[4]),0] # EQNUM = 2, stable
#            return eqPt
#        elif model== 'deleonberne':
#            eqPt = [0, 1/math.sqrt(2)]    # EQNUM = 2, stable
#            return eqPt
#        else:
#            print("The model you are chosen does not exist, enter the definition of Hamilton's equations")
#    elif 	eqNum == 3:
#        if model == 'uncoupled':
#            eqPt = [-math.sqrt(par[3]/par[4]),0]    # EQNUM = 2, stable
#            return eqPt
#        elif model == 'coupled':
#            eqPt = [-math.sqrt(par[3]-par[6]/par[4]),0] # EQNUM = 2, stable
#            return eqPt
#        elif model== 'deleonberne':
#            eqPt = [0, -1/math.sqrt(2)]    # EQNUM = 2, stable
#            return eqPt
#        else:
#            print("The model you are chosen does not exist, enter the definition of Hamilton's equations")
#    
#    # F(xEq) = 0 at the equilibrium point, solve using in-built function
#    F = lambda x: func_vec_field_eq_pt(model,x,par)
#    eqPt = fsolve(F,x0, fprime=None) # Call solver
#    return eqPt


#def stateTransitMat(tf,x0,par,model,fixed_step=0): 
#    """
#    function [x,t,phi_tf,PHI] =
#    stateTransitionMatrix_boatPR(x0,tf,R,OPTIONS,fixed_step)
#    
#    Gets state transition matrix, phi_tf = PHI(0,tf), and the trajectory 
#    (x,t) for a length of time, tf.  
#    
#    In particular, for periodic solutions of % period tf=T, one can obtain 
#    the monodromy matrix, PHI(0,T).
#    """
#    
#
#    N = len(x0)  # N=4 
#    RelTol=3e-14
#    AbsTol=1e-14  
#    tf = tf[-1]
#    if fixed_step == 0:
#        TSPAN = [ 0 , tf ] 
#    else:
#        TSPAN = np.linspace(0, tf, fixed_step)
#    PHI_0 = np.zeros(N+N**2)
#    PHI_0[0:N**2] = np.reshape(np.identity(N),(N**2)) # initial condition for state transition matrix
#    PHI_0[N**2:N+N**2] = x0                    # initial condition for trajectory
#
#    
#    f = lambda t,PHI: varEqns(t,PHI,par, model) # Use partial in order to pass parameters to function
#    soln = solve_ivp(f, TSPAN, list(PHI_0),method='RK45',dense_output=True, events = None,rtol=RelTol, atol=AbsTol)
#    t = soln.t
#    PHI = soln.y
#    PHI = PHI.transpose()
#    x = PHI[:,N**2:N+N**2]		   # trajectory from time 0 to tf
#    phi_tf = np.reshape(PHI[len(t)-1,0:N**2],(N,N)) # state transition matrix, PHI(O,tf)
#
#    
#    return t,x,phi_tf,PHI


#%
#def varEqns(t,PHI,par,model):
#    """
#    PHIdot = varEqns_bp(t,PHI) 
#    
#    This here is a preliminary state transition, PHI(t,t0),
#    matrix equation attempt for a ball rolling on the surface, based on...
#    
#    d PHI(t, t0)
#    ------------ =  Df(t) * PHI(t, t0)
#        dt
#    
#    """
#    phi = PHI[0:16]
#    phimatrix  = np.reshape(PHI[0:16],(4,4))
#    x,y,px,py = PHI[16:20]
#    
#    
#    if model == 'uncoupled':
#        # The first order derivative of the Hamiltonian.
#        dVdx = -par[3]*x+par[4]*x**3
#        dVdy = par[5]*y
#    
#        # The following is the Jacobian matrix 
#        d2Vdx2 = -par[3]+par[4]*3*x**2
#            
#        d2Vdy2 = par[5]
#    
#        d2Vdydx = 0
#    elif model == 'coupled':
#        # The first order derivative of the Hamiltonian.
#        dVdx = (-par[3]+par[6])*x+par[4]*x**3-par[6]*y
#        dVdy = (par[5]+par[6])*y-par[6]*x
#
#        # The following is the Jacobian matrix 
#        d2Vdx2 = -par[3]+par[6]+par[4]*3*x**2
#            
#        d2Vdy2 = par[5]+par[6]
#    
#        d2Vdydx = -par[6]
#    elif model == 'deleonberne':
#        # The first order derivative of the Hamiltonian.
#        dVdx = - 2*par[3]*par[4]*math.e**(-par[4]*x)*(math.e**(-par[4]*x) - 1) - 4*par[5]*par[4]*y**2*(y**2 - 1)*math.e**(-par[5]*par[4]*x) 
#        dVdy = 8*y*(2*y**2 - 1)*math.e**(-par[5]*par[4]*x)
#    
#        # The following is the Jacobian matrix 
#        d2Vdx2 = - ( 2*par[3]*par[4]**2*( math.e**(-par[4]*x) - 2.0*math.e**(-2*par[4]*x) ) - 4*(par[5]*par[4])**2*x**2*(y**2 - 1)*math.e**(-par[5]*par[4]*x) )
#            
#        d2Vdy2 = 8*(6*y**2 - 1)*math.e**( -par[4]*par[5]*x )
#    
#        d2Vdydx = -8*y*par[4]*par[5]*math.e**( -par[4]*par[5]*x )*(2*y**2 - 1)
#        
#    else:
#        print("The model you are chosen does not exist")
#    
#    
#    d2Vdxdy = d2Vdydx    
#
#    Df    = np.array([[  0,     0,    par[0],    0],
#              [0,     0,    0,    par[1]],
#              [-d2Vdx2,  -d2Vdydx,   0,    0],
#              [-d2Vdxdy, -d2Vdy2,    0,    0]])
#
#    
#    phidot = np.matmul(Df, phimatrix) # variational equation
#
#    PHIdot        = np.zeros(20)
#    PHIdot[0:16]  = np.reshape(phidot,(1,16)) 
#    PHIdot[16]    = px/par[0]
#    PHIdot[17]    = py/par[1]
#    PHIdot[18]    = -dVdx 
#    PHIdot[19]    = -dVdy
#    
#    return list(PHIdot)

#%
#def half_period(t,x,model):
#    """
#    Return the turning point where we want to stop the integration                           
#    
#    pxDot = x[0]
#    pyDot = x[1]
#    xDot = x[2]
#    yDot = x[3]
#    """
#    terminal = True
#    # The zero can be approached from either direction
#    direction = 0 #0: all directions of crossing
#    if model =='uncoupled':
#        return x[3]
#    elif model =='coupled':
#        return x[3]
#    elif model =='deleonberne':
#        return x[2]
#    else:
#        print("specify an event for integration, either choose to return p_x or to return p_y")


#% Hamilton's equations
#def Ham2dof(model,t,x, par):
#    """ Enter the definition of Hamilton's equations, used for ploting trajectories later.
#    """
#    xDot = np.zeros(4)
#    
#    if model == 'uncoupled':
#        dVdx = -par[3]*x[0]+par[4]*(x[0])**3
#        dVdy = par[5]*x[1]
#    elif model == 'coupled':
#        dVdx = (-par[3]+par[6])*x[0]+par[4]*(x[0])**3-par[6]*x[1]
#        dVdy = (par[5]+par[6])*x[1]-par[6]*x[0]
#    elif model == 'deleonberne':
#        dVdx = -2*par[3]*par[4]*math.e**(-par[4]*x[0])*(math.e**(-par[4]*x[0]) - 1) - 4*par[5]*par[4]*x[1]**2*(x[1]**2 - 1)*math.e**(-par[5]*par[4]*x[0])
#        dVdy = 8*x[1]*(2*x[1]**2 - 1)*math.e**(-par[5]*par[4]*x[0])
#    else:
#        print("The model you are chosen does not exist, enter the definition of Hamilton's equations")
#        
#    xDot[0] = x[2]/par[0]
#    xDot[1] = x[3]/par[1]
#    xDot[2] = -dVdx 
#    xDot[3] = -dVdy
#    return list(xDot)    

