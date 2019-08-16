# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 10:19:47 2019

@author: wl16298
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint,quad,trapz,solve_ivp, ode
import math
from IPython.display import Image # for Notebook
from IPython.core.display import HTML
from mpl_toolkits.mplot3d import Axes3D
from numpy import linalg as LA
import scipy.linalg as linalg
from scipy.optimize import fsolve
import time
from functools import partial
import matplotlib as mpl

from pylab import rcParams
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'


#%%
def get_eq_pts_uncoupled(eqNum, parameters):
    #GET_EQ_PTS_BP solves the saddle center equilibrium point for a system with
    #KE + PE. 
    # H = 1/2*px^2+omega/2*py^2 -(1/2)*alpha*x^2+1/4*beta*x^4+ omega/2y^2 with alpha > 0
    #--------------------------------------------------------------------------
    #   Uncouled potential energy surface notations:
    #
    #           Well (stable, EQNUM = 2)    
    #
    #               Saddle (EQNUM=1)
    #
    #           Well (stable, EQNUM = 3)    
    #
    #--------------------------------------------------------------------------
    #   
    
    
    #fix the equilibrium point numbering convention here and make a
    #starting guess at the solution
    if 	eqNum == 1:
        
        x0 = [0, 0];                  # EQNUM = 1, saddle  
    elif 	eqNum == 2: 
        eqPt = [+math.sqrt(parameters[3]/parameters[4]),0]    # EQNUM = 2, stable
        return eqPt
    elif 	eqNum == 3:
        eqPt = [-math.sqrt(parameters[3]/parameters[4]),0]   # EQNUM = 3, stable
        return eqPt
    
    
    # F(xEq) = 0 at the equilibrium point, solve using in-built function
    # enter the definition of the potential energy function
    def func_vec_field_eq_pt(x,par):
        x1, x2 = x
        dVdx = -par[3]*x1+par[4]*x1**3
        
        dVdy = par[5]*x2
    
        F = [-dVdx, -dVdy]
        return F

    eqPt = fsolve(func_vec_field_eq_pt,x0, fprime=None,args=(parameters,)); # Call solver
    return eqPt

#%%
# enter the definition of the potential energy function
def get_potential_energy(x,y,par):
            
    pot_energy =  -0.5*par[3]*x**2+0.25*par[4]*x**4 +0.5*par[5]*y**2
                
    return pot_energy


#%%
def get_total_energy_uncoupled(orbit, parameters):

#   get_total_energy_uncoupled computes the total energy of an input orbit
#   (represented as M x N with M time steps and N = 4, dimension of phase
#   space for the model) for the 2 DoF Uncoupled potential.
# 
#   Orbit can be different initial conditions for the periodic orbit of
#   different energy. When trajectory is input, the output energy is mean.
#
    

    x  = orbit[0]
    y  = orbit[1]
    px = orbit[2]
    py = orbit[3]
    
    
    e = (0.5*parameters[0])*(px**2) + (0.5*parameters[1])*(py**2) +  get_potential_energy(x, y,parameters)   
        
    return e


#%%
def get_pot_surf_proj(xVec, yVec,par):            

    resX = np.size(xVec)
    resY = np.size(xVec)
    surfProj = np.zeros([resX, resY])
    for i in range(len(xVec)):
        for j in range(len(yVec)):
            surfProj[i,j] = get_potential_energy(xVec[j], yVec[i],par)

    return surfProj    


#%%
def get_y(x, V,par):
    # this function returns the initial position of y-coordinate on the potential energy surface(PES) for a specific energy V.
    ypositive=math.sqrt((V +0.5*par[3]*x**2-0.25*par[4]*x**4)/(0.5*par[1]))
    #ynegative=-math.sqrt((V +0.5*par[3]*x**2-0.25*par[4]*x**4)/(0.5*par[1]))
    
    return ypositive


#%%
def stateTransitMat_uncoupled(tf,x0,parameters,fixed_step=0): 

    # function [x,t,phi_tf,PHI] =
    # stateTransitionMatrix_boatPR(x0,tf,R,OPTIONS,fixed_step)
    #
    # Gets state transition matrix, phi_tf = PHI(0,tf), and the trajectory 
    # (x,t) for a length of time, tf.  
    # 
    # In particular, for periodic solutions of % period tf=T, one can obtain 
    # the monodromy matrix, PHI(0,T).
    

    N = len(x0);  # N=4 
    RelTol=3e-14
    AbsTol=1e-14  
    tf = tf[-1];
    if fixed_step == 0:
        TSPAN = [ 0 , tf ]; 
    else:
        TSPAN = np.linspace(0, tf, fixed_step)
    PHI_0 = np.zeros(N+N**2)
    PHI_0[0:N**2] = np.reshape(np.identity(N),(N**2)); # initial condition for state transition matrix
    PHI_0[N**2:N+N**2] = x0;                    # initial condition for trajectory

    
    f = partial(varEqns_uncoupled, par=parameters)  # Use partial in order to pass parameters to function
    soln = solve_ivp(f, TSPAN, list(PHI_0),method='RK45',dense_output=True, events = half_period,rtol=RelTol, atol=AbsTol)
    t = soln.t
    PHI = soln.y
    PHI = PHI.transpose()
    x = PHI[:,N**2:N+N**2]		   # trajectory from time 0 to tf
    phi_tf = np.reshape(PHI[len(t)-1,0:N**2],(N,N)) # state transition matrix, PHI(O,tf)

    
    return t,x,phi_tf,PHI

#%%
def varEqns_uncoupled(t,PHI,par):
#        PHIdot = varEqns_bp(t,PHI) ;
#
# This here is a preliminary state transition, PHI(t,t0),
# matrix equation attempt for a ball rolling on the surface, based on...
#
#        d PHI(t, t0)
#        ------------ =  Df(t) * PHI(t, t0)
#             dt
#
    phi = PHI[0:16]
    phimatrix  = np.reshape(PHI[0:16],(4,4))
    x,y,px,py = PHI[16:20];
    

    # The first order derivative of the Hamiltonian.
    dVdx = -par[3]*x+par[4]*x**3
    dVdy = par[5]*y
    
    # The following is the Jacobian matrix 
    d2Vdx2 = -par[3]+par[4]*3*x**2
            
    d2Vdy2 = par[5]
    
    d2Vdydx = 0
    
    d2Vdxdy = d2Vdydx;    

    Df    = np.array([[  0,     0,    par[0],    0],
              [0,     0,    0,    par[1]],
              [-d2Vdx2,  -d2Vdydx,   0,    0],
              [-d2Vdxdy, -d2Vdy2,    0,    0]])

    
    phidot = np.matmul(Df, phimatrix); # variational equation

    PHIdot        = np.zeros(20);
    PHIdot[0:16]  = np.reshape(phidot,(1,16)); 
    PHIdot[16]    = px*par[0]
    PHIdot[17]    = py*par[1]
    PHIdot[18]    = -dVdx; 
    PHIdot[19]    = -dVdy;
    
    return list(PHIdot)

#%%
def half_period(t,x):
    # Return the turning point where we want to stop the integration                           
    #    
    #                                
    #pxDot = x[0]
    #pyDot = x[1]
    #xDot = x[2]
    #yDot = x[3]
    # In the uncoupled problem we use the 4th equation
    
    #pxDot = x[0]
    #pyDot = x[1]
    #xDot = x[2]
    #yDot = x[3]
    terminal = True
    # The zero can be approached from either direction
    direction = 0; #0: all directions of crossing
    return x[3]

#%%
def uncoupled2dof(t,x, par):
    # Hamilton's equations, used for ploting trajectories later.
    xDot = np.zeros(4)

    dVdx = -par[3]*x[0]+par[4]*(x[0])**3
    dVdy = par[5]*x[1]
    
    xDot[0] = x[2]*par[0]
    xDot[1] = x[3]*par[1]
    xDot[2] = -dVdx 
    xDot[3] = -dVdy
    return list(xDot)



#%%
def configdiff_uncoupled(guess1, guess2,n_turn,par):
    # n_turn is the nth turning point we want to choose as our 'turning point for defining the configration difference'
    # this function calculates the difference of x(or y) coordinates between the guess initial conditions and the ith turning points
    # the result is either difference in x coordintes or difference in y coordinates
    TSPAN = [0,40]
    RelTol = 3.e-10
    AbsTol = 1.e-10 
    f1 = partial(uncoupled2dof, par=par) 
    soln1 = solve_ivp(f1, TSPAN, guess1,method='RK45',dense_output=True, events = half_period,rtol=RelTol, atol=AbsTol)
    te1 = soln1.t_events[0]
    t1 = [0,te1[n_turn]]#[0,te1[1]]
    turn1 = soln1.sol(t1)
    x_turn1 = turn1[0,-1] 
    y_turn1 = turn1[1,-1]
    x_diff1 = guess1[0] - x_turn1
    y_diff1 = guess1[1] - y_turn1
    f2 = partial(uncoupled2dof, par=par) 
    soln2 = solve_ivp(f2, TSPAN, guess2,method='RK45',dense_output=True, events = half_period,rtol=RelTol, atol=AbsTol)
    te2 = soln2.t_events[0]
    t2 = [0,te2[n_turn]]#[0,te2[1]]
    turn2 = soln2.sol(t2)
    x_turn2 = turn2[0,-1] 
    y_turn2 = turn2[1,-1] 
    x_diff2 = guess2[0] - x_turn2
    y_diff2 = guess2[1] - y_turn2
    print("Initial guess1%s, intial guess2%s, x_diff1 is %s, x_diff2 is%s " %(guess1,guess2,x_diff1, x_diff2))
    #print("Initial guess1%s, intial guess2%s, y_diff1 is %s, y_diff2 is%s " %(guess1,guess2,y_diff1, y_diff2))
    return x_diff1, x_diff2


def tpcd_uncoupled(begin1,begin2,par,e,n,n_turn,po_fam_file):
    # n is the number of divisons we want to divide
    # n_turn is the nth turning point we want to choose as our 'turning point for defining the configration difference'
    # e is the energy of the PES 
    # po_fam_file is the file we want to save our data into 
    # we assume x coordinate of guess1 is smaller than the x coordinate of guess2
    #
    #
    # we define the tolerance as the distance(of y coordinate) between the turning point and the point on the PES with the same x coordinate.
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
    x0po = np.zeros((MAXiter ,4))
    i_turn = np.zeros((MAXiter ,1))
    T = np.zeros((MAXiter ,1))
    energyPO = np.zeros((MAXiter ,1))
    iter = 0
    iter_diff =0  # for counting the correct index
    #while toler < 1e-6 or iter < MAXiter:
    while iter < MAXiter and n_turn < 5:
        #y_PES = -get_y(guess1[0], e,par)
        #y_turning = guess1[1]
        #toler = math.sqrt((y_PES-y_turning)**2)
        for i in range(0,n+1):
            # the x difference between guess1 and each guess is recorded in "result" matrix
            h = (guess2[0] - guess1[0])*i/n
            print("h is ",h)
            xguess = guess1[0]+h
            yguess = get_y(xguess, e,par)
            guess = [xguess,yguess,0, 0]
            x_diff1, x_diff2 = configdiff_uncoupled(guess1, guess,n_turn,par)
            result[i,0] = np.sign(x_diff1)
            result[i,1] = guess[0]
            result[i,2] = guess[1]
            result[i,3] = np.sign(x_diff2)
        for i in range(1,n+1):
            if np.sign(result[i,0]) != np.sign(result[i,3]) and np.sign(result[i-1,0]) == np.sign(result[i-1,3]):
                i_turn[iter] = i


        # if the follwing condition holds, we can zoom in to a smaller interval and continue our procedure
        if i_turn[iter]>0:
            index = int(i_turn[iter])
            guesspo  = [result[index-1,1],result[index-1,2],0,0]
            print("Our guess for the periodic orbit is",guesspo)
            x0po[iter,:] = guesspo[:]
            TSPAN = [0,30]
            RelTol = 3.e-10
            AbsTol = 1.e-10
            f = partial(uncoupled2dof, par=par) 
            soln = solve_ivp(f, TSPAN, guesspo,method='RK45',dense_output=True, events = half_period,rtol=RelTol, atol=AbsTol)
            te = soln.t_events[0]
            tt = [0,te[1]]
            t,x,phi_t1,PHI = stateTransitMat_uncoupled(tt,guesspo,par)
            T[iter] = tt[-1]*2
            print("period is%s " %T[iter])
            energy = np.zeros(len(x))
            #print(len(t))
            for j in range(len(t)):
                energy[j] = get_total_energy_uncoupled(x[j,:],par)
            energyPO[iter] = np.mean(energy)
            ax = plt.gca(projection='3d')
            ax.plot(x[:,0],x[:,1],x[:,3],'-')
            ax.plot(x[:,0],x[:,1],-x[:,3],'--')
            ax.scatter(x[0,0],x[0,1],x[0,3],s=20,marker='*');
            ax.scatter(x[-1,0],x[-1,1],x[-1,3],s=20,marker='o');
            ax.set_xlabel('$x$', fontsize=axis_fs)
            ax.set_ylabel('$y$', fontsize=axis_fs)
            ax.set_zlabel('$p_y$', fontsize=axis_fs)
            ax.set_title('$\Delta E$ = %e' %(np.mean(energy) - par[2] ) ,fontsize=axis_fs)
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            #x_turn= x[-1,0]  # x coordinate of turning point
            #y_turn= x[-1,1] # y coordinate of turning point
            #y_PES = -get_y(x_turn,e,par)
            #toler = math.sqrt((y_PES-y_turn)**2)
            plt.grid()
            plt.show()
            guess2 = np.array([result[index,1], result[index,2],0,0])
            guess1 = np.array([result[index-1,1], result[index-1,2],0,0])
            iter_diff =0
        # If the if condition does not hold, it indicates that the interval we picked for performing 'configration difference' is wrong and it needs to be changed.
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
            # we take a larger interval so that it contains the true value of the initial condition and avoids to reach the limitation of the configration difference
            iter_diff = iter_diff +1
            if iter_diff> 1:
                   # return to the iteration that is before the previous 
                print("Warning: the result after this iteration may not be accurate, try to increase the number of intervals or use other ways ")
                break
            n_turn = n_turn+1
            print("nth turningpoint we pick is ", n_turn)
            #index = int((n+1)*(re_iter-1)+i_turn[re_iter-1])
            index = int((n+1)*(iter-iter_diff)+i_turn[iter-iter_diff])
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
        iter = iter +1
        print(iter)

    
    
    dum =np.concatenate((x0po,T, energyPO),axis=1)
    np.savetxt(po_fam_file.name,dum,fmt='%1.16e')
    return x0po, T,energyPO