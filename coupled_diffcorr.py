"""
Created on Wed Jul 17 11:41:17 2019

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


def get_eq_pts_coupled(eqNum, parameters):
    #GET_EQ_PTS_BP solves the saddle center equilibrium point for a system with
    #KE + PE. 
    # H = 1/2*px^2+omega/2*py^2 -(1/2)*alpha*x^2+1/4*beta*x^4+ omega/2y^2+0.5*epsilon*(x-y)**2
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
        eqPt = [+sqrt(parameters[3]-parameters[6]/parameters[4]),0]    # EQNUM = 2, stable
        return eqPt
    elif 	eqNum == 3:
        eqPt = [-math.sqrt(parameters[3]-parameters[6]/parameters[4]),0]   # EQNUM = 3, stable
        return eqPt
    
    
    # F(xEq) = 0 at the equilibrium point, solve using in-built function
 
    def func_vec_field_eq_pt(x,par):
        x1, x2 = x
        # constant parameters for the surface

        dVdx = (-par[3]+par[6])*x1+par[4]*x1**3 -par[6]*x2
        
        dVdy = (par[5]+par[6])*x2-par[6]*x1
    
        F = [-dVdx, -dVdy]
        return F

    eqPt = fsolve(func_vec_field_eq_pt,x0, fprime=None,args=(parameters,)); # Call solver
    return eqPt

#%%
def get_total_energy_coupled(orbit, parameters):

#   get_total_energy_deleonberne computes the total energy of an input orbit
#   (represented as M x N with M time steps and N = 4, dimension of phase
#   space for the model) for the 2 DoF DeLeon-Berne potential.
# 
#   Orbit can be different initial conditions for the periodic orbit of
#   different energy. When trajectory is input, the output energy is mean.
#
    

    x  = orbit[0]
    y  = orbit[1]
    px = orbit[2]
    py = orbit[3]
    

    
    # enter the definition of the potential energy function
    def get_potential_energy(x,y,par):

        pot_energy =  -0.5*par[3]*x**2+0.25*par[4]*x**4 +0.5*par[5]*y**2+0.5*par[6]*(x-y)**2
                
        return pot_energy
    
    
    e = (0.5*parameters[0])*(px**2) + (0.5*parameters[1])*(py**2) +  get_potential_energy(x, y,parameters)   
        
    return e



#%%
def stateTransitMat_coupled(tf,x0,parameters,fixed_step=0): 

    # function [x,t,phi_tf,PHI] =
    # stateTransitionMatrix_boatPR(x0,tf,R,OPTIONS,fixed_step)
    #
    # Gets state transition matrix, phi_tf = PHI(0,tf), and the trajectory 
    # (x,t) for a length of time, tf.  
    # 
    # In particular, for periodic solutions of % period tf=T, one can obtain 
    # the monodromy matrix, PHI(0,T).
    

    N = len(x0);  #N=4
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

    
    f = partial(varEqns_coupled, par=parameters)  # Use partial in order to pass parameters to function
    soln = solve_ivp(f, TSPAN, list(PHI_0),method='RK45',dense_output=True, events = half_period,rtol=RelTol, atol=AbsTol)
    t = soln.t
    PHI = soln.y
    PHI = PHI.transpose()
    x = PHI[:,N**2:N+N**2]		   # trajectory from time 0 to tf
    phi_tf = np.reshape(PHI[len(t)-1,0:N**2],(N,N)) # state transition matrix, PHI(O,tf)

    
    return t,x,phi_tf,PHI


#%%
def varEqns_coupled(t,PHI,par):
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
    dVdx = (-par[3]+par[6])*x+par[4]*x**3-par[6]*y
    dVdy = (par[5]+par[6])*y-par[6]*x

    # The following is the Jacobian matrix 
    d2Vdx2 = -par[3]+par[6]+par[4]*3*x**2
            
    d2Vdy2 = par[5]+par[6]
    
    d2Vdydx = -par[6]
    
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
def coupled2dof(t,x, par):
    # Hamilton's equations, used for ploting trajectories later.


    xDot = np.zeros(4)

    dVdx = (-par[3]+par[6])*x[0]+par[4]*(x[0])**3-par[6]*x[1]
    dVdy = (par[5]+par[6])*x[1]-par[6]*x[0]
    
    xDot[0] = x[2]*par[0]
    xDot[1] = x[3]*par[1]
    xDot[2] = -dVdx 
    xDot[3] = -dVdy
    return list(xDot)



#%%
def jacobian_coupled(eqPt, par):
    # The Jacobian matrix of the system evaluated at the equilibrium point.
    x,y,px,py = eqPt[0:4]
    dVdx = (-par[3]+par[6])*x+par[4]*x**3-par[6]*y
    dVdy = (par[5]+par[6])*y-par[6]*x
    
    # The following is the Jacobian matrix 
    d2Vdx2 = -par[3]+par[6]+par[4]*3*x**2
            
    d2Vdy2 = par[5]+par[6]
    
    d2Vdydx = -par[6]
    
    d2Vdxdy = d2Vdydx;    

    Df    = np.array([[  0,     0,    par[0],    0],
              [0,     0,    0,    par[1]],
              [-d2Vdx2,  -d2Vdydx,   0,    0],
              [-d2Vdxdy, -d2Vdy2,    0,    0]])
    return Df

#%%
def RemoveInfinitesimals(A):

    # Remove all entries with absolute value smaller than TOL
    TOL=1.e-14;
    A = np.array(A,dtype=np.complex_)

    for k in range(0,len(A)):
        if abs(A[k].real) < TOL:
            A[k]=A[k] - A[k].real
        if abs(A[k].imag) < TOL:
            A[k]=A[k] - 1j*A[k].imag
    return list(A)

#%%
def cleanUpMatrix(A):

    #        A = CLEANUPMATRIX(A) ;
    #
    # Remove all entries in matrix A with absolute value smaller than TOL
    # where TOL is set inside cleanUpMatrix.m
    TOL=1.e-14
    A = A + np.zeros((len(A), len(A[0])),dtype=np.complex_)
    for k in range(0,len(A)):
        for l in range(0,len(A)):
            if abs(A[k,l].real) < TOL:
                a_kl = 1j*(A[k,l].imag)
                A[k,l]= a_kl 
            if abs(A[k,l].imag) < TOL:
                a_kl = A[k,l].real 
                A[k,l] = a_kl
    return A

#%%
def eigGet(A,discrete):
    
    

    #   [sn,un,cn,Ws,Wu,Wc] = eigGet(A,discrete) ;
    #
    #   Compute the eigenvalues and eigenvectors of the matrix A spanning the
    #   three local subspaces <Es,Eu,Ec> where A is MxM and s+u+c=M, which
    #   locally approximate the invariant manifolds <Ws,Wu,Wc>
    #
    #   if discrete   time system, discrete = 1
    #   if continuous time system, discrete = 0
    #

    sn=[]
    un=[]
    cn=[]
    Ws=[]
    Wu=[]
    Wc=[]

    # arbitrary small displacement for use in discrete case 
    delta = 1.e-4  # <== maybe needs tweeking?
    M,dum=len(A), len(A[0])
    D,V  =np.linalg.eig(A) # obtain eigenvectors (V) and eigenvalues (D) of matrix A
    D  = np.diagflat(D) + np.zeros((len(A), len(A[0])),dtype=np.complex_)
    V  = V + np.zeros((len(A), len(A[0])),dtype=np.complex_)
    V = cleanUpMatrix(V) 
    D = cleanUpMatrix(D) 
    s=0
    u=0
    c=0
    for k in range(0,M):
        if discrete == 0:	         # continuous time system                 
            if D[k,k].real < 0:
                sn = np.append(sn,D[k,k])                
                jj=0 
                while abs(V[jj,k]) == 0:
                    jj=jj+1 
                Ws = np.append(Ws,V[:,k] /V[jj,k])# stable   (s dimensional)
                Ws = np.reshape(Ws,(s+1,M))
                Ws[:,s] = RemoveInfinitesimals(Ws[:,s])
                s=s+1
            elif D[k,k].real > 0:
                un = np.append(un,D[k,k])
                jj=0
                while abs(V[jj,k]) == 0:
                    jj=jj+1
                Wu = np.append(Wu,V[:,k]/V[jj,k])# unstable (u dimensional)
                Wu = np.reshape(Wu,(u+1,M))
                Wu[:,u] = RemoveInfinitesimals(Wu[:,u])
                u=u+1
                
            else:
                cn = np.append(cn,D[k,k])
                jj=0
                while abs(V[jj,k]) == 0:
                    jj=jj+1
                Wc = np.append(Wc,V[:,k]/V[jj,k])# center   (c dimensional)
                Wc = np.reshape(Wc,(c+1,M))
                Wc[:,c] = RemoveInfinitesimals(Wc[:,c])
                c=c+1
    Ws = np.transpose(Ws)
    Wu = np.transpose(Wu)
    Wc = np.transpose(Wc)
    return sn,un,cn,Ws,Wu,Wc

#%%
def eigvector_coupled(par):
    # eigenvectors and eigenvalues of the Jacobian evaluated at the equilibrium point, which is the correction of the initial condition.
    # check the result obtained from the Jacobian matches the analytic result.
    evaluelamb = math.sqrt(-0.5*(par[3]-par[6]-par[1]*(par[1]+par[6]) -math.sqrt(par[1]**4 + 2*par[1]**3*par[6] + par[1]**2*(par[6]**2+2*par[3]-2*par[6]) +par[1]*( 2*par[6]**2 + 2*par[3]*par[6]) +(par[3]- par[6])**2)))
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
#%%
def get_POGuessLinear_coupled(eqNum,Ax,par):

#   [x0poGuess,TGuess] = get_POGuessLinear_deleonberne(eqNum,Ax) ;

#   par = [MASS_A MASS_B EPSILON_S D_X LAMBDA ALPHA];  
# 
# Uses a small displacement from the equilibrium point (in a direction 
# on the collinear point's center manifold) as a first guess for a planar 
# periodic orbit (called a Lyapunov orbit in th rest. three-body problem).
#
# The initial condition and period are to be used as a first guess for
# a differential correction routine.
#
# Output:
# x0poGuess = initial state on the periodic orbit (first guess)
#           = [ x 0  0 yvel]  , (i.e., perp. to x-axis and in the plane)
# TGuess    = period of periodic orbit (first guess)
#
# Input:
# eqNum = the number of the equilibrium point of interest
# Ax    = nondim. amplitude of periodic orbit (<< 1) 
#

    x0poGuess  = np.zeros(4);
    
    eqPos = get_eq_pts_coupled(eqNum, par);
    eqPt = [eqPos[0], eqPos[1], 0, 0]; # phase space location of equil. point

    # Get the eigenvalues and eigenvectors of Jacobian of ODEs at equil. point
    
    def eqPointEig_coupled(ep, parameters):

        #   [Es,Eu,Ec,Vs,Vu,Vc] = eqPointEig_deleonberne(ep,mu);
        #
        #   Find all the eigenvectors locally spanning the 3 subspaces for the phase
        #   space in an infinitesimal region around an equilibrium point 
        #
        #   Our convention is to give the +y directed column eigenvectors
        #

        Df = jacobian_coupled(ep, parameters);
        Es,Eu,Ec,Vs,Vu,Vc = eigGet(Df,0);	# find the eigenvectors
                                        # i.e., local manifold approximations 

        # give the +y directed column vectors 
        if Vs[1,0]<0:
            Vs=-Vs
        if Vu[1,0]<0:
            Vu=-Vu
        return Es,Eu,Ec,Vs,Vu,Vc

    [Es,Eu,Ec,Vs,Vu,Vc] = eqPointEig_coupled(eqPt, par)

    L = abs(Ec[0].imag)

    correcx, correcy = eigvector_coupled(par)
        
    ### This is where the linearized guess based on center manifold needs
    ### to be entered.
    if eqNum == 1:
        x0poGuess[0]  = eqPt[0]-Ax*correcx
        x0poGuess[1]  = eqPt[1]-Ax*correcy;

    
    ### This is where the linearized guess based on center manifold needs
    ### to be entered.
    
    TGuess = 2*math.pi/L ;
    
    return x0poGuess,TGuess

#%%
def half_period(t,x):

    # The value that we want to be zero
    # In the uncoupled problem we use the 4th equation
    #
    #       t = 0                    t = T/2                             t = T
    #      Initial point --------- Half period  --------------------- Full period
    # (x,y,px,py) = (x0,y0,0,0)---(x,y,px,py) = (x_T/2,y_T/2,0,0)----(x,y,px,py) = (x_T,y_T,0,0)                          
    #  one side of the PES--------the other of the PES---------------returns the original side of the PES                            
    #    
    # 
    #pxDot = x[0]
    #pyDot = x[1]
    #xDot = x[2]
    #yDot = x[3]
    terminal = True
    # The zero can be approached from either direction
    direction = 0; #0: all directions of crossing
    return x[3]

#%%
def get_PODiffCorr_POFam(x0, par):

    # set show = 1 to plot successive approximations (default=0)
    show = 1 
    # axesFontName = 'factory';
    # axesFontName = 'Times New Roman';
    # label_fs = 20; axis_fs = 30; # fontsize for publications 
    label_fs = 10
    axis_fs = 15 # fontsize for publications 
    # set(0,'Defaulttextinterpreter','latex', ...
    #     'DefaultAxesFontName', axesFontName, ...
    #     'DefaultTextFontName', axesFontName, ...
    #     'DefaultAxesFontSize', axis_fs, ...
    #     'DefaultTextFontSize', label_fs, ...
    #     'Defaultuicontrolfontweight','normal', ...
    #     'Defaulttextfontweight','normal', ...
    #     'Defaultaxesfontweight','normal');


    # tolerances for integration and perpendicular crossing of x-axis
    # MAXdxdot1 = 1.e-8 ;RelTol = 3.e-10; AbsTol = 1.e-10; 
    # MAXdxdot1 = 1.e-12 ; RelTol = 3.e-14; AbsTol = 1.e-14; 
    MAXdxdot1 = 1.e-10 
    RelTol = 3.e-14
    AbsTol = 1.e-14

    MAXattempt = 100;     	# maximum number of attempts before error is declared
    # Using dydot or dxdot depends on which variable we want to keep fixed during differential correction.
    # dydot-----x fixed, dxdot------y fixed
    dxdot1 	   = 1;         # to start while loop
    #dydot1 	   = 1;         # to start while loop
    attempt    = 0;         # begin counting number of attempts
    # y0(attempt) = 1;
    correctx0= 0
    #correcty0 = 0
    while abs(dxdot1) > MAXdxdot1:
        if attempt > MAXattempt:
            ERROR= 'Maximum iterations exceeded' 
            print(ERROR) 
            break
        # Find first half-period crossing event
        TSPAN = [0 ,20];        # allow sufficient time for the half-period crossing event                   
    
        f = partial(coupled2dof, par=par)  # Use partial in order to pass parameters to function
        soln1 = solve_ivp(f, TSPAN, x0,method='RK45',dense_output=True, events = half_period,rtol=RelTol, atol=AbsTol)
        te = soln1.t_events[0]
        t1 = [0,te[1]]
        xx1 = soln1.sol(t1)
        x1 = xx1[0,-1] 
        y1 = xx1[1,-1] 
        dxdot1 = xx1[2,-1]
        dydot1  = xx1[3,-1];
    
        # Compute the state transition matrix from the initial state to
	     # the final state at the half-period event crossing
      
        # Events option not necessary anymore
        t,x,phi_t1,PHI = stateTransitMat_coupled(t1,x0,par) ;
        
        
        print('::poDifCor : iteration',attempt+1) ;
        
        if show == 1:
            
            e = np.zeros(len(x))
            for i in range(0,len(x)):
                
                e[i] = get_total_energy_coupled(x[i,:],par);
            ax = plt.gca(projection='3d')
            ax.plot(x[:,0],x[:,1],x[:,3],'-')
            ax.plot(x[:,0],x[:,1],-x[:,3],'--')
            ax.scatter(x[0,0],x[0,1],x[0,3],s=20,marker='*');
            ax.scatter(x[-1,0],x[-1,1],x[-1,3],s=20,marker='o');
            ax.set_xlabel('$x$', fontsize=axis_fs)
            ax.set_ylabel('$y$', fontsize=axis_fs)
            ax.set_zlabel('$v_x$', fontsize=axis_fs)
            ax.set_title('$\Delta E$ = %e' %(np.mean(e) - par[2] ) ,fontsize=axis_fs)
            #par(3) is the energy of the saddle
            ax.set_xlim(-0.1, 0.1)
            #ax.set_ylim(-1, 1)
            #time.sleep(0.01)
            plt.grid()
            plt.show()




        #=========================================================================
        # differential correction and convergence test, adjust according to
        # the particular problem

        #compute acceleration values for correction term
        dVdx = (-par[3]+par[6])*x1+par[4]*x1**3 -par[6]*y1
        dVdy = (par[5]+par[6])*y1-par[6]*x1
        
        vxdot1 = -dVdx
        vydot1 = -dVdy
    
        #correction to the initial x0
        correctx0 = dxdot1/(phi_t1[2,0] - phi_t1[3,0]*(vxdot1/vydot1));	
        x0[0] = x0[0] - correctx0
        
        #correction to the initial y0
        #correcty0 = 1/(phi_t1[3,1] - phi_t1[2,1]*vydot1*(1/vxdot1))*dydot1
        #x0[1] = x0[1] - correcty0
        attempt = attempt+1

    x0po=x0;
    t1 = t1[-1];
    return x0po,t1


#%%
def get_PODiffCorr_coupled(x0, par):

    # set show = 1 to plot successive approximations (default=0)
    show = 1 
    # axesFontName = 'factory';
    # axesFontName = 'Times New Roman';
    # label_fs = 20; axis_fs = 30; # fontsize for publications 
    label_fs = 10
    axis_fs = 15 # fontsize for publications 
    # set(0,'Defaulttextinterpreter','latex', ...
    #     'DefaultAxesFontName', axesFontName, ...
    #     'DefaultTextFontName', axesFontName, ...
    #     'DefaultAxesFontSize', axis_fs, ...
    #     'DefaultTextFontSize', label_fs, ...
    #     'Defaultuicontrolfontweight','normal', ...
    #     'Defaulttextfontweight','normal', ...
    #     'Defaultaxesfontweight','normal');


    # tolerances for integration and perpendicular crossing of x-axis
    # MAXdxdot1 = 1.e-8 ;RelTol = 3.e-10; AbsTol = 1.e-10; 
    # MAXdxdot1 = 1.e-12 ; RelTol = 3.e-14; AbsTol = 1.e-14; 
    MAXdxdot1 = 1.e-10 
    RelTol = 3.e-14
    AbsTol = 1.e-14 

    MAXattempt = 100     	# maximum number of attempts before error is declared
    # Using dydot or dxdot depends on which variable we want to keep fixed during differential correction.
    # dydot-----x fixed, dxdot------y fixed
    
    dxdot1 	   = 1         # to start while loop
    #dydot1 	   = 1        # to start while loop
    attempt    = 0         # begin counting number of attempts
    # y0(attempt) = 1
    correctx0 = 0
    #correcty0=0
    while abs(dxdot1) > MAXdxdot1:
        if attempt > MAXattempt:
            ERROR= 'Maximum iterations exceeded' 
            print(ERROR) 
            break
        # Find first half-period crossing event
        TSPAN = [0 ,20];        # allow sufficient time for the half-period crossing event                 
    
    
        f = partial(coupled2dof, par=par)  # Use partial in order to pass parameters to function
        soln1 = solve_ivp(f, TSPAN, x0,method='RK45',dense_output=True, events = half_period,rtol=RelTol, atol=AbsTol)
        te = soln1.t_events[0]
        t1 = [0,te[1]]
        xx1 = soln1.sol(t1)
        x1 = xx1[0,-1] 
        y1 = xx1[1,-1] 
        dxdot1 = xx1[2,-1]
        dydot1  = xx1[3,-1];
   
    
        # Compute the state transition matrix from the initial state to
	     # the final state at the half-period event crossing
      
        # Events option not necessary anymore
        t,x,phi_t1,PHI = stateTransitMat_coupled(t1,x0,par) ;

        print('::poDifCor : iteration',attempt+1) ;
        
        if show == 1:
            e = np.zeros(len(x))
            for i in range(0,len(x)):
                
                e[i] = get_total_energy_coupled(x[i,:],par); 
            ax = plt.gca(projection='3d')
            plt.plot(x[:,0],x[:,1],x[:,3],'-')
            plt.plot(x[:,0],x[:,1],-x[:,3],'--')
            ax.scatter(x[0,0],x[0,1],x[0,3],s=60,marker='*');
            ax.scatter(x[-1,0],x[-1,1],x[-1,3],s=60,marker='o');
            ax.set_xlabel('$x$', fontsize=axis_fs)
            ax.set_ylabel('$y$', fontsize=axis_fs)
            ax.set_zlabel('$v_x$', fontsize=axis_fs)
            ax.set_title('$\Delta E$ = %e' %(np.mean(e) - par[2])  ,fontsize=axis_fs)
            #par(3) is the energy of the saddle
            ax.set_xlim(-0.1, 0.1)
            #             ax.set_ylim(-1, 1)
            #time.sleep(0.01) ;
            plt.grid()
            plt.show()




        #=========================================================================
        # differential correction and convergence test, adjust according to
        # the particular problem

        #compute acceleration values for correction term
        dVdx = (-par[3]+par[6])*x1+par[4]*x1**3 -par[6]*y1
        dVdy = (par[5]+par[6])*y1-par[6]*x1
        
        vxdot1 = -dVdx;
        vydot1 = -dVdy;
    
        #correction to the initial x0
        correctx0 = dxdot1/(phi_t1[2,0] - phi_t1[3,0]*(vxdot1/vydot1));	
        x0[0] = x0[0] - correctx0;
        
        #correction to the initial y0
        #correcty0 = 1/(phi_t1[3,1] - phi_t1[2,1]*vydot1*(1/vxdot1))*dydot1;
        #x0[1] = x0[1] - correcty0;
        attempt = attempt+1 ;
        
    x0po=x0;
    t1 = t1[-1];
    return x0po,t1

#%%
def get_POFam_coupled(eqNum,Ax1,Ax2,nFam,po_fam_file,par):

    # [x0po,T] = get_POFam_deleonberne(eqNum,Ax1,Ax2,nFam,po_fam_file);
    #
    # Generate a family of periodic orbits (po) given a pair of 
    # seed initial conditions and periods
    #
    
    # delt = guessed change in period between successive orbits in family

    delt = -1.e-9    # <==== may need to be changed
    #delt = -1.e-12 ;    
    N = 4 # dimension of phase space
    x0po = np.zeros((nFam,N)) ;
    T    = np.zeros((nFam,1)) ;
    energyPO = np.zeros((nFam,1)) ;
    
    x0poGuess1,TGuess1 = get_POGuessLinear_coupled(eqNum,Ax1,par);
    x0poGuess2,TGuess2 = get_POGuessLinear_coupled(eqNum,Ax2,par);
    
    # Get the first two periodic orbit initial conditions
    iFam = 0 ;
    print('::poFamGet : number',iFam) ;
    x0po1,tfpo1 = get_PODiffCorr_POFam(x0poGuess1, par) ; 

    energyPO[iFam] = get_total_energy_coupled(x0po1, par) ;


    iFam = 1 ;
    print('::poFamGet : number',iFam) ;
    
    x0po2,tfpo2 = get_PODiffCorr_POFam(x0poGuess2, par) ;               

    energyPO[iFam] = get_total_energy_coupled(x0po2, par) ;

    x0po[0,:] =  x0po1[:]
    x0po[1,:] =  x0po2[:]
    T[0]      = 2*tfpo1 
    T[1]      = 2*tfpo2
    #Generate the other members of the family using numerical continuation
    for i in range (2,nFam):

        print('::poFamGet : number', i) ;
        
        dx  = x0po[i-1,0] - x0po[i-2,0] ;
        dy  = x0po[i-1,1] - x0po[i-2,1] ;
#         dOrbit = x0po[iFam-1,:] - x0po[iFam-2,:];
        dt  = T[i-1] - T[i-2] ;

#         x0po_g = x0po[iFam-1,:]' + dOrbit';
        t1po_g =   (T[i-1] + dt)/2 + delt ;
        x0po_g = [ x0po[i-1,0] + dx, x0po[i-1,1] + dy, 0, 0] ;
#         t1po_g = T[iFam-2] + abs(dt);

      # differential correction takes place in the following function
        x0po_i,tfpo_i = get_PODiffCorr_POFam(x0po_g, par) ; 


        x0po[i,:] = x0po_i;
#         T   [iFam]     = 2*t1_iFam ;	
        T[i]        = 2*tfpo_i;
        
            
        energyPO[i] = get_total_energy_coupled(x0po[i,:], par) ;
        
        if iFam+1 % 10 == 0:
            
            dum = dum[:,0:N+1] ;
    dum =np.concatenate((x0po,T, energyPO),axis=1)
    np.savetxt(po_fam_file.name,dum,fmt='%1.16e')
    return x0po,T

#%%
def poBracketEnergy_coupled(energyTarget,x0podata, po_brac_file, par):

    # POBRACKETENERGY_SHIPRP Generates a family of periodic orbits (po)
    # given a pair of seed initial conditions from a data file, while targeting
    # a specific periodic orbit. This is performed using a scaling factor of
    # the numerical continuation step size which is used to obtain initial
    # guess for the periodic orbit. The energy of target periodic orbit should
    # be higher than the input periodic orbit.
    # 
    #   [X0PO,T] = POBRACKETENERGY_SHIPRP(ENERGYTARGET, X0PODATA)
    #   computes a family of periodic orbit (X0PO and T) while targeting
    #   energy, ENERGYTARGET from the starting initial condition of periodic
    #   orbit, X0PODATA
    #


    #delt = guessed change in period between successive orbits in family
    #delt = -1.e-9    # <==== may need to be changed
    #delt = -1.e-12
    #energyTol = 1e-10; # decrease tolerance for higher target energy
    energyTol = 1e-6
    N = 4  # dimension of phase space
    
    x0po = np.zeros((200,N))
    T  = np.zeros((200,1))
    energyPO = np.zeros((200,1))
    x0po[0,:]   = x0podata[-2,0:N];
    x0po[1,:]   = x0podata[-1,0:N];
    T[0]        = x0podata[-2,N];
    T[1]        = x0podata[-1,N];
    energyPO[0] = get_total_energy_coupled(x0po[0,0:N],par)
    energyPO[1] = get_total_energy_coupled(x0po[1,0:N],par)
    
    iFam = 2;
    scaleFactor = 1.25   #scaling the change in initial guess, usually in [1,2]
    finished = 1
    
    while finished == 1 or iFam < 200:
        
#         FAMNUM = sprintf('::poFamGet : number %d',iFam) ;
#         disp(FAMNUM) ;
        
        #change in initial guess for next step
        dx  = x0po[iFam-1,0] - x0po[iFam-2,0] ;
        dy  = x0po[iFam-1,1] - x0po[iFam-2,1] ;

        #check p.o. energy and set the initial guess with scale factor
        if energyPO[iFam-1] < energyTarget:
            scaleFactor = scaleFactor;
            x0po_g = [ x0po[iFam-1,0] + scaleFactor*dx, x0po[iFam-1,1] + scaleFactor*dy, 0, 0] ;
    #             x0po_g = [ (x0po(iFam-1,1) + scaleFactor*dx), ...
    #                 (x0po(iFam-1,2)), 0, 0] ;

            [x0po_iFam,tfpo_iFam] = get_PODiffCorr_coupled(x0po_g, par) ; 
            energyPO[iFam] = get_total_energy_coupled(x0po_iFam, par);
            x0po[iFam,:] = x0po_iFam ;
            T[iFam]      = 2*tfpo_iFam ;
            iFam = iFam + 1;
            
            #to improve speed of stepping, increase scale factor for very
            #close p.o., this is when target p.o. hasn't been crossed,
            if abs(dx) < 1e-4 and abs(dy) < 1e-4:
                scaleFactor = 1.5;
            else:
                scaleFactor = scaleFactor;
            
        elif energyPO[iFam-1] > energyTarget:
            break
            scaleFactor = scaleFactor*1e-2;
            x0po_g = [ x0po[iFam-2,0] + scaleFactor*dx,x0po[iFam-2,1] + scaleFactor*dy, 0, 0] ;
    #             x0po_g = [ (x0po(iFam-2,1) + scaleFactor*dx) ...
    #                 (x0po(iFam-2,2)) 0 0] ;

            [x0po_iFam,tfpo_iFam] = get_PODiffCorr_coupled(x0po_g, par) ; 
            energyPO[iFam-1] = get_total_energy_coupled(x0po_iFam, par);
            x0po[iFam-1,:] = x0po_iFam ;
            T[iFam-1]      = 2*tfpo_iFam ;

        
        if abs(energyTarget - energyPO[iFam-1]) > energyTol*energyPO[iFam-1]:
            finished = 1;
        else:
            finished = 0;
            

    POENERGYERR = ('Relative error in the po energy from target',abs(energyTarget - energyPO[iFam-1])/energyPO[iFam-1]) ;
    print(POENERGYERR) ;
    dum =np.concatenate((x0po,T, energyPO),axis=1)
    for i in range(1,200):
        if T[i] == 0 and T[i-1] !=0 :
            Tend = i  #index of last row of the data, each row below this index is identitically zero.
    dum1 = dum[0:Tend,:]
    np.savetxt(po_brac_file.name, dum1 ,fmt='%1.16e')
    return x0po[0:Tend,:],T[0:Tend]

#%%
def poTargetEnergy_coupled(x0po, energyTarget, po_target_file,par):
    #
    # poTargetEnergy_coupled computes the periodic orbit of target energy using
    # bisection method. Using bisection method on the lower and higher energy
    # values of the POs to find the PO with the target energy. Use this
    # condition to integrate with event function of half-period defined by
    # maximum distance from the initial point on the PO
    # 
    # INPUT
    # x0po:     Initial conditions for the periodic orbit with the last two
    # initial conditions bracketing (lower and higher than) the target energy
    # par: parameters
    # 
    # OUTPUTS
    # x0_PO:    Initial condition of the periodic orbit (P.O.)
    # T_PO:     Time period 
    # ePO:     Energy of the computed P.O.
    # 
    #     label_fs = 10; axis_fs = 15; # small fontsize
    label_fs = 20
    axis_fs = 30 # fontsize for publications 
    
    iFam = len(x0po[0]);

    energyTol = 1e-10
    
    tpTol = 1e-6
    show = 1   # for plotting the final PO
    
    # bisection method begins here
    iter = 0;
    iterMax = 200;
    a = x0po[-2,:]
    b = x0po[-1,:]
    
    print('Bisection method begins \n');
    while iter < iterMax:
    
        
    #         dx = 0.5*(b(1) - a(1));
    #         dy = 0.5*(b(2) - a(2));
        
    #         c = [a(1) + dx a(2) + dy 0 0];
        c = 0.5*(a + b) # guess based on midpoint
        [x0po_iFam,tfpo_iFam] = get_PODiffCorr_coupled(c, par)
        
        energyPO = get_total_energy_coupled(x0po_iFam, par)
    #         x0po(iFam,1:N) = x0po_iFam ;
    #         T(iFam,1)      = 2*tfpo_iFam ;
        
        c = x0po_iFam;
        iter = iter + 1;
        
        if (abs(get_total_energy_coupled(c, par) - energyTarget) < energyTol) or (iter == iterMax):
            print('Initial condition: %s\n' %c);
            print('Energy of the initial condition for PO %s\n' %get_total_energy_coupled(c, par) );
            x0_PO = c
            T_PO = 2*tfpo_iFam; 
            ePO = get_total_energy_coupled(c, par);
            break
        
        
        if np.sign( get_total_energy_coupled(c, par) - energyTarget ) == np.sign ( get_total_energy_coupled(a, par) - energyTarget ):
            a = c
        else:
            b = c;
        print('Iteration number %s, energy of PO: %s\n' %(iter, energyPO)) ;
        

    print('Iterations completed: %s, error in energy: %s \n' %(iter, abs(get_total_energy_coupled(c, par) - energyTarget)));
    
    
    dum =np.concatenate((c, 2*tfpo_iFam, energyPO),axis=None)
    np.savetxt(po_target_file.name, dum ,fmt='%1.16e')
    return x0_PO, T_PO, ePO