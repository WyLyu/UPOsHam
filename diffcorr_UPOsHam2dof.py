# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 18:32:28 2019

@author: Wenyang Lyu and Shibabrat Naik
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'


#%
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
    
    init_guess_eqpt_model : str
        function name that returns the initial guess for the equilibrium point
    
    grad_pot_model : str
        function name that defines the vector of potential gradient
    
    par : float (list)
        model parameters 
   
    Returns
    -------
    float (list)
        configuration space coordinates
        
    """
    # Fix the equilibrium point numbering convention here and make a starting guess at the solution
    x0 = init_guess_eqpt_model(eqNum, par)
    
    # F(xEq) = 0 at the equilibrium point, solve using in-built function
    # F = lambda x: func_vec_field_eq_pt(model,x,par)
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
    
    pot_energy_model : str
        function name that returns the potential energy of Hamiltonian
    
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
    
    get_total_energy(orbit, pot_energy_model, parameters) returns the total energy for a 
    Hamiltonian of the form kinetic energy (KE) + potential energy (PE). 
    
    Parameters
    ----------
    xVec, yVec : 1d numpy arrays
        x,y-coordinates that discretizes the x, y domain of the configuration space
    
    pot_energy_model : str
        function name that returns the potential energy of Hamiltonian
    
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

    
    
#%%
def stateTransitMat(tf,x0,par,varEqns_model,fixed_step=0): 
    """
    function [x,t,phi_tf,PHI] =
    stateTransitionMatrix_boatPR(x0,tf,R,OPTIONS,fixed_step)
    
    Gets state transition matrix, phi_tf = PHI(0,tf), and the trajectory 
    (x,t) for a length of time, tf.  
    
    In particular, for periodic solutions of % period tf=T, one can obtain 
    the monodromy matrix, PHI(0,T).
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
    PHI_0[0:N**2] = np.reshape(np.identity(N),(N**2)) # initial condition for state transition matrix
    PHI_0[N**2:N+N**2] = x0                    # initial condition for trajectory

    
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
def RemoveInfinitesimals(A):
    """
    Remove all entries with absolute value smaller than TOL
    """
    
    TOL=1.e-14
    A = np.array(A,dtype=np.complex_)

    for k in range(0,len(A)):
        if abs(A[k].real) < TOL:
            A[k]=A[k] - A[k].real
        if abs(A[k].imag) < TOL:
            A[k]=A[k] - 1j*A[k].imag
    return list(A)


#%%
def cleanUpMatrix(A):
    """
    A = CLEANUPMATRIX(A) 
    Remove all entries in matrix A with absolute value smaller than TOL
    where TOL is set inside cleanUpMatrix.m
    """
    
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
    
    """
    [sn,un,cn,Ws,Wu,Wc] = eigGet(A,discrete) 
    
    Compute the eigenvalues and eigenvectors of the matrix A spanning the
    three local subspaces <Es,Eu,Ec> where A is MxM and s+u+c=M, which
    locally approximate the invariant manifolds <Ws,Wu,Wc>
    
    This function is designed for continuous dynamical systems only.
    """
    
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
def get_POGuessLinear(eqNum, Ax, init_guess_eqpt_model, grad_pot_model, jacobian_model, \
                      guess_lin_model, par):
    """
    Uses a small displacement from the equilibrium point (in a direction 
    on the collinear point's center manifold) as a first guess for a planar 
    periodic orbit (called a Lyapunov orbit in th rest. three-body problem).

    The initial condition and period are to be used as a first guess for
    a differential correction routine.

    Output:
    x0poGuess = initial state on the periodic orbit (first guess)
           = [ x 0  0 yvel]  , (i.e., perp. to x-axis and in the plane)
    TGuess    = period of periodic orbit (first guess)

    Input:
    eqNum = the number of the equilibrium point of interest
    Ax    = nondim. amplitude of periodic orbit (<< 1) 
    """

    x0poGuess  = np.zeros(4)
    
    eqPos = get_eq_pts(eqNum, init_guess_eqpt_model, grad_pot_model, par)
    eqPt = [eqPos[0], eqPos[1], 0, 0] # phase space location of equil. point

    # Get the eigenvalues and eigenvectors of Jacobian of ODEs at equil. point
    
    def eqPointEig(eqPt, parameters):
        """
        eqPointEig(ep, parameters, model)
        
        Find all the eigenvectors locally spanning the 3 subspaces for the phase
        space in an infinitesimal region around an equilibrium point 
    
        Our convention is to give the +y directed column eigenvectors
        """
        

        Df = jacobian_model(eqPt, parameters)
        Es,Eu,Ec,Vs,Vu,Vc = eigGet(Df,0)	# find the eigenvectors

        # give the +y directed column vectors 
        if Vs[1,0]<0:
            Vs = -Vs
        if Vu[1,0]<0:
            Vu = -Vu
        return Es,Eu,Ec,Vs,Vu,Vc

    [Es,Eu,Ec,Vs,Vu,Vc] = eqPointEig(eqPt, par)

    L = abs(Ec[0].imag)
    # This is where the linearized guess based on center manifold needs
    # to be entered.
    x0poGuess = guess_lin_model(eqPt, Ax, par)
#    print(np.shape(x0poGuess))
    
#    if eqNum == 1:
#        if model =='uncoupled':
#            x0poGuess[0]  = eqPt[0]+Ax
#            x0poGuess[1]  = eqPt[1]+Ax
#        elif model =='coupled':
#            correcx, correcy = eigvector_coupled(par)
#            x0poGuess[0]  = eqPt[0]-Ax*correcx
#            x0poGuess[1]  = eqPt[1]-Ax*correcy
#        elif model == 'deleonberne':
#            x0poGuess[0]  = eqPt[0] - Ax
#            x0poGuess[1]  = eqPt[1]
#        else:
#            print("The model you are chosen does not exist")
    
    TGuess = 2*math.pi/L 
    
    
    return x0poGuess,TGuess


#%%
def get_PODiffCorr(x0, diffcorr_setup_model, conv_coord_model, diffcorr_acc_corr_model, \
                   ham2dof_model, half_period_model, pot_energy_model, varEqns_model, \
                   plot_iter_orbit_model, par):

    # set show = 1 to plot successive approximations (default=0)
    show = 1 
    # axesFontName = 'factory'
    # axesFontName = 'Times New Roman'
    # label_fs = 20; axis_fs = 30; # fontsize for publications 
    


    # tolerances for integration and perpendicular crossing of x-axis
    # MAXdxdot1 = 1.e-8; RelTol = 3.e-10; AbsTol = 1.e-10; 
    # MAXdxdot1 = 1.e-12 ; RelTol = 3.e-14; AbsTol = 1.e-14; 
    RelTol = 3.e-14
    AbsTol = 1.e-14

    MAXattempt = 100     	# maximum number of attempts before error is declared
    
    # Using dydot or dxdot depends on which variable we want to keep fixed during differential correction.
    # dydot-----x fixed, dxdot------y fixed
    drdot1, correctr0, MAXdrdot1 = diffcorr_setup_model()
    
#    if model =='uncoupled':
#       dxdot1 = 1
#       correctx0= 0
#       MAXdxdot1 = 1.e-10 
#       drdot1 = dxdot1
#       correctr0 = correctx0
#       MAXdrdot1 = MAXdxdot1
#    elif model =='coupled':
#        dxdot1 = 1
#        correctx0= 0
#        MAXdxdot1 = 1.e-10
#        drdot1 = dxdot1
#        correctr0 = correctx0
#        MAXdrdot1 = MAXdxdot1
#    elif model == 'deleonberne':
#        dydot1 = 1
#        correcty0= 0
#        MAXdydot1 = 1.e-10
#        drdot1 = dydot1
#        correctr0 = correcty0
#        MAXdrdot1 = MAXdydot1
#    else:
#        print("The model you are chosen does not exist")
    
    attempt = 0
    while abs(drdot1) > MAXdrdot1:
        
        if attempt > MAXattempt:
            print('Maximum iterations exceeded') 
            break
        
        # Find first half-period crossing event
        TSPAN = [0,20]        # allow sufficient time for the half-period crossing event                   
    
        f = lambda t,x: ham2dof_model(t,x,par) # Use partial in order to pass parameters to function
        soln1 = solve_ivp(f, TSPAN, x0,method='RK45',dense_output=True, \
                          events = lambda t,x: half_period_model(t,x,par), rtol=RelTol, atol=AbsTol)
        
        te = soln1.t_events[0]
        t1 = [0,te[1]]
        xx1 = soln1.sol(t1)
        x1 = xx1[0,-1] 
        y1 = xx1[1,-1] 
        dxdot1 = xx1[2,-1]
        dydot1  = xx1[3,-1]
        
        drdot1 = conv_coord_model(x1, y1, dxdot1, dydot1)
        
#        if model =='uncoupled': 
#           drdot1 = dxdot1
#        elif model =='coupled':
#            drdot1 = dxdot1
#        elif model == 'deleonberne':
#            drdot1 = dydot1
#        else:
#            print("The model you are chosen does not exist")
    
        # Compute the state transition matrix from the initial state to the final state at the 
        # half-period event crossing
      
        # Events option not necessary anymore
        t,x,phi_t1,PHI = stateTransitMat(t1,x0,par,varEqns_model) 
        
        
        print('::poDifCor : iteration',attempt+1) 
        
        if show == 1:
            
            e = np.zeros(len(x))
            for i in range(0,len(x)):
                e[i] = get_total_energy(x[i,:], pot_energy_model, par)
                
            ax = plt.gca(projection='3d')
            
            plot_iter_orbit_model(x, ax, e, par)
            
#            if model== 'uncoupled':
#                ax.plot(x[:,0],x[:,1],x[:,3],'-')
#                ax.plot(x[:,0],x[:,1],-x[:,3],'--')
#                ax.scatter(x[0,0],x[0,1],x[0,3],s=20,marker='*')
#                ax.scatter(x[-1,0],x[-1,1],x[-1,3],s=20,marker='o')
#                ax.set_xlabel(r'$x$', fontsize=axis_fs)
#                ax.set_ylabel(r'$y$', fontsize=axis_fs)
#                ax.set_zlabel(r'$p_y$', fontsize=axis_fs)
#            elif model== 'coupled':
#                ax.plot(x[:,0],x[:,1],x[:,3],'-')
#                ax.plot(x[:,0],x[:,1],-x[:,3],'--')
#                ax.scatter(x[0,0],x[0,1],x[0,3],s=20,marker='*')
#                ax.scatter(x[-1,0],x[-1,1],x[-1,3],s=20,marker='o')
#                ax.set_xlabel(r'$x$', fontsize=axis_fs)
#                ax.set_ylabel(r'$y$', fontsize=axis_fs)
#                ax.set_zlabel(r'$p_y$', fontsize=axis_fs)
#                ax.set_title(r'$\Delta E$ = %e' %(np.mean(e) - par[2] ) ,fontsize=axis_fs)
#                #par(3) is the energy of the saddle
#                ax.set_xlim(-0.1, 0.1)
#            elif model == 'deleonberne':
#                ax.plot(x[:,0],x[:,1],x[:,2],'-')
#                ax.plot(x[:,0],x[:,1],-x[:,2],'--')
#                ax.scatter(x[0,0],x[0,1],x[0,2],s=20,marker='*')
#                ax.scatter(x[-1,0],x[-1,1],x[-1,2],s=20,marker='o')
#                ax.set_xlabel(r'$x$', fontsize=axis_fs)
#                ax.set_ylabel(r'$y$', fontsize=axis_fs)
#                ax.set_zlabel(r'$p_x$', fontsize=axis_fs)
#            else:
#                print("The model you are chosen does not exist ")
            
            #ax.set_title(r'$\Delta E$ = %e' %(np.mean(e) - par[2] ) ,fontsize=axis_fs)
            #par(3) is the energy of the saddle
            #ax.set_xlim(-0.1, 0.1)
            #ax.set_ylim(-1, 1)
            #time.sleep(0.01)
            
            plt.grid()
            plt.show()




        #=========================================================================
        # differential correction and convergence test, adjust according to
        # the particular problem

        x0 = diffcorr_acc_corr_model(xx1[:,-1], phi_t1, x0, par)
        
        #compute acceleration values for correction term
#        if model == 'uncoupled':
#            dVdx = -par[3]*x1+par[4]*(x1)**3
#            dVdy = par[5]*y1
#            vxdot1 = -dVdx
#            vydot1 = -dVdy
#            #correction to the initial x0
#            correctx0 = dxdot1/(phi_t1[2,0] - phi_t1[3,0]*(vxdot1/vydot1))
#            x0[0] = x0[0] - correctx0
#        elif model == 'coupled':
#            dVdx = (-par[3]+par[6])*x1+par[4]*(x1)**3-par[6]*y1
#            dVdy = (par[5]+par[6])*y1-par[6]*x1
#            vxdot1 = -dVdx
#            vydot1 = -dVdy
#            #correction to the initial x0
#            correctx0 = dxdot1/(phi_t1[2,0] - phi_t1[3,0]*(vxdot1/vydot1))	
#            x0[0] = x0[0] - correctx0
#        elif model == 'deleonberne':
#            dVdx = -2*par[3]*par[4]*math.e**(-par[4]*x1)*(math.e**(-par[4]*x1) - 1) - 4*par[5]*par[4]*y1**2*(y1**2 - 1)*math.e**(-par[5]*par[4]*x1)
#            dVdy = 8*y1*(2*y1**2 - 1)*math.e**(-par[5]*par[4]*x1)
#            vxdot1 = -dVdx
#            vydot1 = -dVdy
#            correcty0 = 1/(phi_t1[3,1] - phi_t1[2,1]*vydot1*(1/vxdot1))*dydot1
#            x0[1] = x0[1] - correcty0
#        else:
#            print("The model you are chosen does not exist")

    attempt = attempt+1

    x0po=x0
    t1 = t1[-1]
    
    return x0po,t1


#%%
def get_POFam(eqNum,Ax1,Ax2,nFam,po_fam_file,init_guess_eqpt_model, grad_pot_model, \
              jacobian_model, guess_lin_model, diffcorr_setup_model, conv_coord_model, \
              diffcorr_acc_corr_model, ham2dof_model, half_period_model, pot_energy_model, \
              varEqns_model, plot_iter_orbit_model, par):
    """
    get_POFam(eqNum,Ax1,Ax2,nFam,po_fam_file,init_guess_eqpt_model, grad_pot_model, \
              jacobian_model, guess_lin_model, par)
    
    Generate a family of periodic orbits (po) given a pair of seed initial conditions and periods
    
    """
    
    #guessed change in period between successive orbits in family
    delt = -1.e-9    # <==== may need to be changed
    #delt = -1.e-12     
    
    N = 4 # dimension of phase space
    x0po = np.zeros((nFam,N))
    T    = np.zeros((nFam,1))
    energyPO = np.zeros((nFam,1))
    
    x0poGuess1,TGuess1 = get_POGuessLinear(eqNum, Ax1, init_guess_eqpt_model, grad_pot_model, \
                                           jacobian_model, guess_lin_model, par)  
    x0poGuess2,TGuess2 = get_POGuessLinear(eqNum, Ax2, init_guess_eqpt_model, grad_pot_model, \
                                           jacobian_model, guess_lin_model, par)
    
    # Get the first two periodic orbit initial conditions
    iFam = 0 
    print('::poFamGet : number',iFam)
    x0po1,tfpo1 = get_PODiffCorr(x0poGuess1, diffcorr_setup_model, conv_coord_model, \
                                 diffcorr_acc_corr_model, ham2dof_model, \
                                 half_period_model, pot_energy_model, \
                                 varEqns_model, plot_iter_orbit_model, par)

    energyPO[iFam] = get_total_energy(x0po1, pot_energy_model, par)


    iFam = 1
    print('::poFamGet : number',iFam)
    
    x0po2,tfpo2 = get_PODiffCorr(x0poGuess2, diffcorr_setup_model, conv_coord_model, \
                                 diffcorr_acc_corr_model, ham2dof_model, \
                                 half_period_model, pot_energy_model, \
                                 varEqns_model, plot_iter_orbit_model, par)             

    energyPO[iFam] = get_total_energy(x0po2, pot_energy_model, par)

    x0po[0,:] =  x0po1[:]
    x0po[1,:] =  x0po2[:]
    T[0]      = 2*tfpo1 
    T[1]      = 2*tfpo2
    #Generate the other members of the family using numerical continuation
    for i in range (2,nFam):

        print('::poFamGet : number', i)
    
        dx  = x0po[i-1,0] - x0po[i-2,0] 
        dy  = x0po[i-1,1] - x0po[i-2,1] 
#         dOrbit = x0po[iFam-1,:] - x0po[iFam-2,:]
        dt  = T[i-1] - T[i-2]

#         x0po_g = x0po[iFam-1,:]' + dOrbit'
        t1po_g =   (T[i-1] + dt)/2 + delt 
        x0po_g = [ x0po[i-1,0] + dx, x0po[i-1,1] + dy, 0, 0]
#         t1po_g = T[iFam-2] + abs(dt)

      # differential correction takes place in the following function
        x0po_i,tfpo_i = get_PODiffCorr(x0po_g, diffcorr_setup_model, conv_coord_model, \
                                       diffcorr_acc_corr_model, ham2dof_model, \
                                       half_period_model, pot_energy_model, \
                                       varEqns_model, plot_iter_orbit_model, par)


        x0po[i,:] = x0po_i
#         T   [iFam]     = 2*t1_iFam 	
        T[i]        = 2*tfpo_i
        
            
        energyPO[i] = get_total_energy(x0po[i,:], pot_energy_model, par)
        
        if iFam+1 % 10 == 0:
            dum = dum[:,0:N+1]
            
    dum = np.concatenate((x0po,T, energyPO),axis=1)
    np.savetxt(po_fam_file.name,dum,fmt='%1.16e')
    
    
    return x0po,T


#%%
def poBracketEnergy(energyTarget, x0podata, po_brac_file, diffcorr_setup_model, \
                    conv_coord_model, diffcorr_acc_corr_model, ham2dof_model, \
                    half_period_model, pot_energy_model, varEqns_model, \
                    plot_iter_orbit_model, par):
    """
    POBRACKETENERGY Generates a family of periodic orbits (po)
    given a pair of seed initial conditions from a data file, while targeting
    a specific periodic orbit. This is performed using a scaling factor of
    the numerical continuation step size which is used to obtain initial
    guess for the periodic orbit. The energy of target periodic orbit should
    be higher than the input periodic orbit.
     
    [X0PO,T] = POBRACKETENERGY_SHIPRP(ENERGYTARGET, X0PODATA)
    computes a family of periodic orbit (X0PO and T) while targeting
    energy, ENERGYTARGET from the starting initial condition of periodic
    orbit, X0PODATA
    """
    
    #delt = guessed change in period between successive orbits in family
    #delt = -1.e-9    # <==== may need to be changed
    #delt = -1.e-12
    #energyTol = 1e-10 # decrease tolerance for higher target energy
    energyTol = 1e-6
    N = 4  # dimension of phase space
    
    x0po = np.zeros((200,N))
    T  = np.zeros((200,1))
    energyPO = np.zeros((200,1))
    x0po[0,:]   = x0podata[-2,0:N]
    x0po[1,:]   = x0podata[-1,0:N]
    T[0]        = x0podata[-2,N]
    T[1]        = x0podata[-1,N]
    energyPO[0] = get_total_energy(x0po[0,0:N], pot_energy_model, par) 
    energyPO[1] = get_total_energy(x0po[1,0:N], pot_energy_model, par)
    
    iFam = 2
    scaleFactor = 1.25   #scaling the change in initial guess, usually in [1,2]
    finished = 1
    
    while finished == 1 or iFam < 200:
        
#         FAMNUM = sprintf('::poFamGet : number %d',iFam) 
#         disp(FAMNUM) 
        
        #change in initial guess for next step
        dx  = x0po[iFam-1,0] - x0po[iFam-2,0]
        dy  = x0po[iFam-1,1] - x0po[iFam-2,1]

        #check p.o. energy and set the initial guess with scale factor
        if energyPO[iFam-1] < energyTarget:
            scaleFactor = scaleFactor
            x0po_g = [ x0po[iFam-1,0] + scaleFactor*dx, x0po[iFam-1,1] + scaleFactor*dy, 0, 0] 
    #             x0po_g = [ (x0po(iFam-1,1) + scaleFactor*dx), ...
    #                 (x0po(iFam-1,2)), 0, 0]

            [x0po_iFam,tfpo_iFam] = get_PODiffCorr(x0po_g, diffcorr_setup_model, \
                                                    conv_coord_model, \
                                                    diffcorr_acc_corr_model, ham2dof_model, \
                                                    half_period_model, pot_energy_model, \
                                                    varEqns_model, plot_iter_orbit_model, \
                                                    par)
            
            energyPO[iFam] = get_total_energy(x0po_iFam, pot_energy_model, par)
            x0po[iFam,:] = x0po_iFam 
            T[iFam]      = 2*tfpo_iFam 
            iFam = iFam + 1
            
            #to improve speed of stepping, increase scale factor for very
            #close p.o., this is when target p.o. hasn't been crossed,
            if abs(dx) < 1e-4 and abs(dy) < 1e-4:
                scaleFactor = 2
            else:
                scaleFactor = scaleFactor
            
        elif energyPO[iFam-1] > energyTarget:
            break
            scaleFactor = scaleFactor*1e-2
            x0po_g = [ x0po[iFam-2,0] + scaleFactor*dx, x0po[iFam-2,1] + scaleFactor*dy, 0, 0]
    #             x0po_g = [ (x0po(iFam-2,1) + scaleFactor*dx) ...
    #                 (x0po(iFam-2,2)) 0 0]

            [x0po_iFam,tfpo_iFam] = get_PODiffCorr(x0po_g, diffcorr_setup_model, \
                                                    conv_coord_model, \
                                                    diffcorr_acc_corr_model, ham2dof_model, \
                                                    half_period_model, pot_energy_model, \
                                                    varEqns_model, plot_iter_orbit_model, \
                                                    par)
            
            energyPO[iFam-1] = get_total_energy(x0po_iFam, pot_energy_model, par)
            x0po[iFam-1,:] = x0po_iFam
            T[iFam-1]      = 2*tfpo_iFam 

        
        if abs(energyTarget - energyPO[iFam-1]) > energyTol*energyPO[iFam-1]:
            finished = 1
        else:
            finished = 0
            

    print('Relative error in the po energy from target ',abs(energyTarget - energyPO[iFam-1])/energyPO[iFam-1])
    dum = np.concatenate((x0po,T, energyPO),axis=1)
    for i in range(1,200):
        if T[i] == 0 and T[i-1] !=0 :
            Tend = i  #index of last row of the data, each row below this index is identitically zero.
    dum1 = dum[0:Tend,:]
    np.savetxt(po_brac_file.name, dum1 ,fmt='%1.16e')
    
    
    return x0po[0:Tend,:],T[0:Tend]


#%%
def poTargetEnergy(x0po, energyTarget, po_target_file, diffcorr_setup_model, conv_coord_model, \
                   diffcorr_acc_corr_model, ham2dof_model, half_period_model, pot_energy_model, \
                   varEqns_model, plot_iter_orbit_model, par):
    """
    poTargetEnergy computes the periodic orbit of target energy using
    bisection method. Using bisection method on the lower and higher energy
    values of the POs to find the PO with the target energy. Use this
    condition to integrate with event function of half-period defined by
    maximum distance from the initial point on the PO
     
    INPUT
    x0po:     Initial conditions for the periodic orbit with the last two
    initial conditions bracketing (lower and higher than) the target energy
    par: [MASS_A MASS_B EPSILON_S D_X LAMBDA ALPHA]
     
    OUTPUTS
    x0_PO:    Initial condition of the periodic orbit (P.O.)
    T_PO:     Time period 
    ePO:     Energy of the computed P.O.
    """
     

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
        [x0po_iFam,tfpo_iFam] = get_PODiffCorr(c, diffcorr_setup_model, \
                                                    conv_coord_model, \
                                                    diffcorr_acc_corr_model, ham2dof_model, \
                                                    half_period_model, pot_energy_model, \
                                                    varEqns_model, plot_iter_orbit_model, \
                                                    par)
        
        energyPO = get_total_energy(x0po_iFam, pot_energy_model, par)
    #         x0po(iFam,1:N) = x0po_iFam ;
    #         T(iFam,1)      = 2*tfpo_iFam ;
        
        c = x0po_iFam;
        iter = iter + 1;
        
        if (abs(get_total_energy(c, pot_energy_model, par) - energyTarget) < \
            energyTol) or (iter == iterMax):
            
            print('Initial condition: %s\n' %c);
            print('Energy of the initial condition for PO %s\n' %get_total_energy(c, \
                                                                  pot_energy_model, par) );
            x0_PO = c
            T_PO = 2*tfpo_iFam; 
            ePO = get_total_energy(c, pot_energy_model, par);
            break
        
        
        if np.sign( get_total_energy(c, pot_energy_model, par) - energyTarget ) == \
            np.sign ( get_total_energy(a, pot_energy_model, par) - energyTarget ):
            a = c
        else:
            b = c;
        print('Iteration number %s, energy of PO: %s\n' %(iter, energyPO)) ;
        

    print('Iterations completed: %s, error in energy: %s \n' %(iter, \
                                                                abs(get_total_energy(c, \
                                                                pot_energy_model, par) - \
                                                                energyTarget)));
    
    
    dum =np.concatenate((c, 2*tfpo_iFam, energyPO),axis=None)
    np.savetxt(po_target_file.name, dum ,fmt='%1.16e')
    
    return x0_PO, T_PO, ePO