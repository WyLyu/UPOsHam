# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 10:02:48 2019

@author: Wenyang Lyu and Shibabrat Naik

Script to compute unstable periodic orbits at specified energies for the coupled quartic 
Hamiltonian using differential correction
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time
import diffcorr_UPOsHam2dof ### import module xxx where xxx is the name of the python file xxx.py 
import matplotlib as mpl
from matplotlib import cm
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'


#% Begin problem specific functions
def init_guess_eqpt_uncoupled(eqNum, par):
    """
    Returns configuration space coordinates of the equilibrium points according to the index:
    Saddle (EQNUM=1)
    Centre (EQNUM=2,3)
    """
    
    if eqNum == 1:
        x0 = [0, 0]
    elif eqNum == 2:
        x0 = [np.sqrt(par[3]/par[4]),0] 
    elif eqNum == 3:
        x0 = [-np.sqrt(par[3]/par[4]),0]  
    
    return x0


def grad_pot_uncoupled(x,par):
    """ Returns the gradient of the potential energy function V(x,y) """ 
    
    dVdx = -par[3]*x[0]+par[4]*(x[0])**3
    dVdy = par[5]*x[1]
    
    F = [-dVdx, -dVdy]
    
    return F


def pot_energy_uncoupled(x, y, par):
    """ Returns the potential energy function V(x,y) """
    
    return -0.5*par[3]*x**2+0.25*par[4]*x**4 +0.5*par[5]*y**2


def eigvector_uncoupled(par):
    """ Returns the correction factor to the eigenvectors for the linear guess """

    correcx = 1
    correcy = 1
    
    return correcx, correcy


def guess_lin_uncoupled(eqPt, Ax, par):
    """ Returns an initial guess for the unstable periodic orbit """ 
    
    correcx, correcy = eigvector_uncoupled(par)
    
    return [eqPt[0] + Ax*correcx,eqPt[1] + Ax*correcy,0,0]


def jacobian_uncoupled(eqPt, par):
    """ Returns Jacobian of the Hamiltonian vector field """
    
    x,y,px,py = eqPt[0:4]
    
    # The first order derivative of the Hamiltonian.
    dVdx = -par[3]*x+par[4]*x**3
    dVdy = par[5]*y

    # The following is the Jacobian matrix 
    d2Vdx2 = -par[3]+par[4]*3*x**2
        
    d2Vdy2 = par[5]

    d2Vdydx = 0
        
    d2Vdxdy = d2Vdydx    

    Df = np.array([[  0,     0,    par[0],    0],
                   [0,     0,    0,    par[1]],
                   [-d2Vdx2,  -d2Vdydx,   0,    0],
                   [-d2Vdxdy, -d2Vdy2,    0,    0]])
    
    return Df


def varEqns_uncoupled(t,PHI,par):
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
    dVdx = -par[3]*x+par[4]*x**3
    dVdy = par[5]*y

    # The second order derivative of the potential energy.  
    d2Vdx2 = -par[3]+par[4]*3*x**2
        
    d2Vdy2 = par[5]

    d2Vdydx = 0

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


def diffcorr_setup_uncoupled():
    """ 
    Returns settings for differential correction method 
        
    Settings include choosing coordinates for event criteria, convergence criteria, and 
    correction (see references for details on how to choose these coordinates).
    """
    
    dxdot1 = 1
    correctx0= 0
    MAXdxdot1 = 1.e-10 
    drdot1 = dxdot1
    correctr0 = correctx0
    MAXdrdot1 = MAXdxdot1
    
    return [drdot1, correctr0, MAXdrdot1]


def conv_coord_uncoupled(x1, y1, dxdot1, dydot1):
    return dxdot1


def diffcorr_acc_corr_uncoupled(coords, phi_t1, x0, par):
    """ 
    Returns the new guess initial condition of the unstable periodic orbit after applying 
    small correction to the guess. 
        
    Correcting x or y coordinate depends on the problem and needs to chosen by inspecting the 
    geometry of the bottleneck in the potential energy surface.
    """
    
    x1, y1, dxdot1, dydot1 = coords
    
    dVdx = -par[3]*x1+par[4]*(x1)**3
    dVdy = par[5]*y1
    vxdot1 = -dVdx
    vydot1 = -dVdy
    #correction to the initial x0
    correctx0 = dxdot1/(phi_t1[2,0] - phi_t1[3,0]*(vxdot1/vydot1))
    x0[0] = x0[0] - correctx0 # correction in x coodinate.

    return x0


def plot_iter_orbit_uncoupled(x, ax, e, par):
    """ 
    Plots the orbit in the 3D space of (x,y,p_y) coordinates with the initial and 
    final points marked 
    """
    
    label_fs = 10
    axis_fs = 15 # fontsize for publications 
    
    ax.plot(x[:,0],x[:,1],x[:,3],'-')
    ax.plot(x[:,0],x[:,1],-x[:,3],'--')
    ax.scatter(x[0,0],x[0,1],x[0,3],s=20,marker='*')
    ax.scatter(x[-1,0],x[-1,1],x[-1,3],s=20,marker='o')
    ax.set_xlabel(r'$x$', fontsize=axis_fs)
    ax.set_ylabel(r'$y$', fontsize=axis_fs)
    ax.set_zlabel(r'$p_y$', fontsize=axis_fs)

    return 


def ham2dof_uncoupled(t, x, par):
    """ Returns the Hamiltonian vector field (Hamilton's equations of motion) """
    
    xDot = np.zeros(4)
    
    dVdx = -par[3]*x[0]+par[4]*(x[0])**3
    dVdy = par[5]*x[1]
        
    xDot[0] = x[2]/par[0]
    xDot[1] = x[3]/par[1]
    xDot[2] = -dVdx 
    xDot[3] = -dVdy
    
    return list(xDot)  


def half_period_uncoupled(t, x, par):
    """ 
    Returns the coordinate for the half-period event for the unstable periodic orbit                          
    
    xDot = x[0]
    yDot = x[1]
    pxDot = x[2]
    pyDot = x[3]
    """
    
    terminal = True
    direction = 0 #0: all directions of crossing, zero can be approached from either direction

    return x[3]
 

#% End problem specific functions

#%% Setting up parameters and global variables

N = 4          # dimension of phase space
omega=1.00
EPSILON_S = 0.0 #Energy of the saddle
alpha = 1.00
beta = 1.00
epsilon= 1e-1
parameters = np.array([1, omega, EPSILON_S, alpha, beta, omega, epsilon])
eqNum = 1 
model = 'uncoupled'
#eqPt = diffcorr_UPOsHam2dof.get_eq_pts(eqNum,model, parameters)
eqPt = diffcorr_UPOsHam2dof.get_eq_pts(eqNum, init_guess_eqpt_uncoupled, \
                                       grad_pot_uncoupled, parameters)

#energy of the saddle eq pt
eSaddle = diffcorr_UPOsHam2dof.get_total_energy([eqPt[0],eqPt[1],0,0], pot_energy_uncoupled, \
                                                parameters) 

#If orbit is an n-array, e.g. orbit = [orbit_0, orbit_1, ..., orbit_n]
# 
#e = np.zeros(n,1)    
#for i in range(n):
#    e[i] = get_total_energy_deleonberne(orbit[i], parameters)
#e = np.mean(e)

#%%
nFam = 100 # use nFam = 10 for low energy

# first two amplitudes for continuation procedure to get p.o. family
Ax1  = 2.e-5 # initial amplitude (1 of 2) values to use: 2.e-3
Ax2  = 2*Ax1 # initial amplitude (2 of 2)

t = time.time()

# get the initial conditions and periods for a family of periodic orbits
po_fam_file = open("x0_diffcorr_fam_eqPt%s_uncoupled.txt" %eqNum,'a+')
[po_x0Fam,po_tpFam] = diffcorr_UPOsHam2dof.get_POFam(eqNum, Ax1, Ax2, nFam, \
                                                    po_fam_file, init_guess_eqpt_uncoupled, \
                                                    grad_pot_uncoupled, jacobian_uncoupled, \
                                                    guess_lin_uncoupled, diffcorr_setup_uncoupled, \
                                                    conv_coord_uncoupled, \
                                                    diffcorr_acc_corr_uncoupled, ham2dof_uncoupled, \
                                                    half_period_uncoupled, pot_energy_uncoupled, \
                                                    varEqns_uncoupled, plot_iter_orbit_uncoupled, \
                                                    parameters)

poFamRuntime = time.time()-t
x0podata = np.concatenate((po_x0Fam, po_tpFam),axis=1)
po_fam_file.close()

#%%
# begins with a family of periodic orbits and steps until crossing the
# initial condition with target energy 
# fileName = 'x0po_T_energy_case1_L41.txt'
# fileName = 'x0po_T.txt'

#deltaE_vals = [0.01, 0.1, 1.00, 2.0, 4.0]
#linecolor = ['b','r','g','m','c']
deltaE_vals = [0.1, 1.00]
linecolor = ['b','r']

for i in range(len(deltaE_vals)):
    deltaE = deltaE_vals[i]
    
    po_fam_file = open("x0_diffcorr_fam_eqPt%s_uncoupled.txt" %eqNum ,'a+')
    eTarget = eSaddle + deltaE 
    print('Loading the periodic orbit family from data file',po_fam_file.name,'\n') 
    x0podata = np.loadtxt(po_fam_file.name)
    po_fam_file.close()
    
    
    #%
    po_brac_file = open("x0po_T_energyPO_eqPt%s_brac%s_uncoupled.txt" %(eqNum,deltaE),'a+')
    t = time.time()
    # [x0poTarget,TTarget] = bracket_POEnergy_bp(eTarget, x0podata, po_brac_file)
    x0poTarget,TTarget = diffcorr_UPOsHam2dof.poBracketEnergy(eTarget, x0podata, po_brac_file, \
                                                              diffcorr_setup_uncoupled, \
                                                              conv_coord_uncoupled, \
                                                              diffcorr_acc_corr_uncoupled, \
                                                              ham2dof_uncoupled, \
                                                              half_period_uncoupled, \
                                                              pot_energy_uncoupled, varEqns_uncoupled, \
                                                              plot_iter_orbit_uncoupled, \
                                                              parameters)

    poTarE_runtime = time.time()-t
    model_parameters_file = open("model_parameters_eqPt%s_DelE%s_uncoupled.txt" %(eqNum,deltaE),'a+')
    np.savetxt(model_parameters_file.name, parameters,fmt='%1.16e')
    model_parameters_file.close()
    po_brac_file.close()
    
    
    # target specific periodic orbit
    # Target PO of specific energy with high precision does not work for the
    # model 
    
    po_target_file = open("x0_diffcorr_deltaE%s_uncoupled.txt" %(deltaE),'a+')

    [x0po, T,energyPO] = diffcorr_UPOsHam2dof.poTargetEnergy(x0poTarget,eTarget, \
                                                        po_target_file, \
                                                        diffcorr_setup_uncoupled, \
                                                        conv_coord_uncoupled, \
                                                        diffcorr_acc_corr_uncoupled, \
                                                        ham2dof_uncoupled, half_period_uncoupled, \
                                                        pot_energy_uncoupled, varEqns_uncoupled, \
                                                        plot_iter_orbit_uncoupled, \
                                                        parameters)
#    [x0po, T,energyPO] = diffcorr_UPOsHam2dof.poTargetEnergy(x0poTarget,eTarget, 
#                                                            po_target_file,parameters,model)
    
    po_target_file.close()


#%% Load periodic orbit data from ascii files
    
x0po = np.zeros((4,len(deltaE_vals)))

for i in range(len(deltaE_vals)):
    deltaE = deltaE_vals[i]

    po_fam_file = open("x0_diffcorr_deltaE%s_uncoupled.txt" %(deltaE),'a+')
    print('Loading the periodic orbit family from data file',po_fam_file.name,'\n') 
    x0podata = np.loadtxt(po_fam_file.name)
    po_fam_file.close()
    x0po[:,i] = x0podata[0:4]


#%% Plotting the unstable periodic orbits at the specified energies

TSPAN = [0,30]
plt.close('all')
axis_fs = 15
RelTol = 3.e-10
AbsTol = 1.e-10

for i in range(len(deltaE_vals)):
    deltaE = deltaE_vals[i]
    f= lambda t,x: ham2dof_uncoupled(t,x,parameters)
    soln = solve_ivp(f, TSPAN, x0po[:,i],method='RK45',dense_output=True, \
                     events = lambda t,x : half_period_uncoupled(t,x,parameters), \
                     rtol=RelTol, atol=AbsTol)
    
    te = soln.t_events[0]
    tt = [0,te[1]]
    t,x,phi_t1,PHI = diffcorr_UPOsHam2dof.stateTransitMat(tt, x0po[:,i], parameters, \
                                                          varEqns_uncoupled)
    ax = plt.gca(projection='3d')
    ax.plot(x[:,0],x[:,1],x[:,3],'-',color = linecolor[i], label = '$\Delta E$ = %.2f'%(deltaE))
    ax.plot(x[:,0],x[:,1],-x[:,3],'-',color = linecolor[i])
    ax.scatter(x[0,0],x[0,1],x[0,3],s=20,marker='*')
    ax.scatter(x[0,0],x[0,1],-x[0,3],s=20,marker='o')
    ax.plot(x[:,0], x[:,1], zs=0, zdir='z')


ax = plt.gca(projection='3d')
resX = 100
xVec = np.linspace(-4,4,resX)
yVec = np.linspace(-4,4,resX)
xMat, yMat = np.meshgrid(xVec, yVec)
cset1 = ax.contour(xMat, yMat, \
                   diffcorr_UPOsHam2dof.get_pot_surf_proj(xVec, yVec, pot_energy_uncoupled, \
                                                          parameters), [0.01,0.1,1,2,4], \
                                                          zdir='z', offset=0, 
                                                          linewidths = 1.0, cmap=cm.viridis, \
                                                          alpha = 0.8)

ax.scatter(eqPt[0], eqPt[1], s = 200, c = 'r', marker = 'X')
ax.set_xlabel('$x$', fontsize=axis_fs)
ax.set_ylabel('$y$', fontsize=axis_fs)
ax.set_zlabel('$p_y$', fontsize=axis_fs)
#ax.set_title('$\Delta E$ = %1.e,%1.e,%1.e,%1.e,%1.e' \
#                %(energyPO_1,energyPO_2,energyPO_3,energyPO_4,energyPO_5) ,fontsize=axis_fs)

legend = ax.legend(loc='upper left')
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_zlim(-4, 4)

plt.grid()
plt.show()

plt.savefig('diffcorr_POfam_uncoupled.pdf', format='pdf', bbox_inches='tight')




