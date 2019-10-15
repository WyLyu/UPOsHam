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
def init_guess_eqpt_deleonberne(eqNum, par):
    """
    Returns configuration space coordinates of the equilibrium points according to the index:
    Saddle (EQNUM=1)
    Centre (EQNUM=2,3)
    """
    
    if eqNum == 1:
        x0 = [0, 0]
    elif eqNum == 2:
        x0 = [0, 1/np.sqrt(2)]  # EQNUM = 2, center-center
    elif eqNum == 3:
        x0 = [0, -1/np.sqrt(2)] # EQNUM = 3, center-center
    
    return x0


def grad_pot_deleonberne(x, par):
    """ Returns the gradient of the potential energy function V(x,y) """
     
    dVdx = -2*par[3]*par[4]*np.exp(-par[4]*x[0])*(np.exp(-par[4]*x[0]) - 1) - \
        4*par[5]*par[4]*x[1]**2*(x[1]**2 - 1)*np.exp(-par[5]*par[4]*x[0])
    dVdy = 8*x[1]*(2*x[1]**2 - 1)*np.exp(-par[5]*par[4]*x[0])
    
    F = [-dVdx, -dVdy]
    
    return F

def pot_energy_deleonberne(x, y, par):
    """ Returns the potential energy function V(x,y) """
    
    return par[3]*( 1 - np.exp(-par[4]*x) )**2 + 4*y**2*(y**2 - 1)*np.exp(-par[5]*par[4]*x) + par[2]


def eigvector_deleonberne(par):
    """ Returns the correction factor to the eigenvectors for the linear guess """

    correcx = 1
    correcy = 0
    
    return correcx, correcy


def guess_lin_deleonberne(eqPt, Ax, par):
    """ Returns an initial guess for the unstable periodic orbit """
    
    correcx, correcy = eigvector_deleonberne(par)
    

    return [eqPt[0] - Ax*correcx,eqPt[1] + Ax*correcy,0,0]


def jacobian_deleonberne(eqPt, par):
    """ Returns Jacobian of the Hamiltonian vector field """
    
    x,y,px,py = eqPt[0:4]
    # The first order derivative of the Hamiltonian.
    dVdx = - 2*par[3]*par[4]*np.exp(-par[4]*x)*(np.exp(-par[4]*x) - 1) - 4*par[5]*par[4]*y**2*(y**2 - 1)*np.exp(-par[5]*par[4]*x) 
    dVdy = 8*y*(2*y**2 - 1)*np.exp(-par[5]*par[4]*x)

    # The following is the Jacobian matrix 
    d2Vdx2 = - ( 2*par[3]*par[4]**2*( np.exp(-par[4]*x) - 2.0*np.exp(-2*par[4]*x) ) - 4*(par[5]*par[4])**2*x**2*(y**2 - 1)*np.exp(-par[5]*par[4]*x) )
        
    d2Vdy2 = 8*(6*y**2 - 1)*np.exp( -par[4]*par[5]*x )

    d2Vdydx = -8*y*par[4]*par[5]*np.exp( -par[4]*par[5]*x )*(2*y**2 - 1)

    d2Vdxdy = d2Vdydx    

    Df = np.array([[  0,     0,    par[0],    0],
                   [0,     0,    0,    par[1]],
                   [-d2Vdx2,  -d2Vdydx,   0,    0],
                   [-d2Vdxdy, -d2Vdy2,    0,    0]])
    
    return Df


def varEqns_deleonberne(t,PHI,par):
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
    dVdx = - 2*par[3]*par[4]*np.exp(-par[4]*x)*(np.exp(-par[4]*x) - 1) - 4*par[5]*par[4]*y**2*(y**2 - 1)*np.exp(-par[5]*par[4]*x) 
    dVdy = 8*y*(2*y**2 - 1)*np.exp(-par[5]*par[4]*x)

    # The second order derivative of the potential energy. 
    d2Vdx2 = - ( 2*par[3]*par[4]**2*( np.exp(-par[4]*x) - 2.0*np.exp(-2*par[4]*x) ) - 4*(par[5]*par[4])**2*x**2*(y**2 - 1)*np.exp(-par[5]*par[4]*x) )
        
    d2Vdy2 = 8*(6*y**2 - 1)*np.exp( -par[4]*par[5]*x )

    d2Vdydx = -8*y*par[4]*par[5]*np.exp( -par[4]*par[5]*x )*(2*y**2 - 1)
        
    
    
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


def diffcorr_setup_deleonberne():
    """ 
    Returns settings for differential correction method 
        
    Settings include choosing coordinates for event criteria, convergence criteria, and 
    correction (see references for details on how to choose these coordinates).
    """
    
    dydot1 = 1
    correcty0= 0
    MAXdydot1 = 1.e-10
    drdot1 = dydot1
    correctr0 = correcty0
    MAXdrdot1 = MAXdydot1
    
    return [drdot1, correctr0, MAXdrdot1]


def conv_coord_deleonberne(x1, y1, dxdot1, dydot1):
    return dydot1


def diffcorr_acc_corr_deleonberne(coords, phi_t1, x0, par):
    """ 
    Returns the new guess initial condition of the unstable periodic orbit after applying 
    small correction to the guess. 
        
    Correcting x or y coordinate depends on the problem and needs to chosen by inspecting the 
    geometry of the bottleneck in the potential energy surface.
    """
    
    x1, y1, dxdot1, dydot1 = coords
    
    dVdx = -2*par[3]*par[4]*np.exp(-par[4]*x1)*(np.exp(-par[4]*x1) - 1) - 4*par[5]*par[4]*y1**2*(y1**2 - 1)*np.exp(-par[5]*par[4]*x1)
    dVdy = 8*y1*(2*y1**2 - 1)*np.exp(-par[5]*par[4]*x1)
    vxdot1 = -dVdx
    vydot1 = -dVdy
    #correction to the initial y0
    correcty0 = 1/(phi_t1[3,1] - phi_t1[2,1]*vydot1*(1/vxdot1))*dydot1
    x0[1] = x0[1] - correcty0

    return x0



def plot_iter_orbit_deleonberne(x, ax, e, par):
    """ 
    Plots the orbit in the 3D space of (x,y,p_y) coordinates with the initial and 
    final points marked 
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



def ham2dof_deleonberne(t, x, par):
    """ Returns the Hamiltonian vector field (Hamilton's equations of motion) """
    
    xDot = np.zeros(4)
    
    dVdx = -2*par[3]*par[4]*np.exp(-par[4]*x[0])*(np.exp(-par[4]*x[0]) - 1) - \
        4*par[5]*par[4]*x[1]**2*(x[1]**2 - 1)*np.exp(-par[5]*par[4]*x[0])
    dVdy = 8*x[1]*(2*x[1]**2 - 1)*np.exp(-par[5]*par[4]*x[0])
        
    xDot[0] = x[2]/par[0]
    xDot[1] = x[3]/par[1]
    xDot[2] = -dVdx 
    xDot[3] = -dVdy
    
    return list(xDot)  



def half_period_deleonberne(t,x,par):
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
    
    return x[2]


#% End problem specific functions
    
        
#%% Setting up parameters and global variables
    
N = 4          # dimension of phase space
MASS_A = 8.0 
MASS_B = 8.0 # De Leon, Marston (1989)
EPSILON_S = 1.0
D_X = 10.0
ALPHA = 1.00
LAMBDA = 1.5
parameters = np.array([MASS_A, MASS_B, EPSILON_S, D_X, LAMBDA, ALPHA])
eqNum = 1  
model = 'deleonberne'
eqPt = diffcorr_UPOsHam2dof.get_eq_pts(eqNum, init_guess_eqpt_deleonberne, \
                                       grad_pot_deleonberne, parameters)

#energy of the saddle eq pt
eSaddle = diffcorr_UPOsHam2dof.get_total_energy([eqPt[0],eqPt[1],0,0], pot_energy_deleonberne, \
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

#  get the initial conditions and periods for a family of periodic orbits


po_fam_file = open("x0_diffcorr_fam_eqPt%s_deleonberne.txt" %eqNum,'a+')
#[po_x0Fam,po_tpFam] = diffcorr_UPOsHam2dof.get_POFam(eqNum, Ax1, Ax2,nFam, po_fam_file, parameters,model)  
[po_x0Fam,po_tpFam] = diffcorr_UPOsHam2dof.get_POFam(eqNum, Ax1, Ax2, nFam, \
                                                    po_fam_file, init_guess_eqpt_deleonberne, \
                                                    grad_pot_deleonberne, jacobian_deleonberne, \
                                                    guess_lin_deleonberne, diffcorr_setup_deleonberne, \
                                                    conv_coord_deleonberne, \
                                                    diffcorr_acc_corr_deleonberne, ham2dof_deleonberne, \
                                                    half_period_deleonberne, pot_energy_deleonberne, \
                                                    varEqns_deleonberne, plot_iter_orbit_deleonberne, \
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
    
    po_fam_file = open("x0_diffcorr_fam_eqPt%s_deleonberne.txt" %eqNum ,'a+')
    eTarget = eSaddle + deltaE 
    print('Loading the periodic orbit family from data file',po_fam_file.name,'\n') 
    x0podata = np.loadtxt(po_fam_file.name)
    po_fam_file.close()

    #%
    po_brac_file = open("x0po_T_energyPO_eqPt%s_brac%s_deleonberne.txt" %(eqNum,deltaE),'a+')
    t = time.time()
    # [x0poTarget,TTarget] = bracket_POEnergy_bp(eTarget, x0podata, po_brac_file)
    x0poTarget,TTarget = diffcorr_UPOsHam2dof.poBracketEnergy(eTarget, x0podata, po_brac_file, \
                                                              diffcorr_setup_deleonberne, \
                                                              conv_coord_deleonberne, \
                                                              diffcorr_acc_corr_deleonberne, \
                                                              ham2dof_deleonberne, \
                                                              half_period_deleonberne, \
                                                              pot_energy_deleonberne, varEqns_deleonberne, \
                                                              plot_iter_orbit_deleonberne, \
                                                              parameters)
    poTarE_runtime = time.time()-t
    model_parameters_file = open("model_parameters_eqPt%s_DelE%s_deleonberne.txt" %(eqNum,deltaE),'a+')
    np.savetxt(model_parameters_file.name, parameters,fmt='%1.16e')
    model_parameters_file.close()
    po_brac_file.close()
    
    


    
    #%
    # target specific periodic orbit
    # Target PO of specific energy with high precision does not work for the
    # model 
    
    po_target_file = open("x0_diffcorr_deltaE%s_deleonberne.txt" %(deltaE),'a+')
                    
    [x0po, T,energyPO] = diffcorr_UPOsHam2dof.poTargetEnergy(x0poTarget,eTarget, \
                                                            po_target_file, \
                                                            diffcorr_setup_deleonberne, \
                                                            conv_coord_deleonberne, \
                                                            diffcorr_acc_corr_deleonberne, \
                                                            ham2dof_deleonberne, half_period_deleonberne, \
                                                            pot_energy_deleonberne, varEqns_deleonberne, \
                                                            plot_iter_orbit_deleonberne, \
                                                            parameters)
    
    po_target_file.close()



#%% Load periodic orbit data from ascii files
    
x0po = np.zeros((4,len(deltaE_vals)))

for i in range(len(deltaE_vals)):
    deltaE = deltaE_vals[i]

    po_fam_file = open("x0_diffcorr_deltaE%s_deleonberne.txt" %(deltaE),'a+')
    print('Loading the periodic orbit family from data file',po_fam_file.name,'\n') 
    x0podata = np.loadtxt(po_fam_file.name)
    po_fam_file.close()
    x0po[:,i] = x0podata[0:4]


#%% Plotting the Family
    
TSPAN = [0,30]
RelTol = 3.e-10
AbsTol = 1.e-10

plt.close('all')
axis_fs = 15


for i in range(len(deltaE_vals)):
    deltaE = deltaE_vals[i]
    f= lambda t,x: ham2dof_deleonberne(t,x,parameters)
    soln = solve_ivp(f, TSPAN, x0po[:,i],method='RK45',dense_output=True, \
                     events = lambda t,x : half_period_deleonberne(t,x,parameters), \
                     rtol=RelTol, atol=AbsTol)
    
    te = soln.t_events[0]
    tt = [0,te[1]] 
    t,x,phi_t1,PHI = diffcorr_UPOsHam2dof.stateTransitMat(tt,x0po[:,i],parameters,varEqns_deleonberne)
    ax = plt.gca(projection='3d')
    ax.plot(x[:,0],x[:,1],x[:,2],'-',color = linecolor[i], label = '$\Delta E$ = %.2f'%(deltaE))
    ax.plot(x[:,0],x[:,1],-x[:,2],'-',color = linecolor[i])
    ax.scatter(x[0,0],x[0,1],x[0,2],s=20,marker='*')
    ax.scatter(x[0,0],x[0,1],-x[0,2],s=20,marker='o')
    ax.plot(x[:,0], x[:,1], zs=0, zdir='z')    
    


ax = plt.gca(projection='3d')
resX = 100
xVec = np.linspace(-1,1,resX)
yVec = np.linspace(-2,2,resX)
xMat, yMat = np.meshgrid(xVec, yVec)
cset1 = ax.contour(xMat, yMat, \
                   diffcorr_UPOsHam2dof.get_pot_surf_proj(xVec, yVec, pot_energy_deleonberne, \
                                                          parameters), [0.01,0.1,1,2,4], \
                                                          zdir='z', offset=0, 
                                                          linewidths = 1.0, cmap=cm.viridis, \
                                                          alpha = 0.8)

ax.scatter(eqPt[0], eqPt[1], s = 200, c = 'r', marker = 'X')
ax.set_xlabel('$x$', fontsize=axis_fs)
ax.set_ylabel('$y$', fontsize=axis_fs)
ax.set_zlabel('$p_x$', fontsize=axis_fs)
#ax.set_title('$\Delta E$ = %1.e,%1.e,%1.e,%1.e,%1.e' \
#                %(energyPO_1,energyPO_2,energyPO_3,energyPO_4,energyPO_5) ,fontsize=axis_fs)

legend = ax.legend(loc='upper left')
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_zlim(-4, 4)

plt.grid()
plt.show()

plt.savefig('diffcorr_POfam_deleonberne.pdf',format='pdf',bbox_inches='tight')