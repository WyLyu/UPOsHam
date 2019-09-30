# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 17:57:34 2019

@author: Wenyang Lyu and Shibabrat Naik
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import math
from scipy import optimize
import matplotlib as mpl
from matplotlib import cm

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'

import tpcd_UPOsHam2dof ### import module xxx where xxx is the name of the python file xxx.py 

#% Begin problem specific functions
def init_guess_eqpt_deleonberne(eqNum, par):
    
    if eqNum == 1:
        x0 = [0, 0]
    if eqNum == 2:
        x0 = [0, 1/np.sqrt(2)]  # EQNUM = 2, center-center
    elif eqNum == 3:
        x0 = [0, -1/np.sqrt(2)] # EQNUM = 3, center-center
    
    return x0


def grad_pot_deleonberne(x, par):
    
    dVdx = -2*par[3]*par[4]*np.exp(-par[4]*x[0])*(np.exp(-par[4]*x[0]) - 1) - \
        4*par[5]*par[4]*x[1]**2*(x[1]**2 - 1)*np.exp(-par[5]*par[4]*x[0])
    dVdy = 8*x[1]*(2*x[1]**2 - 1)*np.exp(-par[5]*par[4]*x[0])
    
    F = [-dVdx, -dVdy]
    
    return F

def pot_energy_deleonberne(x, y, par):
    return par[3]*( 1 - np.exp(-par[4]*x) )**2 + 4*y**2*(y**2 - 1)*np.exp(-par[5]*par[4]*x) + par[2]



def get_coord_deleonberne(x,y, E, par):
    """ 
    this function returns the initial position of x/y-coordinate on the potential energy 
    surface(PES) for a specific energy E.
    """

    return par[3]*( 1 - math.e**(-par[4]*x) )**2 + \
                4*y**2*(y**2 - 1)*math.e**(-par[5]*par[4]*x) + par[2] - E


def varEqns_deleonberne(t,PHI,par):
    """
    PHIdot = varEqns_deleonberne(t,PHI) 
    
    This here is a preliminary state transition, PHI(t,t0),
    matrix equation attempt for a ball rolling on the surface, based on...
    
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


def configdiff_deleonberne(guess1, guess2, ham2dof_model, half_period_model, n_turn, par):
    """
    configdiff_deleonberne(model,guess1, guess2,n_turn,par) returns the difference of x(or y) 
    coordinates between the guess initial conditions and the ith turning points
    n_turn is the nth turning point we want to choose as our 'turning point' for defining the 
    dot product
    either difference in x coordintes(x_diff1, x_diff2) or difference in 
    y coordinates(y_diff1, y_diff2) is returned as the result.
    """
    TSPAN = [0,40]
    RelTol = 3.e-10
    AbsTol = 1.e-10 
    
    f1 = lambda t,x: ham2dof_model(t,x,par) 
    soln1 = solve_ivp(f1, TSPAN, guess1, method='RK45', dense_output=True, \
                      events = lambda t,x: half_period_model(t, x, par), rtol=RelTol, atol=AbsTol)
    te1 = soln1.t_events[0]
    t1 = [0,te1[n_turn]]#[0,te1[1]]
    turn1 = soln1.sol(t1)
    x_turn1 = turn1[0,-1] 
    y_turn1 = turn1[1,-1]
    x_diff1 = guess1[0] - x_turn1
    y_diff1 = guess1[1] - y_turn1
    
    f2 = lambda t,x: ham2dof_model(t,x,par) 
    soln2 = solve_ivp(f2, TSPAN, guess2,method='RK45', dense_output=True, \
                      events = lambda t,x: half_period_model(t, x, par), rtol=RelTol, atol=AbsTol)
    te2 = soln2.t_events[0]
    t2 = [0,te2[n_turn]]#[0,te2[1]]
    turn2 = soln2.sol(t2)
    x_turn2 = turn2[0,-1] 
    y_turn2 = turn2[1,-1] 
    x_diff2 = guess2[0] - x_turn2
    y_diff2 = guess2[1] - y_turn2
    

    print("Initial guess1%s, initial guess2%s, y_diff1 is %s, y_diff2 is%s " %(guess1, guess2, y_diff1, y_diff2))
        
    return y_diff1, y_diff2


def ham2dof_deleonberne(t, x, par):
    """
    Hamilton's equations of motion
    """
    
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
    Return the turning point where we want to stop the integration                           
    
    pxDot = x[0]
    pyDot = x[1]
    xDot = x[2]
    yDot = x[3]
    """
    
    terminal = True
    # The zero can be approached from either direction
    direction = 0 #0: all directions of crossing
    
    return x[2]


def guess_coords_deleonberne(guess1, guess2, i, n, e, get_coord_model, par):
    """
    Returns x and y (configuration space) coordinates as guess for the next iteration of the 
    turning point based on confifuration difference method
    """
    
    h = (guess2[1] - guess1[1])*i/n # h is defined for dividing the interval
    print("h is ",h)
    yguess = guess1[1]+h
    f = lambda x: get_coord_model(x,yguess,e,par)
    xguess = optimize.newton(f,-0.2)   # to find the x coordinate for a given y
    
    return xguess, yguess
    
def plot_iter_orbit_deleonberne(x, ax, e, par):
    
    label_fs = 10
    axis_fs = 15 # fontsize for publications 
    
    ax.plot(x[:,0],x[:,1],x[:,2],'-')
    ax.plot(x[:,0],x[:,1],-x[:,2],'--')
    ax.scatter(x[0,0],x[0,1],x[0,2],s=20,marker='*')
    ax.scatter(x[-1,0],x[-1,1],x[-1,2],s=20,marker='o')
    
    ax.set_xlabel(r'$x$', fontsize=axis_fs)
    ax.set_ylabel(r'$y$', fontsize=axis_fs)
    ax.set_zlabel(r'$p_y$', fontsize=axis_fs)
    ax.set_title(r'$\Delta E$ = %e' %(np.mean(e) - par[2]) ,fontsize=axis_fs)
    #par(3) is the energy of the saddle
    ax.set_xlim(-0.1, 0.1)
    
    return 

#% End problem specific functions


#%% Setting up parameters and global variables
N = 4         # dimension of phase space
MASS_A = 8.0
MASS_B = 8.0 # De Leon, Marston (1989)
EPSILON_S = 1.0
D_X = 10.0
ALPHA = 1.00
LAMBDA = 1.5
parameters = np.array([MASS_A, MASS_B, EPSILON_S, D_X, LAMBDA, ALPHA])
eqNum = 1 
model = 'deleonberne'
#eqPt = tpcd_UPOsHam2dof.get_eq_pts(eqNum, model,parameters)
eqPt = tpcd_UPOsHam2dof.get_eq_pts(eqNum, init_guess_eqpt_deleonberne, \
                                       grad_pot_deleonberne, parameters)



#%%
#E_vals = [1.1, 2.00, 3.00, 5.00]
#linecolor = ['b','r','g','m','c']
E_vals = [1.1, 2.00]
linecolor = ['b','r']
"""
e is the total energy
n is the number of intervals we want to divide
n_turn is the nth turning point we want to choose.
"""

n = 4
n_turn = 1
    
for i in range(len(E_vals)):
    
    e = E_vals[i]
    deltaE = e - parameters[2]
    
    """
    Trial initial Condition s.t. one initial condition is on the LHS of the UPO and the 
    other one is on the RHS of the UPO
    """
    f1 = lambda x: get_coord_deleonberne(x,0.06,e,parameters)
    x0_2 = optimize.newton(f1,-0.15)
    state0_2 = [x0_2,0.06,0.0,0.0]

    f2 = lambda x: get_coord_deleonberne(x,-0.05,e,parameters)
    x0_3 = optimize.newton(f2,-0.15)
    state0_3 = [x0_3,-0.05,0.0,0.0]
    
    po_fam_file = open("x0_tpcd_deltaE%s_deleonberne.txt" %(deltaE),'a+')
    [x0po_1, T_1,energyPO_1] = tpcd_UPOsHam2dof.turningPoint_configdiff(state0_2, state0_3, \
                                                                        get_coord_deleonberne, \
                                                                        pot_energy_deleonberne, \
                                                                        varEqns_deleonberne, \
                                                                        configdiff_deleonberne, \
                                                                        ham2dof_deleonberne, \
                                                                        half_period_deleonberne, \
                                                                        guess_coords_deleonberne, \
                                                                        plot_iter_orbit_deleonberne, \
                                                                        parameters, \
                                                                        e,n,n_turn,po_fam_file) 
    po_fam_file.close()


#%% Load periodic orbit data from ascii files
    
x0po = np.zeros((4,len(E_vals))) #each column is a different initial condition

for i in range(len(E_vals)):
    
    e = E_vals[i]
    deltaE = e - parameters[2]

    po_fam_file = open("x0_tpcd_deltaE%s_deleonberne.txt" %(deltaE),'a+')
    print('Loading the periodic orbit family from data file',po_fam_file.name,'\n') 
    x0podata = np.loadtxt(po_fam_file.name)
    po_fam_file.close()
    x0po[:,i] = x0podata[-1,0:4] 


#%% Plotting the Family
    
TSPAN = [0,30]
plt.close('all')
axis_fs = 15
RelTol = 3.e-10
AbsTol = 1.e-10

f = lambda t,x : ham2dof_deleonberne(t,x,parameters) 

ax = plt.gca(projection='3d')

for i in range(len(E_vals)):
    
    e = E_vals[i]
    deltaE = e - parameters[2]
    
    soln = solve_ivp(f, TSPAN, x0po[:,i], method='RK45', dense_output=True, \
                     events = lambda t,x : half_period_deleonberne(t,x,parameters), \
                     rtol=RelTol, atol=AbsTol)
    
    te = soln.t_events[0]
    tt = [0,te[2]]
    t,x,phi_t1,PHI = tpcd_UPOsHam2dof.stateTransitMat(tt, x0po[:,i], parameters, \
                                                      varEqns_deleonberne)
    
    ax.plot(x[:,0],x[:,1],x[:,2],'-',color=linecolor[i], \
            label='$\Delta E$ = %.2f'%(deltaE))
    ax.scatter(x[0,0],x[0,1],x[0,2],s=20,marker='*')
    ax.plot(x[:,0], x[:,1], zs=0, zdir='z')

    
resX = 100
xVec = np.linspace(-1,1,resX)
yVec = np.linspace(-2,2,resX)
xMat, yMat = np.meshgrid(xVec, yVec)
cset1 = ax.contour(xMat, yMat, 
                   tpcd_UPOsHam2dof.get_pot_surf_proj(xVec, yVec, \
                                                      pot_energy_deleonberne, parameters), \
                                                      [0.01,0.1,1,2,4], zdir='z', offset=0, \
                                                      linewidths = 1.0, cmap=cm.viridis, \
                                                      alpha = 0.8)

ax.scatter(eqPt[0], eqPt[1], s = 200, c = 'r', marker = 'X')
ax.set_xlabel('$x$', fontsize=axis_fs)
ax.set_ylabel('$y$', fontsize=axis_fs)
ax.set_zlabel('$p_x$', fontsize=axis_fs)
#ax.set_title('$\Delta E$ = %1.e,%1.e,%1.e,%1.e,%1.e' %(energyPO_1[-1]-parameters[2],energyPO_2[-1]-parameters[2],energyPO_3[-1]-parameters[2],energyPO_4[-1]-parameters[2],energyPO_5[-1]-parameters[2]) ,fontsize=axis_fs)
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_zlim(-4, 4)
legend = ax.legend(loc='upper left')

plt.grid()
plt.show()

plt.savefig('tpcd_POfam_deleonberne.pdf',format='pdf',bbox_inches='tight')