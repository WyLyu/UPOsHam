# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 13:16:47 2019

@author: Wenyang Lyu and Shibabrat Naik
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy import optimize
import math
import sys
sys.path.append('./src/')
import tpcd_UPOsHam2dof ### import module xxx where xxx is the name of the python file xxx.py 
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
    """
    Returns the gradient of the potential energy function V(x,y)
    """  
    
    dVdx = -par[3]*x[0]+par[4]*(x[0])**3
    dVdy = par[5]*x[1]
    
    F = [-dVdx, -dVdy]
    
    return F


def pot_energy_uncoupled(x, y, par):
    """Returns the potential energy function V(x,y)
    """
    
    return -0.5*par[3]*x**2+0.25*par[4]*x**4 +0.5*par[5]*y**2


def get_coord_uncoupled(x,y, E, par):
    """
    Returns the initial position of x/y-coordinate on the potential energy surface(PES) for a specific energy V.
    """
    
    return -0.5*par[3]*x**2+0.25*par[4]*x**4 +0.5*par[5]*y**2 - E
#    elif model== 'deleonberne':
#        return par[3]*( 1 - math.e**(-par[4]*x) )**2 + 4*y**2*(y**2 - 1)*math.e**(-par[5]*par[4]*x) + par[2]-V
#    else:
#        print("The model you are chosen does not exist, enter the function for finding coordinates on the PES for given x or y and V")


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


def configdiff_uncoupled(guess1, guess2, ham2dof_model, half_period_model, n_turn, par):
    """
    Returns the difference of x(or y) coordinates between the guess initial conditions 
    and the ith turning points

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
    
#    if model == 'uncoupled':
    print("Initial guess1%s, initial guess2%s, x_diff1 is %s, x_diff2 is%s " %(guess1, \
                                                                               guess2, \
                                                                               x_diff1, \
                                                                               x_diff2))
#        return x_diff1, x_diff2

#        return x_diff1, x_diff2
#    elif model == 'deleonberne':
#        print("Initial guess1%s, initial guess2%s, y_diff1 is %s, y_diff2 is%s " %(guess1, guess2, y_diff1, y_diff2))
#        return y_diff1, y_diff2
#    else: 
#        print("Need to decide to use either x_diff or y_diff")
#        print("Initial guess1%s, initial guess2%s, x_diff1 is %s, x_diff2 is%s, y_diff1 is %s, y_diff2 is%s " %(guess1,guess2,x_diff1, x_diff2,y_diff1, y_diff2))
        
    return x_diff1, x_diff2 #,y_diff1, y_diff2


def ham2dof_uncoupled(t, x, par):
    """ Returns the Hamiltonian vector field (Hamilton's equations of motion)
    """
    
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
    Return the turning point where we want to stop the integration                           
    
    pxDot = x[0]
    pyDot = x[1]
    xDot = x[2]
    yDot = x[3]
    """
    
    terminal = True
    # The zero can be approached from either direction
    direction = 0 #0: all directions of crossing
    
    return x[3]


def guess_coords_uncoupled(guess1, guess2, i, n, e, get_coord_model, par):
    """
    Returns x and y (configuration space) coordinates as guess for the next iteration of the 
    turning point based on confifuration difference method
    """
    
    h = (guess2[0] - guess1[0])*i/n
    print("h is ",h)
    xguess = guess1[0]+h
    f = lambda y: get_coord_model(xguess,y,e,par)
    yanalytic = math.sqrt((e +0.5*par[3]*xguess**2-0.25*par[4]*xguess**4)/(0.5*par[1])) #uncoupled
    yguess = optimize.newton(f,yanalytic)   # to find the x coordinate for a given y 
    
    return xguess, yguess
    
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
    ax.set_title(r'$\Delta E$ = %e' %(np.mean(e) - par[2]) ,fontsize=axis_fs)
    #par(3) is the energy of the saddle
    ax.set_xlim(-0.1, 0.1)
    
    return 

#% End problem specific functions



#%%
alpha = 1.0
beta = 1.0
omega = 1.0
EPSILON_S = 0.0 #Energy of the saddle
parameters = np.array([1,omega, EPSILON_S, alpha, beta,omega])
eqNum = 1  
model = 'uncoupled'
#eqPt = tpcd_UPOsHam2dof.get_eq_pts(eqNum,model, parameters)
eqPt = tpcd_UPOsHam2dof.get_eq_pts(eqNum, init_guess_eqpt_uncoupled, \
                                       grad_pot_uncoupled, parameters)
"""
Initial Condition for the true periodic orbit
H = T + V where T is the kinetic energy and V is the potential energy. In our example, 
the potential energy V = -0.5*alpha*x**2+0.25*beta*x**4+0.5*omega*y**2.
If we fix x, then y = +- math.sqrt((V + 0.5*alpha*x**2-0.25*beta*x**4)/(0.5*omega) ) so that
the point starts at the potential energy surface V.
"""

#%%

#deltaE_vals = [0.01, 0.1, 1.00, 2.0, 4.0]
#xLeft = [0.0,-0.05,0.01,0.01,0.18]
#xRight = [0.05,0.10,0.18,0.18,0.18]
#linecolor = ['b','r','g','m','c']
deltaE_vals = [0.1, 1.00]
xLeft = [0.0,0.01]
xRight = [0.05,0.18]
linecolor = ['b','r']
"""
e is the total energy
n is the number of intervals we want to divide
n_turn is the nth turning point we want to choose.
"""
for i in range(len(deltaE_vals)):
    
    e = deltaE_vals[i]
    n = 4
    n_turn = 1
    deltaE = e-parameters[2]
    
    """
    Trial initial Condition s.t. one initial condition is on the LHS of the UPO and the 
    other one is on the RHS of the UPO
    """    
    state0_2 = [-0.1 , -math.sqrt(2*e+0.1**2-0.5*0.1**4),0.0,0.0]
    state0_3 = [0.11 , -math.sqrt(2*e+0.11**2-0.5*0.11**4),0.0,0.0]

    po_fam_file = open("x0_tpcd_deltaE%s_uncoupled.txt" %(deltaE),'a+')
    
    [x0po_1, T_1, energyPO_1] = tpcd_UPOsHam2dof.turningPoint_configdiff(state0_2, state0_3, \
                                                                        get_coord_uncoupled, \
                                                                        pot_energy_uncoupled, \
                                                                        varEqns_uncoupled, \
                                                                        configdiff_uncoupled, \
                                                                        ham2dof_uncoupled, \
                                                                        half_period_uncoupled, \
                                                                        guess_coords_uncoupled, \
                                                                        plot_iter_orbit_uncoupled, \
                                                                        parameters, \
                                                                        e,n,n_turn,po_fam_file) 
    
    po_fam_file.close()



#%% Load periodic orbit data from ascii files
    
x0po = np.zeros((4,len(deltaE_vals))) #each column is a different initial condition

for i in range(len(deltaE_vals)):
    
    deltaE = deltaE_vals[i]

    po_fam_file = open("x0_tpcd_deltaE%s_uncoupled.txt" %(deltaE),'a+')
    print('Loading the periodic orbit family from data file',po_fam_file.name,'\n') 
    x0podata = np.loadtxt(po_fam_file.name)
    po_fam_file.close()
    x0po[:,i] = x0podata[-1,0:4] 


#%% Plotting the family of unstable periodic orbits
    
TSPAN = [0,30]
plt.close('all')
axis_fs = 15
RelTol = 3.e-10
AbsTol = 1.e-10


f = lambda t,x : ham2dof_uncoupled(t,x,parameters) 


for i in range(len(deltaE_vals)):
    
    soln = solve_ivp(f, TSPAN, x0po[:,i], method='RK45', dense_output=True, \
                     events = lambda t,x : half_period_uncoupled(t,x,parameters), \
                     rtol=RelTol, atol=AbsTol)
    te = soln.t_events[0]
    tt = [0,te[2]]
    t,x,phi_t1,PHI = tpcd_UPOsHam2dof.stateTransitMat(tt, x0po[:,i], parameters, varEqns_uncoupled)
    
    ax = plt.gca(projection='3d')
    ax.plot(x[:,0],x[:,1],x[:,3],'-', color=linecolor[i], \
            label='$\Delta E$ = %.2f'%(deltaE_vals[i]))
    ax.scatter(x[0,0],x[0,1],x[0,3], s=20, marker='*')
    ax.plot(x[:,0], x[:,1], zs=0, zdir='z')



ax = plt.gca(projection='3d')
resX = 100
xVec = np.linspace(-4,4,resX)
yVec = np.linspace(-4,4,resX)
xMat, yMat = np.meshgrid(xVec, yVec)
cset1 = ax.contour(xMat, yMat, 
                   tpcd_UPOsHam2dof.get_pot_surf_proj(xVec, yVec, \
                                                      pot_energy_uncoupled, parameters), \
                                                      [0.01,0.1,1,2,4], zdir='z', offset=0, \
                                                      linewidths = 1.0, cmap=cm.viridis, \
                                                      alpha = 0.8)
                   
ax.scatter(eqPt[0], eqPt[1], s = 100, c = 'r', marker = 'X')
ax.set_xlabel('$x$', fontsize=axis_fs)
ax.set_ylabel('$y$', fontsize=axis_fs)
ax.set_zlabel('$p_y$', fontsize=axis_fs)
#ax.set_title('$\Delta E$ = %1.e,%1.e,%1.e,%1.e,%1.e' %(energyPO_1[-1],energyPO_2[-1],energyPO_3[-1],energyPO_4[-1],energyPO_5[-1]) ,fontsize=axis_fs)
legend = ax.legend(loc='upper left')
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_zlim(-2, 2)

plt.grid()
plt.show()

plt.savefig('tpcd_POfam_uncoupled.pdf',format='pdf',bbox_inches='tight')

