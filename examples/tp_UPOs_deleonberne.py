# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 10:02:48 2019

@author: Wenyang Lyu and Shibabrat Naik

Compute unstable peridoic orbits at different energies using turning point method
"""

# For the DeLeon-Berne problem
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import math
from scipy import optimize
import sys
sys.path.append('./src/')
import turning_point ### import module xxx where xxx is the name of the python file xxx.py 
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
    """
    Returns the gradient of the potential energy function V(x,y)
    """    
    
    dVdx = -2*par[3]*par[4]*np.exp(-par[4]*x[0])*(np.exp(-par[4]*x[0]) - 1) - \
        4*par[5]*par[4]*x[1]**2*(x[1]**2 - 1)*np.exp(-par[5]*par[4]*x[0])
    dVdy = 8*x[1]*(2*x[1]**2 - 1)*np.exp(-par[5]*par[4]*x[0])
    
    F = [-dVdx, -dVdy]
    
    return F

def pot_energy_deleonberne(x, y, par):
    """Returns the potential energy function V(x,y)
    """
    
    return par[3]*( 1 - np.exp(-par[4]*x) )**2 + 4*y**2*(y**2 - 1)*np.exp(-par[5]*par[4]*x) + par[2]



def get_coord_deleonberne(x,y, E, par):
    """ 
    Returns the initial position of x/y-coordinate on the potential energy 
    surface(PES) for a specific energy E.
    """

    return par[3]*( 1 - math.e**(-par[4]*x) )**2 + \
                4*y**2*(y**2 - 1)*math.e**(-par[5]*par[4]*x) + par[2] - E


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


def ham2dof_deleonberne(t, x, par):
    """ Returns the Hamiltonian vector field (Hamilton's equations of motion)
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
    Returns the turning point where we want to stop the integration                           
    
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
    ax.set_title(r'$\Delta E$ = %e' %(np.mean(e) - par[2]) ,fontsize=axis_fs)
    #par(3) is the energy of the saddle
    ax.set_xlim(-0.1, 0.1)
    
    return 

#% End problem specific functions


#%% Setting up parameters and global variables
#N = 4          # dimension of phase space
#MASS_A = 8.0
#MASS_B = 8.0 # De Leon, Marston (1989)
#EPSILON_S = 1.0
#D_X = 10.0
#ALPHA = 1.00
#LAMBDA = 1.5
#parameters = np.array([MASS_A, MASS_B, EPSILON_S, D_X, LAMBDA, ALPHA])
#eqNum = 1  
#model = 'deleonberne'
#eqPt = turning_point.get_eq_pts(eqNum, model,parameters)

#%% Setting up parameters and global variables
"""
Initial Condition for the true periodic orbit
H = T + V where T is the kinetic energy and V is the potential energy. In our example, 
the potential energy V = -0.5*alpha*x**2+0.25*beta*x**4+0.5*omega*y**2.
If we fix x, then y = +- math.sqrt((V +0.5*alpha*x**2-0.25*beta*x**4)/(0.5*omega) ) so that
the point starts at the potential energy surface V.
"""
N = 4         # dimension of phase space
MASS_A = 8.0
MASS_B = 8.0 # De Leon, Marston (1989)
EPSILON_S = 1.0
D_X = 10.0
ALPHA = 1.00
LAMBDA = 1.5
parameters = np.array([MASS_A, MASS_B, EPSILON_S, D_X, LAMBDA, ALPHA])
eqNum = 1 
#model = 'deleonberne'
#eqPt = turning_point.get_eq_pts(eqNum,model, parameters)
eqPt = turning_point.get_eq_pts(eqNum, init_guess_eqpt_deleonberne, \
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
    
    po_fam_file = open("x0_turningpoint_deltaE%s_deleonberne.txt"%(deltaE),'a+')
    
    [x0po_1, T_1,energyPO_1] = turning_point.turningPoint( state0_2, state0_3, \
                                                            get_coord_deleonberne, \
                                                            guess_coords_deleonberne, \
                                                            ham2dof_deleonberne, \
                                                            half_period_deleonberne, \
                                                            varEqns_deleonberne, \
                                                            pot_energy_deleonberne, \
                                                            plot_iter_orbit_deleonberne, \
                                                            parameters, \
                                                            e,n,n_turn,po_fam_file) 
    
    po_fam_file.close()

#%% Load periodic orbit data from ascii files
    
x0po = np.zeros((4,len(E_vals))) #each column is a different initial condition

for i in range(len(E_vals)):
    
    e = E_vals[i]
    deltaE = e - parameters[2]

    po_fam_file = open("x0_turningpoint_deltaE%s_deleonberne.txt" %(deltaE),'a+')
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
    t,x,phi_t1,PHI = turning_point.stateTransitMat(tt, x0po[:,i], parameters, \
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
                   turning_point.get_pot_surf_proj(xVec, yVec, \
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

plt.savefig('turningpoint_POfam_deleonberne.pdf',format='pdf',bbox_inches='tight')

#%%
# e is the total energy, n is the number of intervals we want to divide, n_turn is the nth turning point we want to choose.
#e=1.1
#n=12
#n_turn = 1
#deltaE = e-parameters[2] #In this case deltaE = 0.1
#"""Trial initial Condition s.t. one initial condition is on the top of the UPO and the other one is on the bottom of the UPO"""
#f1 = lambda x: turning_point.get_coordinate(model,x,0.06,e,parameters)
#x0_2 = optimize.newton(f1,-0.15)
#state0_2 = [x0_2,0.06,0.0,0.0]
#f2 = lambda x: turning_point.get_coordinate(model,x,-0.05,e,parameters)
#x0_3 = optimize.newton(f2,-0.15)
#state0_3 = [x0_3, -0.05,0.0,0.0]
#
#po_fam_file = open("1111x0_turningpoint_deltaE%s_deleonberne.txt" %(0.1),'a+')
#[x0po_1, T_1,energyPO_1] = turning_point.turningPoint(model,state0_3,state0_2 ,parameters,e,n,n_turn,po_fam_file)  
#
#po_fam_file.close()
#
#
##%%
## e is the total energy, n is the number of intervals we want to divide, n_turn is the nth turning point we want to choose.
#e=2.0
#n=12
#n_turn = 1
#deltaE = e-parameters[2]
#"""Trial initial Condition s.t. one initial condition is on the LHS of the UPO and the other one is on the RHS of the UPO"""
#f1 = lambda x: turning_point.get_coordinate(model,x,0.06,e,parameters)
#x0_2 = optimize.newton(f1,-0.15)
#state0_2 = [x0_2,0.06,0.0,0.0]
#f2 = lambda x: turning_point.get_coordinate(model,x,-0.05,e,parameters)
#x0_3 = optimize.newton(f2,-0.15)
#state0_3 = [x0_3, -0.05,0.0,0.0]
#
#po_fam_file = open("1111x0_turningpoint_deltaE%s_deleonberne.txt" %(deltaE),'a+')
#[x0po_2, T_2,energyPO_2] = turning_point.turningPoint(model,state0_3,state0_2 ,parameters,e,n,n_turn,po_fam_file)  
#
#po_fam_file.close()
##%%
## e is the total energy, n is the number of intervals we want to divide, n_turn is the nth turning point we want to choose.
#e=3.0
#n=12
#n_turn = 1
#deltaE = e-parameters[2]
#"""Trial initial Condition s.t. one initial condition is on the LHS of the UPO and the other one is on the RHS of the UPO"""
#f1 = lambda x: turning_point.get_coordinate(model,x,0.06,e,parameters)
#x0_2 = optimize.newton(f1,-0.15)
#state0_2 = [x0_2,0.06,0.0,0.0]
#f2 = lambda x: turning_point.get_coordinate(model,x,-0.05,e,parameters)
#x0_3 = optimize.newton(f2,-0.15)
#state0_3 = [x0_3, -0.05,0.0,0.0]
#
#po_fam_file = open("1111x0_turningpoint_deltaE%s_deleonberne.txt" %(deltaE),'a+')
#[x0po_3, T_3,energyPO_3] = turning_point.turningPoint(model,state0_3,state0_2 ,parameters,e,n,n_turn,po_fam_file)  
#
#po_fam_file.close()
##%%
## e is the total energy, n is the number of intervals we want to divide, n_turn is the nth turning point we want to choose.
#e=5.0
#n=12
#n_turn = 1
#deltaE = e-parameters[2]
#"""Trial initial Condition s.t. one initial condition is on the LHS of the UPO and the other one is on the RHS of the UPO"""
#f1 = lambda x: turning_point.get_coordinate(model,x,0.06,e,parameters)
#x0_2 = optimize.newton(f1,-0.15)
#state0_2 = [x0_2,0.06,0.0,0.0]
#f2 = lambda x: turning_point.get_coordinate(model,x,-0.05,e,parameters)
#x0_3 = optimize.newton(f2,-0.15)
#state0_3 = [x0_3, -0.05,0.0,0.0]
#
#po_fam_file = open("1111x0_turningpoint_deltaE%s_deleonberne.txt" %(deltaE),'a+')
#[x0po_4, T_4,energyPO_4] = turning_point.turningPoint(model,state0_3,state0_2 ,parameters,e,n,n_turn,po_fam_file)  
#
#po_fam_file.close()

#%% Load Data
#deltaE = 0.10
#po_fam_file = open("1111x0_turningpoint_deltaE%s_deleonberne.txt" %(deltaE),'a+')
#print('Loading the periodic orbit family from data file',po_fam_file.name,'\n') 
#x0podata = np.loadtxt(po_fam_file.name)
#po_fam_file.close()
#x0po_1 = x0podata
#
#deltaE = 1.0
#po_fam_file = open("1111x0_turningpoint_deltaE%s_deleonberne.txt" %(deltaE),'a+')
#print('Loading the periodic orbit family from data file',po_fam_file.name,'\n') 
#x0podata = np.loadtxt(po_fam_file.name)
#po_fam_file.close()
#x0po_2 = x0podata
#
#deltaE = 2.0
#po_fam_file = open("1111x0_turningpoint_deltaE%s_deleonberne.txt" %(deltaE),'a+')
#print('Loading the periodic orbit family from data file',po_fam_file.name,'\n') 
#x0podata = np.loadtxt(po_fam_file.name)
#po_fam_file.close()
#x0po_3 = x0podata
#
#deltaE = 4.0
#po_fam_file = open("1111x0_turningpoint_deltaE%s_deleonberne.txt" %(deltaE),'a+')
#print('Loading the periodic orbit family from data file',po_fam_file.name,'\n') 
#x0podata = np.loadtxt(po_fam_file.name)
#po_fam_file.close()
#x0po_4 = x0podata

#%% Plotting the Family
#TSPAN = [0,30]
#plt.close('all')
#axis_fs = 15
#RelTol = 3.e-10
#AbsTol = 1.e-10
#f = lambda t,x : turning_point.Ham2dof(model,t,x,parameters) 
#soln = solve_ivp(f, TSPAN, x0po_1[-1,0:4],method='RK45',dense_output=True, events = lambda t,x : turning_point.half_period(t,x,model),rtol=RelTol, atol=AbsTol)
#te = soln.t_events[0]
#tt = [0,te[1]]
#t,x,phi_t1,PHI = turning_point.stateTransitMat(tt,x0po_1[-1,0:4],parameters,model)
#ax = plt.gca(projection='3d')
#ax.plot(x[:,0],x[:,1],x[:,2],'-',color='b',label='$\Delta E$ = 0.1')
#ax.plot(x[:,0],x[:,1],-x[:,2],'-',color='b')
#ax.scatter(x[0,0],x[0,1],x[0,2],s=20,marker='*')
#ax.scatter(x[0,0],x[0,1],-x[0,2],s=20,marker='o')
#ax.plot(x[:,0], x[:,1], zs=0, zdir='z')
#
#
#f = lambda t,x : turning_point.Ham2dof(model,t,x,parameters) 
#soln = solve_ivp(f, TSPAN, x0po_2[-1,0:4],method='RK45',dense_output=True, events = lambda t,x : turning_point.half_period(t,x,model),rtol=RelTol, atol=AbsTol)
#te = soln.t_events[0]
#tt = [0,te[1]]
#t,x,phi_t1,PHI = turning_point.stateTransitMat(tt,x0po_2[-1,0:4],parameters,model)
#ax = plt.gca(projection='3d')
#ax.plot(x[:,0],x[:,1],x[:,2],'-',color='r',label='$\Delta E$ = 1.0')
#ax.plot(x[:,0],x[:,1],-x[:,2],'-',color='r')
#ax.scatter(x[0,0],x[0,1],x[0,2],s=20,marker='*')
#ax.scatter(x[0,0],x[0,1],-x[0,2],s=20,marker='o')
#ax.plot(x[:,0], x[:,1], zs=0, zdir='z')
#
#f = lambda t,x : turning_point.Ham2dof(model,t,x,parameters) 
#soln = solve_ivp(f, TSPAN, x0po_3[-1,0:4],method='RK45',dense_output=True, events = lambda t,x : turning_point.half_period(t,x,model),rtol=RelTol, atol=AbsTol)
#te = soln.t_events[0]
#tt = [0,te[1]]
#t,x,phi_t1,PHI = turning_point.stateTransitMat(tt,x0po_3[-1,0:4],parameters,model)
#ax = plt.gca(projection='3d')
#ax.plot(x[:,0],x[:,1],x[:,2],'-',color='g',label='$\Delta E$ = 2.0')
#ax.plot(x[:,0],x[:,1],-x[:,2],'-',color='g')
#ax.scatter(x[0,0],x[0,1],x[0,2],s=20,marker='*')
#ax.scatter(x[0,0],x[0,1],-x[0,2],s=20,marker='o')
#ax.plot(x[:,0], x[:,1], zs=0, zdir='z')
#
#
#f = lambda t,x : turning_point.Ham2dof(model,t,x,parameters) 
#soln = solve_ivp(f, TSPAN, x0po_4[-1,0:4],method='RK45',dense_output=True, events = lambda t,x : turning_point.half_period(t,x,model),rtol=RelTol, atol=AbsTol)
#te = soln.t_events[0]
#tt = [0,te[1]]
#t,x,phi_t1,PHI = turning_point.stateTransitMat(tt,x0po_4[-1,0:4],parameters,model)
#ax = plt.gca(projection='3d')
#ax.plot(x[:,0],x[:,1],x[:,2],'-',color='m',label='$\Delta E$ = 4.0')
#ax.plot(x[:,0],x[:,1],-x[:,2],'-',color='m')
#ax.scatter(x[0,0],x[0,1],x[0,2],s=20,marker='*')
#ax.scatter(x[0,0],x[0,1],-x[0,2],s=20,marker='o')
#ax.plot(x[:,0], x[:,1], zs=0, zdir='z')
#
#ax = plt.gca(projection='3d')
#resX = 100
#xVec = np.linspace(-1,2,resX)
#yVec = np.linspace(-2,2,resX)
#xMat, yMat = np.meshgrid(xVec, yVec)
#cset1 = ax.contour(xMat, yMat, turning_point.get_pot_surf_proj(model,xVec, yVec,parameters), [1.1,2,3,5],zdir='z', offset=0,
#                       linewidths = 1.0, cmap=cm.viridis, alpha = 0.8)
#
#ax.scatter(eqPt[0], eqPt[1], s = 200, c = 'r', marker = 'X')
#ax.set_xlabel('$x$', fontsize=axis_fs)
#ax.set_ylabel('$y$', fontsize=axis_fs)
#ax.set_zlabel('$p_x$', fontsize=axis_fs)
##ax.set_title('$\Delta E$ = %1.e,%1.e,%1.e,%1.e,%1.e' %(energyPO_1[-1]-parameters[2],energyPO_2[-1]-parameters[2],energyPO_3[-1]-parameters[2],energyPO_4[-1]-parameters[2],energyPO_5[-1]-parameters[2]) ,fontsize=axis_fs)
#ax.set_xlim(-1.5, 1.5)
#ax.set_ylim(-1.5, 1.5)
#ax.set_zlim(-4, 4)
#legend = ax.legend(loc='upper left')
#
#plt.grid()
#plt.show()
#
#plt.savefig('turningpoint_POfam_deleonberne.pdf',format='pdf',bbox_inches='tight')

