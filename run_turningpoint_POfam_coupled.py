# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 10:02:48 2019

@author: Wenyang Lyu and Shibabrat Naik

Compute unstable peridoic orbits at different energies using turning point method
"""

# For the coupled problem
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import math
from scipy import optimize
import tp_UPOsHam2dof ### import module xxx where xxx is the name of the python file xxx.py 
import matplotlib as mpl
from matplotlib import cm
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'


#% Begin problem specific functions
def init_guess_eqpt_coupled(eqNum, par):
    
    if eqNum == 1:
        x0 = [0, 0]
    if eqNum == 2:
        x0 = [np.sqrt(par[3]-par[6]/par[4]),0] 
    elif eqNum == 3:
        x0 = [-np.sqrt(par[3]-par[6]/par[4]),0] 
    
    return x0

def grad_pot_coupled(x, par):
    
    dVdx = (-par[3]+par[6])*x[0]+par[4]*(x[0])**3-par[6]*x[1]
    dVdy = (par[5]+par[6])*x[1]-par[6]*x[0]
            
    F = [-dVdx, -dVdy]
    
    return F

def pot_energy_coupled(x, y, par):
    return -0.5*par[3]*x**2+0.25*par[4]*x**4 +0.5*par[5]*y**2+0.5*par[6]*(x-y)**2


def get_coord_coupled(x, y, E, par):
    """ 
    this function returns the initial position of x/y-coordinate on the potential energy 
    surface(PES) for a specific energy E.
    """
#    if model == 'uncoupled':
#        return -0.5*par[3]*x**2+0.25*par[4]*x**4 +0.5*par[5]*y**2-V
#    elif model =='coupled':
    return -0.5*par[3]*x**2+0.25*par[4]*x**4 +0.5*par[5]*y**2+0.5*par[6]*(x-y)**2 - E
#    elif model== 'deleonberne':
#        return par[3]*( 1 - math.e**(-par[4]*x) )**2 + 4*y**2*(y**2 - 1)*math.e**(-par[5]*par[4]*x) + par[2]-V
#    else:
#        print("The model you are chosen does not exist, enter the function for finding coordinates on the PES for given x or y and V")


def varEqns_coupled(t,PHI,par):
    
    """
    PHIdot = varEqns_coupled(t,PHI) 
    
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
    dVdx = (-par[3]+par[6])*x+par[4]*x**3-par[6]*y
    dVdy = (par[5]+par[6])*y-par[6]*x

    # The second order derivative of the potential energy. 
    d2Vdx2 = -par[3]+par[6]+par[4]*3*x**2
        
    d2Vdy2 = par[5]+par[6]

    d2Vdydx = -par[6]

    
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


def ham2dof_coupled(t, x, par):
    """
    Hamilton's equations of motion
    """
    
    xDot = np.zeros(4)
    
    dVdx = (-par[3]+par[6])*x[0]+par[4]*(x[0])**3-par[6]*x[1]
    dVdy = (par[5]+par[6])*x[1]-par[6]*x[0]
        
    xDot[0] = x[2]/par[0]
    xDot[1] = x[3]/par[1]
    xDot[2] = -dVdx 
    xDot[3] = -dVdy
    
    return list(xDot)    


def half_period_coupled(t, x, par):
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


def guess_coords_coupled(guess1, guess2, i, n, e, get_coord_model, par):
    """
    Returns x and y (configuration space) coordinates as guess for the next iteration of the 
    turning point based on confifuration difference method
    """
    
    
    h = (guess2[0] - guess1[0])*i/n
    print("h is ",h)
    xguess = guess1[0]+h
    f = lambda y: get_coord_model(xguess,y,e,par)
    yanalytic = math.sqrt(2/(par[1]+par[6]))*(-math.sqrt( e +0.5*par[3]* xguess**2 - \
                         0.25*par[4]*xguess**4 -0.5*par[6]* xguess**2 + \
                         (par[6]*xguess)**2/(2*(par[1] +par[6]) )) + 
                        par[6]/(math.sqrt(2*(par[1]+par[6])) )*xguess ) #coupled
    yguess = optimize.newton(f,yanalytic) 
    
    return xguess, yguess
    
def plot_iter_orbit_coupled(x, ax, e, par):
    
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


#%% Setting up parameters and global variables

N = 4          # dimension of phase space
omega=1.00
EPSILON_S = 0.0 #Energy of the saddle
alpha = 1.00
beta = 1.00
epsilon= 1e-1
parameters = np.array([1,omega, EPSILON_S, alpha, beta,omega,epsilon]);
eqNum = 1;  
model = 'coupled'
#eqPt = tp_UPOsHam2dof.get_eq_pts(eqNum,model, parameters)
eqPt = tp_UPOsHam2dof.get_eq_pts(eqNum, init_guess_eqpt_coupled, \
                                   grad_pot_coupled, parameters)

#%%
#deltaE_vals = [0.01, 0.1, 1.00, 2.0, 4.0]
#xLeft = [0.0,-0.05,0.01,0.01,0.18]
#xRight = [0.05,0.10,0.18,0.18,0.18]
#linecolor = ['b','r','g','m','c']
deltaE_vals = [0.1, 1.00]
xLeft = [0.0,0.01]
xRight = [0.05,0.11]
linecolor = ['b','r']

"""
e is the total energy, 
n is the number of intervals we want to divide, 
n_turn is the nth turning point we want to choose.
"""
for i in range(len(deltaE_vals)):
    
    e = deltaE_vals[i]
    n = 12
    n_turn = 1
    deltaE = e - parameters[2]
    
    """
    Trial initial Condition s.t. one initial condition is on the LHS of the UPO and the 
    other one is on the RHS of the UPO
    """
    x = xLeft[i]
    f2 = lambda y: get_coord_coupled(x, y, e, parameters)
    yanalytic = math.sqrt(2/(parameters[1]+parameters[6]))*(-math.sqrt( e + \
                         0.5*parameters[3]* x**2- 0.25*parameters[4]*x**4 - 0.5*parameters[6]*x**2 + \
                         (parameters[6]*x)**2/(2*(parameters[1] + parameters[6]) )) + \
                            parameters[6]/(math.sqrt(2*(parameters[1]+parameters[6])) )*x ) #coupled
    state0_2 = [x,optimize.newton(f2,yanalytic),0.0,0.0]
    
    x = xRight[i]
    f3 = lambda y: get_coord_coupled(x, y, e, parameters)
    yanalytic = math.sqrt(2/(parameters[1]+parameters[6]))*(-math.sqrt( e + \
                         0.5*parameters[3]* x**2- 0.25*parameters[4]*x**4 -0.5*parameters[6]*x**2 + \
                         (parameters[6]*x)**2/(2*(parameters[1] + parameters[6]) )) + \
                            parameters[6]/(math.sqrt(2*(parameters[1]+parameters[6])) )*x ) #coupled
    state0_3 = [x, optimize.newton(f3,yanalytic),0.0,0.0]
    
    
    po_fam_file = open("x0_turningpoint_deltaE%s_coupled.txt" %(deltaE),'a+')
    [x0po_1, T_1,energyPO_1] = tp_UPOsHam2dof.turningPoint(model, state0_2, state0_3, \
                                                            get_coord_coupled, \
                                                            guess_coords_coupled, \
                                                            ham2dof_coupled, \
                                                            half_period_coupled, \
                                                            varEqns_coupled, \
                                                            pot_energy_coupled, \
                                                            plot_iter_orbit_coupled, \
                                                            parameters, \
                                                            e, n, n_turn, po_fam_file) 
    po_fam_file.close()



#[x0po_1, T_1,energyPO_1] = tp_UPOsHam2dof.turningPoint(model,state0_2,state0_3 ,parameters,e,n,n_turn,po_fam_file) ; 

#%% Load periodic orbit data from ascii files
    
x0po = np.zeros((4,len(deltaE_vals))) #each column is a different initial condition

for i in range(len(deltaE_vals)):
    deltaE = deltaE_vals[i]

    po_fam_file = open("x0_tpcd_deltaE%s_coupled.txt" %(deltaE),'a+')
    print('Loading the periodic orbit family from data file',po_fam_file.name,'\n') 
    x0podata = np.loadtxt(po_fam_file.name)
    po_fam_file.close()
    x0po[:,i] = x0podata[-1,0:4] 

#%%
#e=0.1
#n=12
#n_turn = 1
#deltaE = e-parameters[2]
#"""Trial initial Condition s.t. one initial condition is on the LHS of the UPO and the other one is on the RHS of the UPO"""
#x=-0.05
#f2 = lambda y: tp_UPOsHam2dof.get_coordinate(model,0.0,y,e,parameters)
#yanalytic = math.sqrt(2/(parameters[1]+parameters[6]))*(-math.sqrt( e +0.5*parameters[3]* x**2- 0.25*parameters[4]*x**4 -0.5*parameters[6]* x**2 + (parameters[6]*x)**2/(2*(parameters[1] +parameters[6]) )) +parameters[6]/(math.sqrt(2*(parameters[1]+parameters[6])) )*x ) #coupled
#state0_2 = [0.0,optimize.newton(f2,yanalytic),0.0,0.0]
#
#x=0.05
#f3 = lambda y: tp_UPOsHam2dof.get_coordinate(model,x,y,e,parameters)
#yanalytic = math.sqrt(2/(parameters[1]+parameters[6]))*(-math.sqrt( e +0.5*parameters[3]* x**2- 0.25*parameters[4]*x**4 -0.5*parameters[6]* x**2 + (parameters[6]*x)**2/(2*(parameters[1] +parameters[6]) )) +parameters[6]/(math.sqrt(2*(parameters[1]+parameters[6])) )*x ) #coupled
#state0_3 = [x, optimize.newton(f3,yanalytic),0.0,0.0]
#
#
#po_fam_file = open("1111x0_turningpoint_deltaE%s_coupled.txt" %(deltaE),'a+')
#[x0po_2, T_2,energyPO_2] = tp_UPOsHam2dof.turningPoint(model,state0_2,state0_3 ,parameters,e,n,n_turn,po_fam_file) ; 
#po_fam_file.close()
#
##%%
#e=1.0
#n=12
#n_turn = 1
#deltaE = e-parameters[2]
#"""Trial initial Condition s.t. one initial condition is on the LHS of the UPO and the other one is on the RHS of the UPO"""
#x=0.01
#f2 = lambda y: tp_UPOsHam2dof.get_coordinate(model,0.0,y,e,parameters)
#yanalytic = math.sqrt(2/(parameters[1]+parameters[6]))*(-math.sqrt( e +0.5*parameters[3]* x**2- 0.25*parameters[4]*x**4 -0.5*parameters[6]* x**2 + (parameters[6]*x)**2/(2*(parameters[1] +parameters[6]) )) +parameters[6]/(math.sqrt(2*(parameters[1]+parameters[6])) )*x ) #coupled
#state0_2 = [0.0,optimize.newton(f2,yanalytic),0.0,0.0]
#
#x=0.11
#f3 = lambda y: tp_UPOsHam2dof.get_coordinate(model,x,y,e,parameters)
#yanalytic = math.sqrt(2/(parameters[1]+parameters[6]))*(-math.sqrt( e +0.5*parameters[3]* x**2- 0.25*parameters[4]*x**4 -0.5*parameters[6]* x**2 + (parameters[6]*x)**2/(2*(parameters[1] +parameters[6]) )) +parameters[6]/(math.sqrt(2*(parameters[1]+parameters[6])) )*x ) #coupled
#state0_3 = [x, optimize.newton(f3,yanalytic),0.0,0.0]
#
#
#po_fam_file = open("1111x0_turningpoint_deltaE%s_coupled.txt" %(deltaE),'a+')
#[x0po_3, T_3,energyPO_3] = tp_UPOsHam2dof.turningPoint(model,state0_2,state0_3 ,parameters,e,n,n_turn,po_fam_file) ; 
#po_fam_file.close()
#
##%%
#e=2.0
#n=12
#n_turn = 1
#deltaE = e-parameters[2]
#"""Trial initial Condition s.t. one initial condition is on the LHS of the UPO and the other one is on the RHS of the UPO"""
#x=0.05
#f2 = lambda y: tp_UPOsHam2dof.get_coordinate(model,0.0,y,e,parameters)
#yanalytic = math.sqrt(2/(parameters[1]+parameters[6]))*(-math.sqrt( e +0.5*parameters[3]* x**2- 0.25*parameters[4]*x**4 -0.5*parameters[6]* x**2 + (parameters[6]*x)**2/(2*(parameters[1] +parameters[6]) )) +parameters[6]/(math.sqrt(2*(parameters[1]+parameters[6])) )*x ) #coupled
#state0_2 = [0.0,optimize.newton(f2,yanalytic),0.0,0.0]
#
#x=0.15
#f3 = lambda y: tp_UPOsHam2dof.get_coordinate(model,x,y,e,parameters)
#yanalytic = math.sqrt(2/(parameters[1]+parameters[6]))*(-math.sqrt( e +0.5*parameters[3]* x**2- 0.25*parameters[4]*x**4 -0.5*parameters[6]* x**2 + (parameters[6]*x)**2/(2*(parameters[1] +parameters[6]) )) +parameters[6]/(math.sqrt(2*(parameters[1]+parameters[6])) )*x ) #coupled
#state0_3 = [x, optimize.newton(f3,yanalytic),0.0,0.0]
#
#
#po_fam_file = open("1111x0_turningpoint_deltaE%s_coupled.txt" %(deltaE),'a+')
#[x0po_4, T_4,energyPO_4] = tp_UPOsHam2dof.turningPoint(model,state0_2,state0_3 ,parameters,e,n,n_turn,po_fam_file) ; 
#po_fam_file.close()
#
##%%
#e=4.0
#n=12
#n_turn = 1
#deltaE = e-parameters[2]
#"""Trial initial Condition s.t. one initial condition is on the LHS of the UPO and the other one is on the RHS of the UPO"""
#x=0.08
#f2 = lambda y: tp_UPOsHam2dof.get_coordinate(model,0.0,y,e,parameters)
#yanalytic = math.sqrt(2/(parameters[1]+parameters[6]))*(-math.sqrt( e +0.5*parameters[3]* x**2- 0.25*parameters[4]*x**4 -0.5*parameters[6]* x**2 + (parameters[6]*x)**2/(2*(parameters[1] +parameters[6]) )) +parameters[6]/(math.sqrt(2*(parameters[1]+parameters[6])) )*x ) #coupled
#state0_2 = [0.0,optimize.newton(f2,yanalytic),0.0,0.0]
#
#x=0.18
#f3 = lambda y: tp_UPOsHam2dof.get_coordinate(model,x,y,e,parameters)
#yanalytic = math.sqrt(2/(parameters[1]+parameters[6]))*(-math.sqrt( e +0.5*parameters[3]* x**2- 0.25*parameters[4]*x**4 -0.5*parameters[6]* x**2 + (parameters[6]*x)**2/(2*(parameters[1] +parameters[6]) )) +parameters[6]/(math.sqrt(2*(parameters[1]+parameters[6])) )*x ) #coupled
#state0_3 = [x, optimize.newton(f3,yanalytic),0.0,0.0]
#
#
#po_fam_file = open("1111x0_turningpoint_deltaE%s_coupled.txt" %(deltaE),'a+')
#[x0po_5, T_5,energyPO_5] = tp_UPOsHam2dof.turningPoint(model,state0_2,state0_3 ,parameters,e,n,n_turn,po_fam_file) ; 
#po_fam_file.close()


#%% Load Data
#deltaE = 0.010
#po_fam_file = open("1111x0_turningpoint_deltaE%s_coupled.txt" %(deltaE),'a+');
#print('Loading the periodic orbit family from data file',po_fam_file.name,'\n'); 
#x0podata = np.loadtxt(po_fam_file.name)
#po_fam_file.close()
#x0po_1 = x0podata
#
#deltaE = 0.10
#po_fam_file = open("1111x0_turningpoint_deltaE%s_coupled.txt" %(deltaE),'a+');
#print('Loading the periodic orbit family from data file',po_fam_file.name,'\n'); 
#x0podata = np.loadtxt(po_fam_file.name)
#po_fam_file.close()
#x0po_2 = x0podata
#
#deltaE = 1.0
#po_fam_file = open("1111x0_turningpoint_deltaE%s_coupled.txt" %(deltaE),'a+');
#print('Loading the periodic orbit family from data file',po_fam_file.name,'\n'); 
#x0podata = np.loadtxt(po_fam_file.name)
#po_fam_file.close()
#x0po_3 = x0podata
#
#deltaE = 2.0
#po_fam_file = open("1111x0_turningpoint_deltaE%s_coupled.txt" %(deltaE),'a+');
#print('Loading the periodic orbit family from data file',po_fam_file.name,'\n'); 
#x0podata = np.loadtxt(po_fam_file.name)
#po_fam_file.close()
#x0po_4 = x0podata
#
#deltaE = 4.0
#po_fam_file = open("1111x0_turningpoint_deltaE%s_coupled.txt" %(deltaE),'a+');
#print('Loading the periodic orbit family from data file',po_fam_file.name,'\n'); 
#x0podata = np.loadtxt(po_fam_file.name)
#po_fam_file.close()
#x0po_5 = x0podata

#%% Plotting the Family

TSPAN = [0,30]
plt.close('all')
axis_fs = 15
RelTol = 3.e-10
AbsTol = 1.e-10

f = lambda t,x : ham2dof_coupled(t,x,parameters) 

ax = plt.gca(projection='3d')

for i in range(len(deltaE_vals)):
    
    soln = solve_ivp(f, TSPAN, x0po[:,i], method='RK45', dense_output=True, \
                     events = lambda t,x : half_period_coupled(t,x,parameters), \
                     rtol=RelTol, atol=AbsTol)
    te = soln.t_events[0]
    tt = [0,te[2]]
    t,x,phi_t1,PHI = tp_UPOsHam2dof.stateTransitMat(tt, x0po[:,i], parameters, varEqns_coupled)
    
    
    ax.plot(x[:,0],x[:,1],x[:,3],'-', color=linecolor[i], \
            label='$\Delta E$ = %.2f'%(deltaE_vals[i]))
    ax.scatter(x[0,0],x[0,1],x[0,3], s=20, marker='*')
    ax.plot(x[:,0], x[:,1], zs=0, zdir='z')


resX = 100
xVec = np.linspace(-4,4,resX)
yVec = np.linspace(-4,4,resX)
xMat, yMat = np.meshgrid(xVec, yVec)
cset1 = ax.contour(xMat, yMat, 
                   tp_UPOsHam2dof.get_pot_surf_proj(xVec, yVec, \
                                                      pot_energy_coupled, parameters), \
                                                      [0.01,0.1,1,2,4], zdir='z', offset=0, \
                                                      linewidths = 1.0, cmap=cm.viridis, \
                                                      alpha = 0.8)

ax.scatter(eqPt[0], eqPt[1], s = 200, c = 'r', marker = 'X')
ax.set_xlabel('$x$', fontsize=axis_fs)
ax.set_ylabel('$y$', fontsize=axis_fs)
ax.set_zlabel('$p_y$', fontsize=axis_fs)
#ax.set_title('$\Delta E$ = %1.e,%1.e,%1.e,%1.e,%1.e' %(energyPO_1[-1],energyPO_2[-1],energyPO_3[-1],energyPO_4[-1],energyPO_5[-1]) ,fontsize=axis_fs)
legend = ax.legend(loc='upper left')
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_zlim(-4, 4)

plt.grid()
plt.show()

plt.savefig('tpcd_POfam_coupled.pdf', format='pdf', bbox_inches='tight')



#%% Plotting the Family

#TSPAN = [0,30]
#plt.close('all')
#axis_fs = 15
#RelTol = 3.e-10
#AbsTol = 1.e-10
#f = lambda t,x : tp_UPOsHam2dof.Ham2dof(model,t,x,parameters) 
#soln = solve_ivp(f, TSPAN, x0po_1[-1,0:4],method='RK45',dense_output=True, events = lambda t,x : tp_UPOsHam2dof.half_period(t,x,model),rtol=RelTol, atol=AbsTol)
#te = soln.t_events[0]
#tt = [0,te[1]]
#t,x,phi_t1,PHI = tp_UPOsHam2dof.stateTransitMat(tt,x0po_1[-1,0:4],parameters,model)
#ax = plt.gca(projection='3d')
#ax.plot(x[:,0],x[:,1],x[:,3],'-',color='b',label='$\Delta E$ = 0.01')
#ax.plot(x[:,0],x[:,1],-x[:,3],'-',color='b')
#ax.scatter(x[0,0],x[0,1],x[0,3],s=20,marker='*');
#ax.scatter(x[0,0],x[0,1],-x[0,3],s=20,marker='o');
#ax.plot(x[:,0], x[:,1], zs=0, zdir='z')
#
#f = lambda t,x : tp_UPOsHam2dof.Ham2dof(model,t,x,parameters) 
#soln = solve_ivp(f, TSPAN, x0po_2[-1,0:4],method='RK45',dense_output=True, events = lambda t,x : tp_UPOsHam2dof.half_period(t,x,model),rtol=RelTol, atol=AbsTol)
#te = soln.t_events[0]
#tt = [0,te[1]]
#t,x,phi_t1,PHI = tp_UPOsHam2dof.stateTransitMat(tt,x0po_2[-1,0:4],parameters,model)
#ax = plt.gca(projection='3d')
#ax.plot(x[:,0],x[:,1],x[:,3],'-',color='r',label='$\Delta E$ = 0.1')
#ax.plot(x[:,0],x[:,1],-x[:,3],'-',color='r')
#ax.scatter(x[0,0],x[0,1],x[0,3],s=20,marker='*');
#ax.scatter(x[0,0],x[0,1],-x[0,3],s=20,marker='o');
#ax.plot(x[:,0], x[:,1], zs=0, zdir='z')
#
#f = lambda t,x : tp_UPOsHam2dof.Ham2dof(model,t,x,parameters) 
#soln = solve_ivp(f, TSPAN, x0po_3[-1,0:4],method='RK45',dense_output=True, events = lambda t,x : tp_UPOsHam2dof.half_period(t,x,model),rtol=RelTol, atol=AbsTol)
#te = soln.t_events[0]
#tt = [0,te[1]]
#t,x,phi_t1,PHI = tp_UPOsHam2dof.stateTransitMat(tt,x0po_3[-1,0:4],parameters,model)
#ax = plt.gca(projection='3d')
#ax.plot(x[:,0],x[:,1],x[:,3],'-',color='g',label='$\Delta E$ = 1.0')
#ax.plot(x[:,0],x[:,1],-x[:,3],'-',color='g')
#ax.scatter(x[0,0],x[0,1],x[0,3],s=20,marker='*');
#ax.scatter(x[0,0],x[0,1],-x[0,3],s=20,marker='o');
#ax.plot(x[:,0], x[:,1], zs=0, zdir='z')
#
#
#f = lambda t,x : tp_UPOsHam2dof.Ham2dof(model,t,x,parameters) 
#soln = solve_ivp(f, TSPAN, x0po_4[-1,0:4],method='RK45',dense_output=True, events = lambda t,x : tp_UPOsHam2dof.half_period(t,x,model),rtol=RelTol, atol=AbsTol)
#te = soln.t_events[0]
#tt = [0,te[1]]
#t,x,phi_t1,PHI = tp_UPOsHam2dof.stateTransitMat(tt,x0po_4[-1,0:4],parameters,model)
#ax = plt.gca(projection='3d')
#ax.plot(x[:,0],x[:,1],x[:,3],'-',color='m',label='$\Delta E$ = 2.0')
#ax.plot(x[:,0],x[:,1],-x[:,3],'-',color='m')
#ax.scatter(x[0,0],x[0,1],x[0,3],s=20,marker='*');
#ax.scatter(x[0,0],x[0,1],-x[0,3],s=20,marker='o');
#ax.plot(x[:,0], x[:,1], zs=0, zdir='z')
#
#
#f = lambda t,x : tp_UPOsHam2dof.Ham2dof(model,t,x,parameters) 
#soln = solve_ivp(f, TSPAN, x0po_5[-1,0:4],method='RK45',dense_output=True, events = lambda t,x : tp_UPOsHam2dof.half_period(t,x,model),rtol=RelTol, atol=AbsTol)
#te = soln.t_events[0]
#tt = [0,te[1]]
#t,x,phi_t1,PHI = tp_UPOsHam2dof.stateTransitMat(tt,x0po_5[-1,0:4],parameters,model)
#ax = plt.gca(projection='3d')
#ax.plot(x[:,0],x[:,1],x[:,3],'-',color='c',label='$\Delta E$ = 4.0')
#ax.plot(x[:,0],x[:,1],-x[:,3],'-',color='c')
#ax.scatter(x[0,0],x[0,1],x[0,3],s=20,marker='*');
#ax.scatter(x[0,0],x[0,1],-x[0,3],s=20,marker='o');
#ax.plot(x[:,0], x[:,1], zs=0, zdir='z')
#
#
#ax = plt.gca(projection='3d')
#resX = 100
#xVec = np.linspace(-4,4,resX)
#yVec = np.linspace(-4,4,resX)
#xMat, yMat = np.meshgrid(xVec, yVec)
#cset1 = ax.contour(xMat, yMat, tp_UPOsHam2dof.get_pot_surf_proj(model,xVec, yVec,parameters), [0.01,0.1,1,2,4],zdir='z', offset=0,
#                       linewidths = 1.0, cmap=cm.viridis, alpha = 0.8)
#
#ax.scatter(eqPt[0], eqPt[1], s = 200, c = 'r', marker = 'X')
#ax.set_xlabel('$x$', fontsize=axis_fs)
#ax.set_ylabel('$y$', fontsize=axis_fs)
#ax.set_zlabel('$p_y$', fontsize=axis_fs)
##ax.set_title('$\Delta E$ = %1.e,%1.e,%1.e,%1.e,%1.e' %(energyPO_1[-1],energyPO_2[-1],energyPO_3[-1],energyPO_4[-1],energyPO_5[-1]) ,fontsize=axis_fs)
#legend = ax.legend(loc='best')
#ax.set_xlim(-4, 4)
#ax.set_ylim(-4, 4)
#ax.set_zlim(-4, 4)
#
#plt.grid()
#plt.show()
#
#plt.savefig('turningpoint_POfam_coupled.pdf',format='pdf',bbox_inches='tight')



