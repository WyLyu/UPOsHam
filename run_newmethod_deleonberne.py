# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 12:19:39 2019

@author: wl16298
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint,quad,trapz,solve_ivp
import math
from IPython.display import Image # for Notebook
from IPython.core.display import HTML
from mpl_toolkits.mplot3d import Axes3D
from numpy import linalg as LA
import scipy.linalg as linalg
from scipy.optimize import fsolve
import time
from functools import partial
import deleonberne_newmethod ### import module xxx where xxx is the name of the python file xxx.py 
from mpl_toolkits.mplot3d import Axes3D
%matplotlib
from scipy import optimize
N = 4;          # dimension of phase space
MASS_A = 8.0; MASS_B = 8.0; # De Leon, Marston (1989)
EPSILON_S = 1.0;
D_X = 10.0;
ALPHA = 1.00;
LAMBDA = 1.5;
parameters = np.array([MASS_A, MASS_B, EPSILON_S, D_X, LAMBDA, ALPHA]);

"""Initial Condition for the true periodic orbit
H = T + V where T is the kinetic energy and V is the potential energy. In our example, 
the potential energy V = -0.5*alpha*x**2+0.25*beta*x**4+0.5*omega*y**2.
If we fix x, then y = +- math.sqrt((V +0.5*alpha*x**2-0.25*beta*x**4)/(0.5*omega) ) so that
the point starts at the potential energy surface V.

"""
# e is the total energy, n is the number of intervals we want to divide, n_turn is the nth turning point we want to choose.
e=2.0
n=4
n_turn = 1
"""Trial initial Condition s.t. one initial condition is on the LHS of the UPO and the other one is on the RHS of the UPO"""
f1 = partial(deleonberne_newmethod.get_x,y=0.06,V=e,par=parameters)
x0_2 = optimize.newton(f1,-0.15)
state0_2 = [x0_2,0.06,0.0,0.0]
f2 = partial(deleonberne_newmethod.get_x,y=-0.05,V=e,par=parameters)
x0_3 = optimize.newton(f2,-0.15)
state0_3 = [x0_3, -0.05,0.0,0.0]

po_fam_file = open("1111x0_newmethod_deleonberne.txt",'a+')
[x0po, T,energyPO] = deleonberne_newmethod.newmethod_deleonberne(state0_3,state0_2 ,parameters,e,n,n_turn,po_fam_file) ; 

po_fam_file.close()