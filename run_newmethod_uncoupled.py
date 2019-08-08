# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 10:19:37 2019

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
import uncoupled_newmethod ### import module xxx where xxx is the name of the python file xxx.py 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl

from pylab import rcParams
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
#%matplotlib

#%%
alpha = 1.0
beta = 1.0
omega = 1.0
EPSILON_S = 0.0 #Energy of the saddle
parameters = np.array([1,omega, EPSILON_S, alpha, beta,omega]);
"""Initial Condition for the true periodic orbit
H = T + V where T is the kinetic energy and V is the potential energy. In our example, 
the potential energy V = -0.5*alpha*x**2+0.25*beta*x**4+0.5*omega*y**2.
If we fix x, then y = +- math.sqrt((V +0.5*alpha*x**2-0.25*beta*x**4)/(0.5*omega) ) so that
the point starts at the potential energy surface V.

"""
e=1.0
"""Trial initial Condition s.t. one initial condition is on the LHS of the UPO and the other one is on the RHS of the UPO"""
state0_2 = [-0.1 , -math.sqrt(2*e+0.1**2-0.5*0.1**4),0.0,0.0]
state0_3 = [0.1 , -math.sqrt(2*e+0.1**2-0.5*0.1**4),0.0,0.0]
n=4
n_turn=1
po_fam_file = open("1111x0_newmethod_uncoupled.txt",'a+')
[x0po, T,energyPO] = uncoupled_newmethod.newmethod_uncoupled(state0_2,state0_3 ,parameters,e,n,n_turn,po_fam_file) ; 

po_fam_file.close()