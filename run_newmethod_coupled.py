# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 10:56:07 2019

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
import coupled_newmethod ### import module xxx where xxx is the name of the python file xxx.py 
from mpl_toolkits.mplot3d import Axes3D

%matplotlib
import matplotlib as mpl

from pylab import rcParams
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
#%% Setting up parameters and global variables
N = 4          # dimension of phase space
omega=1.00
EPSILON_S = 0.0 #Energy of the saddle
alpha = 1.00
beta = 1.00
epsilon= 1e-1
parameters = np.array([1,omega, EPSILON_S, alpha, beta,omega,epsilon]);


# e is the total energy, n is the number of intervals we want to divide, n_turn is the nth turning point we want to choose.
e=1.0
n=4
n_turn = 2

"""Trial initial Condition s.t. one initial condition is on the LHS of the UPO and the other one is on the RHS of the UPO"""
state0_2 = [0.01,coupled_newmethod.get_y(0.01,e,parameters),0.0,0.0]
state0_3 = [0.51, coupled_newmethod.get_y(0.51,e,parameters),0.0,0.0]


po_fam_file = open("1111x0_newmethod_coupled.txt",'a+')
[x0po, T,energyPO] = coupled_newmethod.newmethod_coupled(state0_2,state0_3 ,parameters,e,n,n_turn,po_fam_file) ; 


po_fam_file.close()