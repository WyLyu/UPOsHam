# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 14:59:30 2019

@author: wl16298
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 11:40:41 2019

@author: wl16298
"""

#"""   SCRIPT to compute periodic orbits for the 2 DoF Coupled potential with alpha>0
#--------------------------------------------------------------------------
#   coupled potential energy surface with positive alpha notations:

#   H = 1/2*px^2+omega/2*py^2 -(1/2)*alpha*x^2+1/4*beta*x^4+ omega/2y^2 + 1/2 epsilon*(x-y)^2
#
#           Well (stable, EQNUM = 2)    
#
#               Saddle (EQNUM=1)
#
#           Well (stable, EQNUM = 3)    
#
#--------------------------------------------------------------------------"""


#%% import some useful libraries from python
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
import coupled_diffcorr ### import module xxx where xxx is the name of the python file xxx.py 
from mpl_toolkits.mplot3d import Axes3D

%matplotlib

#%% Setting up parameters and global variables
N = 4          # dimension of phase space
omega=1.00
EPSILON_S = 0.0 #Energy of the saddle
alpha = 1.00
beta = 1.00
epsilon= 1e-2
parameters = np.array([1,omega, EPSILON_S, alpha, beta,omega,epsilon]);
eqNum = 1 
eqPt = coupled_diffcorr.get_eq_pts_coupled(eqNum, parameters)



eSaddle = coupled_diffcorr.get_total_energy_coupled([eqPt[0],eqPt[1],0,0], parameters) #energy of the saddle eq pt
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
deltaE = 0.010
t = time.time()

#  get the initial conditions and periods for a family of periodic orbits


po_fam_file = open("1111x0_tp_fam_eqPt%s_coupled.txt" %eqNum,'a+')
[po_x0Fam,po_tpFam] = coupled_diffcorr.get_POFam_coupled(eqNum, Ax1, Ax2,nFam, po_fam_file, parameters) ; 

poFamRuntime = time.time()-t
x0podata = np.concatenate((po_x0Fam, po_tpFam),axis=1)
po_fam_file.close()

#%%
# begins with a family of periodic orbits and steps until crossing the
# initial condition with target energy 
# fileName = 'x0po_T_energy_case1_L41.txt';
# fileName = 'x0po_T.txt';


po_fam_file = open("1111x0_tp_fam_eqPt%s_coupled.txt" %eqNum ,'a+');
eTarget = eSaddle + deltaE; 
print('Loading the periodic orbit family from data file',po_fam_file.name,'\n'); 
x0podata = np.loadtxt(po_fam_file.name)
po_fam_file.close()


#%%
po_brac_file = open("1111x0po_T_energyPO_eqPt%s_brac%s_coupled.txt" %(eqNum,deltaE),'a+');
t = time.time()
# [x0poTarget,TTarget] = bracket_POEnergy_bp(eTarget, x0podata, po_brac_file);
x0poTarget,TTarget = coupled_diffcorr.poBracketEnergy_coupled(eTarget, x0podata,po_brac_file, parameters);
poTarE_runtime = time.time()-t
model_parameters_file = open("1111model_parameters_eqPt%s_DelE%s_coupled.txt" %(eqNum,deltaE),'a+')
np.savetxt(model_parameters_file.name, parameters,fmt='%1.16e');
model_parameters_file.close()
po_brac_file.close()


#%%
# target specific periodic orbit
# Target PO of specific energy with high precision; does not work for the
# model 

po_target_file = open("1111x0po_T_energyPO_eqPt%s_DelE%s_coupled.txt" %(eqNum,deltaE), 'a+')
                
[x0_PO, T_PO, e_PO] = coupled_diffcorr.poTargetEnergy_coupled(x0poTarget,eTarget,po_target_file,parameters);

po_target_file.close()