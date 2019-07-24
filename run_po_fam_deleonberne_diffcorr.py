# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 09:44:12 2019

@author: wl16298
"""

#"""   SCRIPT to compute periodic orbits for the 2 DoF DeLeon-Berne potential
#--------------------------------------------------------------------------
#   DeLeon-Berne potential energy surface notations:
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
import deleonberne_diffcorr ### import module xxx where xxx is the name of the python file xxx.py 
from mpl_toolkits.mplot3d import Axes3D

%matplotlib

#%% Setting up parameters and global variables
N = 4;          # dimension of phase space
MASS_A = 8.0; MASS_B = 8.0; # De Leon, Marston (1989)
EPSILON_S = 1.0;
D_X = 10.0;
ALPHA = 1.00;
LAMBDA = 1.5;
parameters = np.array([MASS_A, MASS_B, EPSILON_S, D_X, LAMBDA, ALPHA]);
eqNum = 1;  

eqPt = deleonberne_diffcorr.get_eq_pts_deleonberne(eqNum, parameters)



eSaddle = deleonberne_diffcorr.get_total_energy_deleonberne([eqPt[0],eqPt[1],0,0], parameters) #energy of the saddle eq pt
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


#%%
po_fam_file = open("1111x0_tp_fam_eqPt%s_deleonberne.txt" %eqNum,'a+')
[po_x0Fam,po_tpFam] = deleonberne_diffcorr.get_POFam_deleonberne(eqNum, Ax1, Ax2,nFam, po_fam_file, parameters) ; 

poFamRuntime = time.time()-t
x0podata = np.concatenate((po_x0Fam, po_tpFam),axis=1)
po_fam_file.close()




#%%
# begins with a family of periodic orbits and steps until crossing the
# initial condition with target energy 
# fileName = 'x0po_T_energy_case1_L41.txt';
# fileName = 'x0po_T.txt';


po_fam_file = open("1111x0_tp_fam_eqPt%s_deleonberne.txt" %eqNum ,'a+');
eTarget = eSaddle + deltaE; 
print('Loading the periodic orbit family from data file',po_fam_file.name,'\n'); 
x0podata = np.loadtxt(po_fam_file.name)
po_fam_file.close()


#%%
po_brac_file = open("1111x0po_T_energyPO_eqPt%s_brac%s_deleonberne.txt" %(eqNum,deltaE),'a+');
t = time.time()
# [x0poTarget,TTarget] = bracket_POEnergy_bp(eTarget, x0podata, po_brac_file);
x0poTarget,TTarget = deleonberne_diffcorr.poBracketEnergy_deleonberne(eTarget, x0podata,po_brac_file, parameters);
poTarE_runtime = time.time()-t
model_parameters_file = open("1111model_parameters_eqPt%s_DelE%s_deleonberne.txt" %(eqNum,deltaE),'a+')
np.savetxt(model_parameters_file.name, parameters,fmt='%1.16e');
model_parameters_file.close()
po_brac_file.close()

#%%
# target specific periodic orbit
# Target PO of specific energy with high precision; does not work for the
# model 

po_target_file = open("1111x0po_T_energyPO_eqPt%s_DelE%s_deleonberne.txt" %(eqNum,deltaE), 'a+')
                
[x0_PO, T_PO, e_PO] = deleonberne_diffcorr.poTargetEnergy_deleonberne(x0poTarget,eTarget,po_target_file,parameters);

po_target_file.close()


