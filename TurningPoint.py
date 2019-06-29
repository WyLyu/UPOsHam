# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 11:19:08 2019

@author: wl16298
"""

"""
We discuss the Turning Point method for finding periodic orbits.

"""

"""We use a 2-degree-of-freedom Hamiltonian system
H = lamd/2*(px^2-x^2) + omega/2*(py^2+y^2)"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint,quad,trapz
import math
from IPython.display import Image # for Notebook
from IPython.core.display import HTML
from mpl_toolkits.mplot3d import Axes3D
from numpy import linalg as LA

omega = 1.0
lamb  = 1.0
"""Initial Condition for the true periodic orbit"""
state0_1 = [0,0.0 , 0.0,1.0]
"""Trial initial Condition s.t. the dot product of p_perpendicular_1 and p_perpendicular_2 is negative"""
state0_2 = [-0.1,0.0 , -math.sqrt(0.1**2+1),0.0]
state0_3 = [0.1,0.0 , -math.sqrt(0.1**2+1),0.0]
t = np.arange(0.0, 5.0, 0.01)


"""Define the Hamiltonian as a function""" 
def H(fun,t):
    x,px,y,py=fun
    derivative=[lamb*px,lamb*x, omega*py, -omega*y]
    return derivative


states1 = odeint(H, state0_1, t)
states2 = odeint(H, state0_2, t)
states3 = odeint(H, state0_3, t)
fig = plt.figure()
ax = fig.gca(projection='3d')
'ax.plot(states1[:,0], states1[:,2], states1[:,3])'
ax.plot(states2[:,0], states2[:,2], states2[:,3])
ax.plot(states3[:,0], states3[:,2], states3[:,3])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_xlim(-10, 10)
ax.set_ylim(-4, 4)
ax.set_zlim(-4, 4)
plt.show()

def potential(x,y):
    return -0.5*lamb*x**2+0.5*omega*y**2

def H_Contour(V):
    x=np.linspace(-5.0,5.0,200)
    y=np.linspace(-5.0,5.0,200)
    X,Y=np.meshgrid(x,y)
    Z=np.zeros((200,200))
    for i in range(200):
        for j in range(200):
            Z[i,j]=potential(X[i,j],Y[i,j])
    plt.figure(figsize=(12,12))
    plt.contour(X,Y,Z,V)
    ax = plt.axes()
    ax.plot(states1[:,0], states1[:,2])
    ax.plot(states2[:,0], states2[:,2])
    ax.plot(states3[:,0], states3[:,2])
    plt.xlim(-15,15)
    plt.ylim(-2,2)
    'plt.show()'
H_Contour([0.5])

"""
momentum(px,py) of 2 selected trajectories starting on the potential energy surface"""
p_1 = np.zeros((500,2))
p_2 = np.zeros((500,2))
p_1[:,0] = states2[:,1]
p_1[:,1] = states2[:,3]
p_2[:,0] = states3[:,1]
p_2[:,1] = states3[:,3]

for i in range(1,499):
    if np.sign(p_1[i+1,1]) != np.sign(p_1[i,1]) :
        p_perpendicular_1 = np.dot(p_1[i-2,:],p_1[i-2,:])*p_1[i-1,:] - np.dot(p_1[i-1,:],p_1[i-2,:])*p_1[i-2,:]
        p_perpendicular_2 = np.dot(p_2[i-2,:],p_1[i-2,:])*p_2[i-1,:] - np.dot(p_2[i-1,:],p_1[i-2,:])*p_2[i-2,:]
        print(i)
        print(states2[0,:],states3[0,:])
        print(np.dot(p_perpendicular_1,p_perpendicular_2))


"""n is the number of intervals, h is the step size, 
p_1 & p_2 are the two points such that the periodic orbit exists between thses two points
we creat a list of initial conditions between p_1&p_2 in order to find a new p_2 s.t.
the 'dot product' between p_1 and the new p_2 is < 0.
The new p_2 is the initial condition for the perioic orbit
"""
n = 11
h = (state0_3[0]-state0_2[0])/n
p0Matrix = np.zeros((n,4))
stateorbit0_3 = [0,0,0,0]
for i in range(n):
    """consider the change in x coordinate"""
    state0_3 = [state0_2[0]+i*h,0.0 , -math.sqrt((state0_2[0]+i*h)**2+1),0.0]
    states3 = odeint(H, state0_3, t)
    p_2[:,0] = states3[:,1]
    p_2[:,1] = states3[:,3]
    for j in range(1,499):
        if np.sign(p_1[j+1,1]) != np.sign(p_1[j,1]) :
            p_perpendicular_1 = np.dot(p_1[j-2,:],p_1[j-2,:])*p_1[j-1,:] - np.dot(p_1[j-1,:],p_1[j-2,:])*p_1[j-2,:]
            p_perpendicular_2 = np.dot(p_2[j-2,:],p_1[j-2,:])*p_2[j-1,:] - np.dot(p_2[j-1,:],p_1[j-2,:])*p_2[j-2,:]
            print(j)
            tp = j
            p0Matrix[i,0] = tp
            print(states2[0,:],states3[0,:])
            p0Matrix[i,2] = states3[0,0]
            p0Matrix[i,3] = states3[0,2] 
            print(np.dot(p_perpendicular_1,p_perpendicular_2))
            p0Matrix[i,1] = np.dot(p_perpendicular_1,p_perpendicular_2)
for i in range(n-1):
    if p0Matrix[i+1,1] < 0 and p0Matrix[i,1] > 0:
        stateorbit0_3[0] = p0Matrix[i,2]
        stateorbit0_3[2] = p0Matrix[i,3]

statesnew0 = odeint(H, stateorbit0_3, t)
ax = plt.axes()
ax.plot(statesnew0[:,0], statesnew0[:,2])
        
    
















    
fig = plt.figure()
ax = plt.axes()
ax.plot(states1[:,0], states1[:,2])
ax.plot(states2[:,0], states2[:,2])
plt.xlim(-10,10)
plt.ylim(-2,2)