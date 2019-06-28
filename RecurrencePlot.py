import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint,quad,trapz
import math
from IPython.display import Image # for Notebook
from IPython.core.display import HTML
from mpl_toolkits.mplot3d import Axes3D

''' Recurrence Plot of the Lorenz Attractor'''
sigma = 16.0
rho = 45.0
beta = 4.0
state0 = [10.0, -4.32395, 55.4605]
t = np.arange(0.0, 2.0, 0.001)

def Lorenz(fun,t):
    x,y,z=fun
    derivative=[sigma * (y - x), x * (rho - z) - y, x * y - beta * z]
    return derivative
    

states = odeint(Lorenz, state0, t)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(states[:,0], states[:,1], states[:,2])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.xlim(-40,40)
plt.ylim(-40,40)
plt.show()

t1=np.linspace(0,20,0.01)
t2=np.linspace(0,20,0.01)
T1,T2=np.meshgrid(t,t)
Z=np.zeros((2000,2000))
delta_h = 2
delta_l = 0
for i in range(1999):
    for j in range(1999):
            Z[i,j]=math.sqrt((states[:,0][i]-states[:,0][j])**2 +(states[:,1][i]-states[:,1][j])**2 +(states[:,2][i]-states[:,2][j])**2)
plt.figure(figsize=(12,12))
plt.pcolormesh(T1,T2,Z)
plt.xlabel('time')
plt.ylabel('time')
plt.show()

''' solution of the Variational Equation'''
states = np.zeros((1000,3,3)) 
state0 = [10.0, -2.34199, 53.8658,1.0,0,0,0,1.0,0,0,0,1.0]
t = np.arange(0, 1, 0.001)


def Variational(fun,t):
    x,y,z,xx,xy,xz,yx,yy,yz,zx,zy,zz=fun
    derivative=[sigma * (y - x), x * (rho - z) - y, x * y - beta * z, -sigma*xx+sigma*yx, -sigma*xy+sigma*yy,-sigma*xz+sigma*yz, (rho-z)*xx-yx-x*zx,(rho-z)*xy-yy-x*zy,(rho-z)*xz-yz-x*zz, y*xx+x*yx-beta*zx, y*xy+x*yy-beta*zy,y*xz+x*yz-beta*zz]
    return derivative


states = odeint(Variational, state0, t)

Phistates = states[:,3:12]

'''Period of the periodic orbit'''
D=np.zeros((1000))

''' the corresponding time point of the end point of the period'''
for i in range(1000):
    D[i]=math.sqrt((states[0,0]-states[:,0][i])**2+ (states[0,1]-states[:,1][i])**2 + (states[0,2]-states[:,2][i])**2)


print("period T is ",np.argmin(D[1:999])*0.001)

PhistatesT = Phistates[np.argmin(D[1:999]),:]

print("Solution of the Variational equation at Phi_T(x*) is ", PhistatesT )

PhistatesTMatrix = np.zeros((3,3))
PhistatesTMatrix[0,0] = PhistatesT[0]
PhistatesTMatrix[0,1] = PhistatesT[1]
PhistatesTMatrix[0,2] = PhistatesT[2]
PhistatesTMatrix[1,0] = PhistatesT[3]
PhistatesTMatrix[1,1] = PhistatesT[4]
PhistatesTMatrix[1,2] = PhistatesT[5]
PhistatesTMatrix[2,0] = PhistatesT[6]
PhistatesTMatrix[2,1] = PhistatesT[7]
PhistatesTMatrix[2,2] = PhistatesT[8]
'''eigenvalues and eiqenvectors of the matrix'''
print("eigenvalues, eigenvectors are ", LA.eig(np.array(PhistatesTMatrix) )   )
