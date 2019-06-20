import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint,quad,trapz
import math
from IPython.display import Image # for Notebook
from IPython.core.display import HTML
from mpl_toolkits.mplot3d import Axes3D

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
