Differential correction
=======================

This method is based on small (:math:`\approx 10^{-5}`) corrections to the initial condition of an unstable periodic orbit at a desired total energy. The procedure is started from the linear solutions of the Hamilton's equations at the index-1 saddle equilibrium point and which generates a small amplitude (:math:`\approx 10^{-5}`) periodic orbit. This is then fed into an iterative procedure based on state transition matrix that calculates correction to the initial condition based on error in the terminal coordinates of the periodic orbit. This leads to convergence within 3 steps in the sense of the trajectory returning to the initial condition. Once a small amplitude periodic orbit is obtained, numerical continuation increases the amplitude and correspondingly the total energy of the unstable periodic orbit. Then, a combination of bracketing and bisection method computes the UPO at the desired energy for a specified tolerance. 


   .. automodule:: differential_correction
      :members:



