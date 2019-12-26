Unstable Periodic Orbits in Hamiltonian systems
===============================================

Theory 
------

Computation of unstable periodic orbits in two degrees of freedom Hamiltonian systems arises in studying transition dynamics in physical sciences (for example chemical reactions, celestial mechanics) and engineering (for example, ship dynamics and capsize, structural mechanics) [Parker1989]_, [Wiggins2003]_

A two degree-of-freedom Hamiltonian system of the form kinetic plus potential energy is represented by

.. math::
    \begin{equation}
    H(x,y,p_x,p_y)  = T(p_x,p_y) + V(x,y)
    \end{equation}


where :math:`x, y` are configuration space coordinates, :math:`p_x, p_y` are corresponding momenta, :math:`V(x,y)` is the potential energy, and :math:`T(p_{x},p_{y})` is the kinetic energy. The unstable periodic orbits exist in the bottleneck of the equipotential contour given by :math:`V(x,y) = E` where :math:`E` is the total energy. For the Hamiltonian system of the form kinetic plus potential energy, the unstable periodic orbit projects as a line on the configuration space :math:`(x,y)` [Wiggins2016]_. The objective is to compute this orbit which exists for energies above the energy of the index-1 saddle equilibrium point located in the bottleneck.


Available methods 
-----------------

This section gives a broad overview of the methods as a reference for implementing new methods and as how to guide to solve new systems. The methods are implemented as modules and are part of the internal code that do not require modification. More details can be found in the papers listed below:

- Differential correction [Koon2000]_, [Koon2011]_, [Naik2017]_, [Ross2018]_, [Naik2019finding]_
- Turning point [Pollak1980]_, [DeleonBerne1981]_
- Turning point based on configuration difference: this is a modification of the turning point method and does not rely on dot product computation.


Example Hamiltonian systems 
---------------------------

In the sections below, we briefly describe the Hamiltonian systems with potential wells connected by a bottleneck and these are used to demonstrate the methods mentioned in Introduction.


- De Leon-Berne Hamiltonian. [DeleonBerne1981]_, [DeLeonMarston1989]_, [Marston1989]_

- Quartic Hamiltonian: uncoupled and coupled.



.. include:: references.txt
