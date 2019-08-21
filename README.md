-   [Summary](#summary)
-   [Usage](#usage)
-   [Copyright and License](#copyright-and-license)
-   [References](#references)

Summary
-------

UPOsHam is a Python package for computing unstable periodic orbits in two degrees of freedom Hamiltonian systems.

This Python package is a collection of three methods for computing unstable periodic orbits in Hamiltonian systems that model a diverse array of problems in physical sciences and engineering. The unstable periodic orbits exist in the bottleneck of the equipotential line *V*(*x*, *y*)=*E* and project as lines on the configuration space (*x*, *y*) (Wiggins 2016). The three methods (Pollak, Child, and Pechukas 1980,Koon et al. (2011),Naik and Wiggins (2019)) described below have been implemented for three Hamiltonian systems of the form kinetic plus potential energy and a brief description can be found in the [paper.pdf](https://github.com/WyLyu/UPOsHam/tree/master/paper/paper.pdf). The scripts are written as demonstration of how to modify and adapt the code for a problem of interest.

Usage
-----

**Comparison of the three methods for the coupled quartic Hamiltonian** Comparison of the results (data is located [here](https://github.com/WyLyu/UPOsHam/tree/master/data)) obtained using the three methods for the coupled quartic Hamiltonian can be done using

``` python
>> run run_comparison_coupled.py
```

and the generated figure is located [here](tests/comparison_coupled.pdf)

To obtain the unstable periodic orbits for a specific model Hamiltonian using a specific method, one uses

``` python
>> run run_diffcorr_POfam_coupled.py
```

Other tests can be performed by running the cells in the run\_alltests.ipynb

Copyright and License
---------------------

Copyright 2019 WenYang Lyu, Shibabrat Naik, Stephen Wiggins.

All content is under Creative Commons Attribution [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode.txt) and all the python scripts are under [BSD-3 clause](https://github.com/WyLyu/UPOsHam/blob/master/LICENSE).

References
----------

Koon, W. S., M. W. Lo, J. E. Marsden, and S. D. Ross. 2011. *Dynamical systems, the three-body problem and space mission design*. Marsden books.

Naik, Shibabrat, and Stephen Wiggins. 2019. “Finding Normally Hyperbolic Invariant Manifolds in Two and Three Degrees of Freedom with Hénon-Heiles-Type Potential.” *Phys. Rev. E* 100 (2). American Physical Society: 022204. doi:[10.1103/PhysRevE.100.022204](https://doi.org/10.1103/PhysRevE.100.022204).

Pollak, Eli, Mark S. Child, and Philip Pechukas. 1980. “Classical Transition State Theory: A Lower Bound to the Reaction Probability.” *The Journal of Chemical Physics* 72 (3): 1669–78. doi:[10.1063/1.439276](https://doi.org/10.1063/1.439276).

Wiggins, Stephen. 2016. “The Role of Normally Hyperbolic Invariant Manifolds (NHIMS) in the Context of the Phase Space Setting for Chemical Reaction Dynamics.” *Regular and Chaotic Dynamics* 21 (6): 621–38.
