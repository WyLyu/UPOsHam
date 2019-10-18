-   [Summary](#summary)
-   [Installation](#installation)
-   [Usage](#usage)
-   [Contributing](#contributing)
-   [Copyright and License](#copyright-and-license)
-   [References](#references)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3373396.svg)](https://doi.org/10.5281/zenodo.3373396)

Summary
-------

This Python package is a collection of three methods for computing unstable periodic orbits in Hamiltonian systems that model a diverse array of problems in physical sciences and engineering (Wiggins 2003). The unstable periodic orbits exist in the bottleneck of the equipotential line *V*(*x*, *y*)=*E* and project as lines on the configuration space (*x*, *y*) (Wiggins 2016). The three methods (Pollak, Child, and Pechukas 1980,Koon et al. (2011),Naik and Wiggins (2019)) described below have been implemented for three Hamiltonian systems of the form kinetic plus potential energy and a brief description can be found in the [paper.pdf](https://github.com/WyLyu/UPOsHam/tree/master/paper/paper.pdf). The scripts are written as demonstration of how to modify and adapt the code for a problem of interest.

Installation
------------

Clone/download the git repository using

``` git
$ git clone git@github.com:WyLyu/UPOsHam.git
$ cd UPOsHam
$ pip install -r requirements.txt (or pip3 install -r requirements.txt)
```

and check the modules shown in [requirements.txt](https://github.com/WyLyu/UPOsHam/tree/master/requirements.txt) are installed using conda/pip. Specific problems can be imported as modules for use in further analysis.

Usage
-----

**Comparison of the three methods for the coupled quartic Hamiltonian**

Comparison of the results (data is located [here](https://github.com/WyLyu/UPOsHam/tree/master/data)) obtained using the three methods for the coupled quartic Hamiltonian can be done using

    $ ipython
    []: run ./tests/compare_methods_coupled.py

and the generated figure is located [here](tests/comparison_coupled.pdf)

To obtain the unstable periodic orbits for a specific model Hamiltonian using a specific method, one uses

    $ ipython
    []: run ./examples/diffcorr_UPOs_coupled.py

Other tests can be performed by running the cells in the ./tests/tests.ipynb

Contributing
------------

Guidelines on how to contribute to this package can be found [here](https://github.com/WyLyu/UPOsHam/blob/master/docs/contributing.md) and also be sure to check the [code of conduct](https://github.com/WyLyu/UPOsHam/blob/master/CODE_OF_CONDUCT.md).

Copyright and License
---------------------

Copyright 2019 Wenyang Lyu, Shibabrat Naik, Stephen Wiggins.

All content is under Creative Commons Attribution [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode.txt) and all the python scripts are under [BSD-3 clause](https://github.com/WyLyu/UPOsHam/blob/master/LICENSE).

References
----------

Koon, W. S., M. W. Lo, J. E. Marsden, and S. D. Ross. 2011. *Dynamical systems, the three-body problem and space mission design*. Marsden books.

Naik, Shibabrat, and Stephen Wiggins. 2019. “Finding Normally Hyperbolic Invariant Manifolds in Two and Three Degrees of Freedom with Hénon-Heiles-Type Potential.” *Phys. Rev. E* 100 (2). American Physical Society: 022204. doi:[10.1103/PhysRevE.100.022204](https://doi.org/10.1103/PhysRevE.100.022204).

Pollak, Eli, Mark S. Child, and Philip Pechukas. 1980. “Classical Transition State Theory: A Lower Bound to the Reaction Probability.” *The Journal of Chemical Physics* 72 (3): 1669–78. doi:[10.1063/1.439276](https://doi.org/10.1063/1.439276).

Wiggins, Stephen. 2003. *Introduction to Applied Nonlinear Dynamical Systems and Chaos*. 2nd ed. Texts in Applied Mathematics 2. New York: Springer.

———. 2016. “The Role of Normally Hyperbolic Invariant Manifolds (NHIMs) in the Context of the Phase Space Setting for Chemical Reaction Dynamics.” *Regular and Chaotic Dynamics* 21 (6): 621–38. doi:[10.1134/S1560354716060034](https://doi.org/10.1134/S1560354716060034).
