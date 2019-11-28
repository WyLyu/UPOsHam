-   [Summary](#summary)
-   [Installation](#installation)
-   [Usage](#usage)
    -   [Comparing the methods for the coupled quartic Hamiltonian](#comparing-the-methods-for-the-coupled-quartic-hamiltonian)
-   [Contributing](#contributing)
-   [Copyright and License](#copyright-and-license)
-   [References](#references)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3373396.svg)](https://doi.org/10.5281/zenodo.3373396)

Summary
-------

This Python package is a collection of three methods for computing unstable periodic orbits in two degrees of freedom Hamiltonian systems that model a diverse array of problems in physical sciences and engineering (Parker and Chua [1989](#ref-Parker_1989); Wiggins [2003](#ref-wiggins_introduction_2003)). The unstable periodic orbits exist in the bottleneck of the equipotential line *V*(*x*, *y*)=*E* and project as lines on the configuration space (*x*, *y*) for the Hamiltonian system of the form kinetic plus potential energy (Wiggins [2016](#ref-wiggins_role_2016)). The three methods implemented here (and available under [src](src/) directory) has been used in (Pollak, Child, and Pechukas [1980](#ref-Pollak_1980); De Leon and Berne [1981](#ref-Deleon_Berne_1981); Wang Sang Koon et al. [2000](#ref-koon_heteroclinic_2000); W. S. Koon et al. [2011](#ref-Koon2011); Naik and Ross [2017](#ref-naik_geometry_2017); Ross et al. [2018](#ref-ross_experimental_2018); Naik and Wiggins [2019](#ref-naik_finding_2019b)) for transition dynamics in chemical reactions, celestial mechanics, and ship capsize. We have chosen three Hamiltonian systems that have two wells connected by a bottleneck where the unstable periodic orbits exist for energy above the energy of the saddle equilibrium point. A brief description of these systems can be found in the [paper](https://github.com/WyLyu/UPOsHam/tree/master/paper/paper.pdf).

Installation
------------

Clone/download the git repository using

``` git
$ git clone git@github.com:WyLyu/UPOsHam.git
$ cd UPOsHam
$ pip install -r requirements.txt (or pip3 install -r requirements.txt)
```

and check the modules shown in [requirements.txt](https://github.com/WyLyu/UPOsHam/tree/master/requirements.txt) are installed using conda/pip.

Usage
-----

The example scripts that computes the unstable periodic orbits (UPOs) for a given Hamiltonian system is in the [examples](examples/) directory with names `<method>_<system>.py`. For example, to compute the unstable periodic orbits (the default shows at 2 total energy levels) for the De Leon-Berne Hamiltonian ([1981](#ref-Deleon_Berne_1981)) using the differential correction method can be run from the command line by calling

    $ python examples/differential_correction_deleonberne.py

This will save data files in the current directory and save the following plot in the [plots](tests/plots/) directory.

<img src="tests/plots/diff_corr_deleonberne_upos.png" style="width:80.0%" />

**Fig. Unstable periodic orbits for De Leon-Berne Hamiltonian**

The example scripts import the system specific functions and method specific functions to obtain the UPOs. These are written as illustration of how to apply these methods to new systems.

### Comparing the methods for the coupled quartic Hamiltonian

We demonstrate how the UPOs computed using the different methods compare with each other for a specific system: coupled quartic Hamiltonian.

Comparison of the results (data is located [here](https://github.com/WyLyu/UPOsHam/tree/master/data)) obtained using the three methods for the coupled quartic Hamiltonian can be done using

    $ ipython
    >>> run ./tests/compare_methods_coupled.py

and the generated figure is located [here](tests/comparison_coupled.pdf)

To obtain the unstable periodic orbits for a specific model Hamiltonian using a specific method, one uses

    $ ipython
    >>> run ./examples/diffcorr_UPOs_coupled.py

Other tests can be performed by running the cells in the ./tests/tests.ipynb

Contributing
------------

Guidelines on how to contribute to this package can be found [here](https://github.com/WyLyu/UPOsHam/blob/master/docs/contributing.md) along with the code of conduct [here](https://github.com/WyLyu/UPOsHam/blob/master/CODE_OF_CONDUCT.md) for engaging with the fellow contributors.

Copyright and License
---------------------

Copyright 2019 Wenyang Lyu, Shibabrat Naik, Stephen Wiggins.

All content is under Creative Commons Attribution [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode.txt) and all the python scripts are under [BSD-3 clause](https://github.com/WyLyu/UPOsHam/blob/master/LICENSE).

References
----------

De Leon, Nelson, and B. J. Berne. 1981. “Intramolecular Rate Process: Isomerization Dynamics and the Transition to Chaos.” *The Journal of Chemical Physics* 75 (7): 3495–3510. doi:[10.1063/1.442459](https://doi.org/10.1063/1.442459).

Koon, W. S., M. W. Lo, J. E. Marsden, and S. D. Ross. 2011. *Dynamical systems, the three-body problem and space mission design*. Marsden books.

Koon, Wang Sang, Martin W. Lo, Jerrold E. Marsden, and Shane D. Ross. 2000. “Heteroclinic Connections Between Periodic Orbits and Resonance Transitions in Celestial Mechanics.” *Chaos: An Interdisciplinary Journal of Nonlinear Science* 10 (2): 427–69.

Naik, Shibabrat, and Shane D. Ross. 2017. “Geometry of Escaping Dynamics in Nonlinear Ship Motion.” *Communications in Nonlinear Science and Numerical Simulation* 47 (June): 48–70.

Naik, Shibabrat, and Stephen Wiggins. 2019. “Finding Normally Hyperbolic Invariant Manifolds in Two and Three Degrees of Freedom with Hénon-Heiles-Type Potential.” *Phys. Rev. E* 100 (2). American Physical Society: 022204. doi:[10.1103/PhysRevE.100.022204](https://doi.org/10.1103/PhysRevE.100.022204).

Parker, T. S., and L. O. Chua. 1989. *Practical Numerical Algorithms for Chaotic Systems*. New York, NY, USA: Springer-Verlag New York, Inc.

Pollak, Eli, Mark S. Child, and Philip Pechukas. 1980. “Classical Transition State Theory: A Lower Bound to the Reaction Probability.” *The Journal of Chemical Physics* 72 (3): 1669–78. doi:[10.1063/1.439276](https://doi.org/10.1063/1.439276).

Ross, Shane D., Amir E. BozorgMagham, Shibabrat Naik, and Lawrence N. Virgin. 2018. “Experimental Validation of Phase Space Conduits of Transition Between Potential Wells.” *Phys. Rev. E* 98 (5): 052214.

Wiggins, Stephen. 2003. *Introduction to Applied Nonlinear Dynamical Systems and Chaos*. 2nd ed. Texts in Applied Mathematics 2. New York: Springer.

———. 2016. “The Role of Normally Hyperbolic Invariant Manifolds (NHIMs) in the Context of the Phase Space Setting for Chemical Reaction Dynamics.” *Regular and Chaotic Dynamics* 21 (6): 621–38. doi:[10.1134/S1560354716060034](https://doi.org/10.1134/S1560354716060034).
