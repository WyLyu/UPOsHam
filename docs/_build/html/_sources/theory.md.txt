-   [Theory](#theory)
-   [Available methods](#methods)
-   [Example Hamiltonian systems](#examples)
-   [References](#references)

Theory
======

Computation of unstable periodic orbits in two degrees of freedom
Hamiltonian systems arises in studying transition dynamics in physical
sciences (for example chemical reactions, celestial mechanics) and
engineering (for example, ship dynamics and capsize, structural
mechanics) (Parker and Chua [1989](#ref-Parker_1989); Wiggins
[2003](#ref-wiggins_introduction_2003)).

A two degree-of-freedom Hamiltonian system of the form kinetic plus
potential energy is represented by

ℋ(*x*, *y*, *p*<sub>*x*</sub>, *p*<sub>*y*</sub>) = *T*(*p*<sub>*x*</sub>, *p*<sub>*y*</sub>) + *V*(*x*, *y*)

where *x*, *y* are configuration space coordinates,
*p*<sub>*x*</sub>, *p*<sub>*y*</sub> are corresponding momenta,
*V*(*x*, *y*) is the potential energy, and
*T*(*p*<sub>*x*</sub>, *p*<sub>*y*</sub>) is the kinetic energy. The
unstable periodic orbits exist in the bottleneck of the equipotential
contour given by *V*(*x*, *y*) = *E* where *E* is the total energy. For
the Hamiltonian system of the form kinetic plus potential energy, the
unstable periodic orbit projects as a line on the configuration space
(*x*, *y*) (Wiggins [2016](#ref-wiggins_role_2016)). The objective is to
compute this orbit which exists for energies above the energy of the
index-1 saddle equilibrium point located in the bottleneck (Wiggins
[2003](#ref-wiggins_introduction_2003)).

Available methods
=================

This section gives a broad overview of the methods as a reference for
implementing new methods and as how to guide to solve new systems. The
methods are implemented as modules and are part of the internal code
that do not require modification. More details can be found in the
papers listed below:

-   Differential correction (Koon et al.
    [2000](#ref-koon_heteroclinic_2000), [2011](#ref-Koon2011); Naik and
    Ross [2017](#ref-naik_geometry_2017); Ross et al.
    [2018](#ref-ross_experimental_2018); Naik and Wiggins
    [2019](#ref-naik_finding_2019b))
-   Turning point (Pollak, Child, and Pechukas [1980](#ref-Pollak_1980);
    De Leon and Berne [1981](#ref-Deleon_Berne_1981))
-   Turning point based on configuration difference: this is a
    modification of the turning point method and does not rely on dot
    product computation.

Example Hamiltonian systems
===========================

In the sections below, we briefly describe the Hamiltonian systems with
potential wells connected by a bottleneck and these are used to
demonstrate the methods mentioned in [Introduction](#introduction).

-   De Leon-Berne Hamiltonian (De Leon and Berne
    [1981](#ref-Deleon_Berne_1981); De Leon and Marston
    [1989](#ref-DeLeon_Marston_1989); Marston and De Leon
    [1989](#ref-marston_reactive_1989)).

-   Uncoupled and coupled quartic Hamiltonian

References
==========

De Leon, Nelson, and B. J. Berne. 1981. “Intramolecular Rate Process:
Isomerization Dynamics and the Transition to Chaos.” *The Journal of
Chemical Physics* 75 (7): 3495–3510. <https://doi.org/10.1063/1.442459>.

De Leon, N, and C. Clay Marston. 1989. “Order in Chaos and the Dynamics
and Kinetics of Unimolecular Conformational Isomerization.” *The Journal
of Chemical Physics* 91 (6): 3405–25.
<https://doi.org/10.1063/1.456915>.

Koon, Wang Sang, Martin W. Lo, Jerrold E. Marsden, and Shane D. Ross.
2000. “Heteroclinic Connections Between Periodic Orbits and Resonance
Transitions in Celestial Mechanics.” *Chaos: An Interdisciplinary
Journal of Nonlinear Science* 10 (2): 427–69.

Koon, W. S., M. W. Lo, J. E. Marsden, and S. D. Ross. 2011. *Dynamical
systems, the three-body problem and space mission design*. Marsden
books.

Marston, C. Clay, and N. De Leon. 1989. “Reactive Islands as Essential
Mediators of Unimolecular Conformational Isomerization: A Dynamical
Study of 3‐phospholene.” *The Journal of Chemical Physics* 91 (6):
3392–3404. <https://doi.org/10.1063/1.456914>.

Naik, Shibabrat, and Shane D. Ross. 2017. “Geometry of Escaping Dynamics
in Nonlinear Ship Motion.” *Communications in Nonlinear Science and
Numerical Simulation* 47 (June): 48–70.

Naik, Shibabrat, and Stephen Wiggins. 2019. “Finding Normally Hyperbolic
Invariant Manifolds in Two and Three Degrees of Freedom with
Hénon-Heiles-Type Potential.” *Phys. Rev. E* 100 (2): 022204.
<https://doi.org/10.1103/PhysRevE.100.022204>.

Parker, T. S., and L. O. Chua. 1989. *Practical Numerical Algorithms for
Chaotic Systems*. New York, NY, USA: Springer-Verlag New York, Inc.

Pollak, Eli, Mark S. Child, and Philip Pechukas. 1980. “Classical
Transition State Theory: A Lower Bound to the Reaction Probability.”
*The Journal of Chemical Physics* 72 (3): 1669–78.
<https://doi.org/10.1063/1.439276>.

Ross, Shane D., Amir E. BozorgMagham, Shibabrat Naik, and Lawrence N.
Virgin. 2018. “Experimental Validation of Phase Space Conduits of
Transition Between Potential Wells.” *Phys. Rev. E* 98 (5): 052214.

Wiggins, Stephen. 2003. *Introduction to Applied Nonlinear Dynamical
Systems and Chaos*. 2nd ed. Texts in Applied Mathematics 2. New York:
Springer.

———. 2016. “The Role of Normally Hyperbolic Invariant Manifolds (NHIMs)
in the Context of the Phase Space Setting for Chemical Reaction
Dynamics.” *Regular and Chaotic Dynamics* 21 (6): 621–38.
<https://doi.org/10.1134/S1560354716060034>.
