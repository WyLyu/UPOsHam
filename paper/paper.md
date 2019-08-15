---
title: "UPOsHam: A Python package for computing unstable periodic orbits in two degrees of freedom Hamiltonian systems"
authors:
- affiliation: 1
  name: WenYang Lyu
  orcid: 0000-0003-2570-9879
- affiliation: 1
  name: Shibabrat Naik
  orcid: 0000-0001-7964-2513
- affiliation: 1
  name: Stephen Wiggins
  orcid: 0000-0002-5036-5863
date: \today
output:
  pdf_document:
    fig_caption: yes
    fig_height: 3
  html_document:
    fig_caption: yes
    fig_height: 3
bibliography: paper.bib
tags:
- Hamiltonian dynamics
- Dynamical systems
- Chemical reaction dynamics
- Unstable periodic orbits
- State transition matrix
- Differential correction
- Numerical continuation
- Turning point
affiliations:
- index: 1
  name: School of Mathematics, Univesity Walk, University of Bristol, Clifton BS8 1TW, Bristol, United Kingdom
---

**[Software repository](https://github.com/WenYangLyu/UPOsHam)**

## Statement of Need

In Hamiltonian systems the fundamental phase space structure that partitions dynamically disparate trajectories and mediates transition between multi-stable regions is an invariant manifold. In a 2N dimensional Hamiltonian phase space, the invariant manifold has 2 less dimension than the phase space and is anchored to the normally hyperbolic invariant manifold which has 3 less dimension. This becomes an unstable periodic orbit (UPO) for 2 degrees of freedom or four dimensional phase space [@wiggins_role_2016]. Since the UPO forms the basis for partitioning trajectories, hence their computation and stability analysis is the starting point for dynamical systems analysis. UPOsHam is meant to serve this purpose by providing examples of how to implement numerical methods for computing the unstable periodic orbits (UPOs) at any specified total energy as long as their existence is guaranteed. Even though, there is no lack of numerical methods for computing UPOs, we have found that they either lack in reproducibility, or have steep learning curve for using the software, or have been written using closed source software, and at times combination of these. Our aim is to provide an open source package that implements some of the standard methods and shows the results in the context of model problems. This is meant as a starting point to integrate other numerical methods in an open source package such that UPOs computed in dynamical systems papers can be reproduced with minimal tweaking while providing an exploratory environment to learn and develop the underlying methods.  



## Summary

This Python package, UPOsHam, is a collection of three methods for computing unstable periodic orbits in Hamiltonian systems that model a diverse array of problems in physical sciences and engineering. The unstable periodic orbits exist in the bottleneck of the equipotential line $V(x,y) = E$ and project as lines on the configuration space $(x,y)$. The three methods described below have been implemented for three Hamiltonian systems of the form kinetic plus potential energy and are described in [\S:Examples](#examples). The scripts are written as demonstration of how to modify and adapt the code for a problem of interest. 
 
The computed unstable periodic orbits using the three methods are compared for a model problem in Figure \ref{fig:allinone_coupled}.

### Features: Available Methods

In this package, the user has the option to choose between the three methods described below. These are implemented in separate scripts with functions that can be modified to define the total energy (Hamiltonian), potential energy, vector field, Jacobian, variational equations [@Parker_1989].   

__Turning point (TP)__

This method is based on finding the UPO by checking for trajectories that turn in the opposite directions and iteratively bringing them closer to approximate the UPO [@Pollak_1980].

__Turning point based on configuration difference  (TPCD)__



__Differential correction and numerical continuation (DCNC)__


## Examples {#examples}

Consider the following two degrees-of-freedom Hamiltonian model of a reaction in a bath (solvent) 

### Uncoupled quartic Hamiltonian

\begin{equation}
    \mathcal{H}(x,y,p_x,p_y) = \frac{p_x^2}{2} - \alpha \frac{x^2}{2} + \beta \frac{x^4}{4} + \frac{\omega}{2}\left( p_y^2 + y^2 \right)
\end{equation}

### Coupled quartic Hamiltonian

\begin{equation}
    \mathcal{H}(x,y,p_x,p_y) = \frac{p_x^2}{2} - \alpha \frac{x^2}{2} + \beta \frac{x^4}{4} + \frac{\omega}{2}\left( p_y^2 + y^2 \right) + \frac{\epsilon}{2}(x - y)^2
\end{equation}


### DeLeon-Berne Hamiltonian {#dbham}

This Hamiltonian has been studied in chemical reaction dynamics as a model of isomerization of a single molecule that undergoes structural changes [@Deleon_Berne_1981; @DeLeon_Marston_1989]. This model Hamiltonian exhibits chaotic dynamics when the coupling between the double well and Morse oscillator is increased.

\begin{equation}
\mathcal{H}(x,y,p_x,p_y) = T(p_x, p_y) + V_{\rm DB}(x, y) = \frac{p_x^2}{2m_A} + \frac{p_y^2}{2m_B} + V_{\rm DB}(x, y)
\end{equation}    
where the potential energy function $V_{\rm DB}(x,y)$ is 

\begin{equation}
\begin{aligned}
V_{\rm DB}(x,y) = &  V(x) + V(y) + V(x,y) \\
V(y) = & 4y^2(y^2 - 1) + \epsilon_s \\
V(x) = & D_x\left[ 1 - \exp(-\lambda x) \right]^2 \\
V(x,y) = & 4y^2(y^2 - 1)\left[ \exp(-\zeta \lambda x) - 1 \right]
\end{aligned}
\label{eqn:pot_energy_db}
\end{equation}

The parameters in the model are $m_A, m_B$ which represent mass of the isomers, while $\epsilon_s, D_x$ denote the energy of the saddle, dissociation energy of the Morse oscillator, respectively, and will be kept fixed in this study, $\lambda, \zeta$ denote the range of the Morse oscillator and coupling parameter between the $x$ and $y$ configuration space coordinates, respectively.

## Visualization: Unstable periodic orbits 

![Comparing the TP, TPCD, DCNC methods for the coupled quartic Hamiltonian. \label{fig:allinone_coupled}](allinone_coupled.pdf)


![Comparing the TPCD method for the three Hamiltonians \label{fig:allinone_newmethod}](allinone_tpcd.pdf)


## Relation to ongoing research projects

We are developing geometric methods of phase space transport in the context of chemical reaction dynamics that rely heavily on identifying and computing the unstable periodic orbits. Manuscript related to the [De Leon-Berne model](#dbham) is under preparation.


## Acknowledgements

We acknowledge the support of EPSRC Grant No. EP/P021123/1 and Office of Naval Research (Grant No. N00014-01-1-0769). The authors would like to acknowledge the London Mathematical Society and School of Mathematics at University of Bristol for supporting the undergraduate research bursary. We acknowledge contributions from Shane Ross for writing the early MATLAB version of the differential correction and numerical continuation code.


## References

