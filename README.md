---
output:
  html_document: default
  pdf_document:
    fig_caption: yes
    fig_height: 3
---
## UPOsHam
UPOsHam is a package for computing unstable periodic orbits in Hamiltonian dynamics.

Across all applied dynamical systems the fundamental phase space structure is the unstable periodic orbit (UPO) and its generalization to higher dimension called the normally hyperbolic invariant manifold. This phase space structure is associated with an index-1 saddle equilibrium point in Hamiltonian systems and acts as an anchor of invariant manifolds that partition dynamically different trajectories in phase space. Their significance has led to development of algorithms that can locate and compute these structures quickly with high accuracy and have stable convergence properties. One of the reasons for focusing on stable efficient algorithms is that the UPO has exponential instability and numerical inaccuracies can grow rapidly and converge to a completely different periodic orbit which are dense in Hamiltonian phase space.

## Usage

To compare the three methods for the coupled quartic Hamiltonian:

```python
>> run run_comparison_coupled.py
```

![Comparison of the three methods](comparison_coupled.pdf)


## Copyright and License
Copyright 2019 WenYang Lyu, Shibabrat Naik, Stephen Wiggins. 

All content is under Creative Commons Attribution [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode.txt) and all the python scripts are under [BSD-3 clause](https://github.com/WyLyu/UPOsHam/blob/master/LICENSE).

