Quartic Hamiltonian
===================

.. math::
    \begin{equation}
        \mathcal{H}(x,y,p_x,p_y) = T(p_x, p_y) + V(x, y) = \frac{p_x^2}{2m_A} + \frac{p_y^2}{2m_B} + V(x, y)
    \end{equation}

where the potential energy function :math:`V_(x,y)` is

.. math::
    \begin{align}
        V(x,y) = - \alpha \frac{x^2}{2} + \beta \frac{x^4}{4} + \frac{\omega}{2}y^2 + \frac{\epsilon}{2}(x - y)^2
    \end{align}

The parameters in the model are :math:`m_A, m_B` which represent mass of the molecular configurations A and B, while :math:`\alpha, \beta` denote features of the potential energy surface geometry, :math:`\omega, \epsilon` denote the frequency of the harmonic bath mode, and coupling strength.  

Coupled quartic Hamiltonian
---------------------------

We describe the coupled case (:math:`\epsilon \neq 0.0`) to illustrate how to set up a script for computing the unstable periodic orbit using the different methods.

The expected result of the computation using differential correction method is the figure below. 

.. _fig-upo-coupled:

.. figure:: ../tests/plots/diff_corr_coupled_upos.png

   Figure showing the unstable periodic orbits in the bottleneck of the two degrees of freedom coupled quartic Hamiltonian. The unstable periodic orbits at two values of the total energy are shown in red and blue, and the equipotential contours are projected on the :math:`x-y` plane (:math:`p_y = 0`).


The functions below implement the expressions for this specific Hamiltonian and that are sent as parameters to the functions for a method described in :ref:`sect-methods`.

.. automodule:: coupled_quartic_hamiltonian
   :members:



