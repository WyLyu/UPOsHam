Turning point based on configuration difference
===============================================

Based on the turning point approach, we have implemented a new method which shows stable convergence and does not rely on the dot product formula. Suppose we have found two initial conditions on a given equipotential contour and they turn in the opposite directions. If the difference in :math:`x`-coordinates is small (:math:`\approx 10^{-2}`), the generated trajectories will approach the UPO from either sides. If the difference in :math:`x`-coordinates is large, we can integrate the Hamilton's equations for a guess time interval and find the turning point (event using ODE event detection) at which the trajectories bounce back from the far side of the equipotential contour in opposite directions. We choose these two points as our initial guess and the difference of :math:`x`-coordinates become small now. Without loss of generality, this method can be modified to either pick the difference of :math:`y`-coordinates or a combination of :math:`x` and :math:`y` coordinates. This choice will depend on the orientation of the potential energy surface's bottleneck in the configuration space.

   .. automodule:: turning_point_coord_difference
      :members:


