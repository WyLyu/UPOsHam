Turning point
=============

This method is based on finding the unstable periodic orbit by detecting trajectories initialized on the equipotential contour (:math:`V(x,y) = E` where :math:`V(x,y)` is the potetial energy function and :math:`E` is the total energy) that turn in the opposite directions. This method relies on the fact that for Hamiltonians of the form kinetic plus potential energy the unstable periodic orbit is the limiting trajectory that bounces back and forth between the equipotential contour corresponding to the given total energy. So to converge on this limiting trajectory, the turning point method iteratively decreases the gap between the bounding trajectories that turn in the opposite directions. Detection of turning is done using a dot product condition which leads to stalling of the method beyond a certain tolerance (typically :math:`10^{-6}` in the examples here.)

   .. automodule:: turning_point
      :members:


