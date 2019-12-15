"""
Test suite for obtaining unstable periodic orbits in the uncoupled 
quartic Hamiltonian
"""

import numpy as np
from scipy.integrate import solve_ivp

import unittest

import sys
sys.path.insert(0, './src/')
import differential_correction as diffcorr
sys.path.insert(0, './examples/')
import uncoupled_quartic_hamiltonian as uncoupled
import differential_correction_uncoupled as diff_corr_unc
import turning_pt_uncoupled

from scipy.spatial.distance import directed_hausdorff

def hausd_dist_numeric_analytic(orbit, t, deltaE_val, parameters):
    """
    Obtain Hausdorff distance between the unstable periodic orbits obtained
    using a numerical method and analytical solution for the uncoupled 
    quartic Hamiltonian
    """

    total_energy = deltaE_val + parameters[2]
    y, py = uncoupled.upo_analytical(total_energy, t, parameters)

    numerical_orbit = np.array([orbit[:,1], orbit[:,3]])
    analytical_orbit = np.array([y, py])

    hausd_dist = directed_hausdorff(numerical_orbit, analytical_orbit)[0]

    return hausd_dist

class TestUnstablePeriodicOrbit(unittest.TestCase):
    """ Test unstable periodic orbit for the uncoupled system """

    def test_differential_correction(self):
        """ Test solution obtained using differential correction """
    
        MASS_A = 1.0
        MASS_B = 1.0
             
        SADDLE_ENERGY = 0.0 # Energy of the saddle
        ALPHA = 1.00
        BETA = 1.00
        OMEGA = 1
        EPSILON = 0.00 # uncoupled
        parameters = np.array([MASS_A, MASS_B, SADDLE_ENERGY, \
                            ALPHA, BETA, OMEGA, EPSILON])


        deltaE_vals = [0.1]
        linecolors = ['b']
        diff_corr_unc.upo(deltaE_vals, linecolors)
        
        data_path = "./data/"
        TSPAN = [0, 30] # arbitrary range, just to finish the integration
        RelTol = 3.e-12
        AbsTol = 1.e-12

        for deltaE_val in deltaE_vals:
            po_fam_file = "x0_diffcorr_deltaE%s_uncoupled.dat" %(deltaE_val)
            x0podata = np.loadtxt(data_path + po_fam_file)
            x0po_diffcorr = x0podata[0:4]

            f = lambda t,x: uncoupled.ham2dof_uncoupled(t,x,parameters)
            soln = solve_ivp(f, TSPAN, x0po_diffcorr,method='RK45',dense_output=True, \
                 events = lambda t,x : uncoupled.half_period_uncoupled(t,x,parameters), \
                 rtol=RelTol, atol=AbsTol)

            te = soln.t_events[0]
            tt = [0,te[2]]
            t,x_diffcorr,phi_t1,PHI = diffcorr.state_transit_matrix(
                tt, x0po_diffcorr, parameters, \
                uncoupled.variational_eqns_uncoupled)

            total_energy = deltaE_val + parameters[2]
            y, py = uncoupled.upo_analytical(total_energy, t, parameters)

            numerical_orbit = np.array([x_diffcorr[:,1], x_diffcorr[:,3]])
            analytical_orbit = np.array([y, py])

            np.testing.assert_array_almost_equal(numerical_orbit, 
                                                analytical_orbit)

            hausd_dist = hausd_dist_numeric_analytic(
                x_diffcorr, t, deltaE_val, parameters)
            self.assertLessEqual(hausd_dist, 1e-8)


    def test_turning_point(self):
        """Test the solution obtained using turning point"""
    
        MASS_A = 1.0
        MASS_B = 1.0
             
        SADDLE_ENERGY = 0.0 # Energy of the saddle
        ALPHA = 1.00
        BETA = 1.00
        OMEGA = 1
        EPSILON = 0.00 # uncoupled
        parameters = np.array([MASS_A, MASS_B, SADDLE_ENERGY, \
                            ALPHA, BETA, OMEGA, EPSILON])


        deltaE_vals = [0.1]
        linecolors = ['b']
        turning_pt_uncoupled.upo(deltaE_vals, linecolors)
        
        data_path = "./data/"
        TSPAN = [0, 30] # arbitrary range, just to finish the integration
        RelTol = 3.e-12
        AbsTol = 1.e-12

        for deltaE_val in deltaE_vals:
            po_fam_file = "x0_turningpoint_deltaE%s_uncoupled.dat" %(deltaE_val)
            x0podata = np.loadtxt(data_path + po_fam_file)
            x0po_turning_pt = x0podata[-1,0:4]

            f = lambda t,x: uncoupled.ham2dof_uncoupled(t,x,parameters)
            soln = solve_ivp(f, TSPAN, x0po_turning_pt,method='RK45', dense_output=True, \
                            events = lambda t,x : uncoupled.half_period_uncoupled(t,x,parameters), \
                            rtol=RelTol, atol=AbsTol)

            te = soln.t_events[0]
            tt = [0,te[2]]
            t,x_turning_pt,phi_t1,PHI = diffcorr.state_transit_matrix(
                tt, x0po_turning_pt, parameters, \
                uncoupled.variational_eqns_uncoupled)

            total_energy = deltaE_val + parameters[2]
            y, py = uncoupled.upo_analytical(total_energy, t, parameters)

            numerical_orbit = np.array([x_turning_pt[:,1], x_turning_pt[:,3]])
            analytical_orbit = np.array([y, py])

            np.testing.assert_array_almost_equal(numerical_orbit, 
                                                analytical_orbit)

            hausd_dist = hausd_dist_numeric_analytic(
                x_turning_pt, t, deltaE_val, parameters)
            self.assertLessEqual(hausd_dist, 1e-8)


    def test_turning_point_coord_difference(self):
        """Test the solution obtained using turning point based on coordinate difference"""
    
        MASS_A = 1.0
        MASS_B = 1.0
             
        SADDLE_ENERGY = 0.0 # Energy of the saddle
        ALPHA = 1.00
        BETA = 1.00
        OMEGA = 1
        EPSILON = 0.00 # uncoupled
        parameters = np.array([MASS_A, MASS_B, SADDLE_ENERGY, \
                            ALPHA, BETA, OMEGA, EPSILON])


        deltaE_vals = [0.1]
        linecolors = ['b']
        turning_pt_uncoupled.upo(deltaE_vals, linecolors)
        
        data_path = "./data/"
        TSPAN = [0, 30] # arbitrary range, just to finish the integration
        RelTol = 3.e-12
        AbsTol = 1.e-12

        for deltaE_val in deltaE_vals:
            po_fam_file = "x0_tpcd_deltaE%s_uncoupled.dat" %(deltaE_val)
            x0podata = np.loadtxt(data_path + po_fam_file)
            x0po_tpcd = x0podata[-1,0:4]

            f = lambda t,x: uncoupled.ham2dof_uncoupled(t,x,parameters)
            soln = solve_ivp(f, TSPAN, x0po_tpcd,method='RK45', dense_output=True, \
                            events = lambda t,x : uncoupled.half_period_uncoupled(t,x,parameters), \
                            rtol=RelTol, atol=AbsTol)

            te = soln.t_events[0]
            tt = [0,te[2]]
            t,x_tpcd,phi_t1,PHI = diffcorr.state_transit_matrix(
                tt, x0po_tpcd, parameters, \
                uncoupled.variational_eqns_uncoupled)

            total_energy = deltaE_val + parameters[2]
            y, py = uncoupled.upo_analytical(total_energy, t, parameters)

            numerical_orbit = np.array([x_tpcd[:,1], x_tpcd[:,3]])
            analytical_orbit = np.array([y, py])

            np.testing.assert_array_almost_equal(numerical_orbit, 
                                                analytical_orbit)

            hausd_dist = hausd_dist_numeric_analytic(
                x_tpcd, t, deltaE_val, parameters)
            self.assertLessEqual(hausd_dist, 1e-8)





if __name__ == "__main__":
    unittest.main()








