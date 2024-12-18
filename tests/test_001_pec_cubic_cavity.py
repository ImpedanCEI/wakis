import os, sys
import numpy as np
import pyvista as pv

sys.path.append('../wakis')

from wakis import SolverFIT3D
from wakis import GridFIT3D 
from wakis import WakeSolver

import pytest 

@pytest.mark.slow
class TestPecCubicCavity:
    # Regression data
    WP = np.array([-8.31875855e-18, -2.24114674e-15, -1.38218248e-12, -4.25071185e-10,
       -6.54369673e-08, -5.05528015e-06, -1.96027717e-04, -3.80754091e-03,
       -3.68351597e-02, -1.74702032e-01, -3.83101173e-01, -2.70709983e-01,
        3.13360888e-01,  6.68086657e-01,  2.79048065e-01, -3.12682108e-01,
       -3.76842112e-01,  1.23349740e-01,  4.86759211e-01,  5.61645266e-02,
       -5.18431886e-01, -1.99116540e-01,  4.41812175e-01,  3.11957816e-01,
       -2.68217213e-01, -4.05614872e-01,  4.84047544e-02,  4.72187252e-01,
        1.53226676e-01, -4.78098417e-01, -2.97190404e-01,  3.90623008e-01,
        3.81622029e-01, -2.08580527e-01, -4.27735630e-01, -2.25254005e-02,
        4.44300511e-01,  2.36574106e-01, -4.16145424e-01, -3.81781277e-01,
        3.17310742e-01,  4.43681426e-01, -1.38616244e-01, -4.42525247e-01,
       -9.00830356e-02,  4.05256072e-01,  3.07050920e-01, -3.39585826e-01,
       -4.49525529e-01,  2.28859590e-01,  4.90870709e-01, -5.85844334e-02,
       -4.48189454e-01, -1.55385210e-01,  3.60362252e-01,  3.60028843e-01,
       -2.52609312e-01, -4.93956075e-01,  1.25940339e-01,  5.19895197e-01,
        3.12718645e-02, -4.44021047e-01, -2.17909370e-01,  3.09361559e-01,
        3.98665263e-01, -1.60522772e-01, -5.17207717e-01,  1.81451130e-02,
        5.26978861e-01,  1.25853748e-01, -4.26024407e-01, -2.78841581e-01,
        2.55176053e-01,  4.23200828e-01, -7.04112799e-02, -5.14796734e-01,
       -9.11352723e-02])

    Z = np.array([ 8.34681676e+00   -0.j, -7.87358694e+00   -4.37756805j,
       -4.15976866e+00  +17.21001754j,  8.38135123e+00   +7.10591041j,
       -9.75626349e+00   +9.47670898j,  1.30533751e+00  +29.0581918j ,
        8.43780907e+00  +14.45246819j, -9.58940029e+00  +24.49282636j,
        8.48522248e+00  +39.8252252j ,  8.39770851e+00  +22.40422977j,
       -6.93398758e+00  +40.8489626j ,  1.72801136e+01  +49.28578505j,
        8.14340301e+00  +31.57574076j, -1.06306285e+00  +58.66719406j,
        2.74109968e+01  +57.14071546j,  7.69071429e+00  +42.96384934j,
        9.09012982e+00  +77.88948401j,  3.82970813e+01  +63.14042114j,
        7.40651470e+00  +58.09642329j,  2.49714486e+01  +98.12221233j,
        4.89304148e+01  +67.31583442j,  8.36415535e+00  +79.22542777j,
        4.84483518e+01 +118.4436641j ,  5.77505658e+01  +70.40048153j,
        1.29630465e+01 +109.64499396j,  8.19118952e+01 +137.15605741j,
        6.25129245e+01  +74.65256013j,  2.61617332e+01 +154.38366946j,
        1.28631213e+02 +151.41469179j,  6.00980847e+01  +85.70846602j,
        5.84874877e+01 +222.14390848j,  1.94093239e+02 +156.45583905j,
        4.59858716e+01 +117.89618395j,  1.35819704e+02 +332.30551177j,
        2.91902834e+02 +142.98612535j,  1.17002302e+01 +216.35953621j,
        3.48738109e+02 +552.12295519j,  4.83636245e+02  +80.61481563j,
       -7.94527060e+01 +646.72743826j,  1.56460117e+03+1541.70218997j,
        3.08836109e+03 -998.73834851j,  3.54797928e+02-3037.32237706j,
       -1.55740525e+03 -799.6020204j , -3.44957489e+01 +276.70580775j,
       -1.06833073e+01 -701.23028409j, -6.01502220e+02  -33.9602918j ,
        2.02787484e+02 +131.30127401j, -1.07474628e+02 -439.39672784j,
       -3.15587846e+02 +184.70219686j,  2.89287726e+02  +22.25026014j,
       -1.74895524e+02 -283.6577144j ])

    def test_simulation(self):
        print("\n---------- Initializing simulation ------------------")
        # Number of mesh cells
        Nx = 50
        Ny = 50
        Nz = 150

        # Embedded boundaries
        stl_file = 'tests/stl/001_cubic_cavity.stl' 
        surf = pv.read(stl_file)

        stl_solids = {'cavity': stl_file}
        stl_materials = {'cavity': 'vacuum'}

        # Domain bounds
        xmin, xmax, ymin, ymax, zmin, zmax = surf.bounds

        # set grid and geometry
        grid = GridFIT3D(xmin, xmax, ymin, ymax, zmin, zmax, Nx, Ny, Nz, 
                        stl_solids=stl_solids, 
                        stl_materials=stl_materials)
            
        # Beam parameters
        beta = 1.          # beam beta
        sigmaz = 18.5e-3*beta    #[m]
        q = 1e-9            #[C]
        xs = 0.             # x source position [m]
        ys = 0.             # y source position [m]
        xt = 0.             # x test position [m]
        yt = 0.             # y test position [m]

        global wake 
        wake = WakeSolver(q=q, sigmaz=sigmaz, beta=beta,
                    xsource=xs, ysource=ys, xtest=xt, ytest=yt,
                    save=False, logfile=False, Ez_file='tests/001_Ez.h5')

        # boundary conditions
        bc_low=['pec', 'pec', 'pec']
        bc_high=['pec', 'pec', 'pec']

        # set Solver object
        solver = SolverFIT3D(grid, wake,
                            bc_low=bc_low, bc_high=bc_high, 
                            use_stl=True, bg='pec')

        wakelength = 1. #[m]
        add_space = 12  # no. cells
        solver.wakesolve(wakelength=wakelength, add_space=add_space, save_J=False)
        os.remove('tests/001_Ez.h5')

    def test_long_wake_potential(self):
        global wake
        assert np.allclose(wake.WP[::100], self.WP), "Wake potential samples failed"
        assert np.cumsum(np.abs(wake.WP))[-1] == pytest.approx(2133.916823624629, 0.1), "Wake potential cumsum failed"

    def test_long_impedance(self):
        global wake
        assert np.allclose(np.abs(wake.Z)[::20], np.abs(self.Z)), "Abs Impedance samples failed"
        assert np.allclose(np.real(wake.Z)[::20], np.real(self.Z)), "Real Impedance samples failed"
        assert np.allclose(np.imag(wake.Z)[::20], np.imag(self.Z)), "Imag Impedance samples failed"
        assert np.cumsum(np.abs(wake.Z))[-1] == pytest.approx(373333.4461071534, 0.1), "Abs Impedance cumsum failed"
