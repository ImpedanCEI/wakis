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
    WP = np.array([-7.82927244e-18, -5.90860497e-16, -1.19806535e-13, -1.57303857e-11,
       -1.32039935e-09, -7.09710294e-08, -2.44454996e-06, -5.39548557e-05,
       -7.62410241e-04, -6.88363851e-03, -3.95543072e-02, -1.43343441e-01,
       -3.19453895e-01, -3.99972857e-01, -1.51160201e-01,  3.49813641e-01,
        6.62895147e-01,  4.74468395e-01, -3.09625689e-02, -3.99716869e-01,
       -3.51900366e-01,  5.49605875e-02,  4.45736690e-01,  3.68587395e-01,
       -1.63249810e-01, -5.32791132e-01, -2.63028461e-01,  3.07629388e-01,
        4.86200533e-01,  1.25327280e-01, -3.21807471e-01, -4.18505615e-01,
       -1.05020417e-01,  3.38140758e-01,  4.57931451e-01,  5.01811483e-02,
       -4.60012790e-01, -4.33220217e-01,  1.15282700e-01,  5.00489230e-01,
        2.95748120e-01, -1.94351334e-01, -4.39658451e-01, -2.44534445e-01,
        1.97542197e-01,  4.70159911e-01,  2.38902888e-01, -3.07117085e-01,
       -5.21977131e-01, -1.08152865e-01,  4.25958412e-01,  4.36817373e-01,
       -2.74884090e-02, -4.09922819e-01, -3.54928662e-01,  4.31862228e-02,
        4.16394430e-01,  3.75690618e-01, -1.10623838e-01, -5.16153616e-01,
       -3.14246426e-01,  2.70430066e-01,  5.13870625e-01,  1.60392558e-01,
       -3.23201903e-01, -4.28419810e-01, -1.09979492e-01,  3.16410068e-01,
        4.44327893e-01,  8.76442964e-02, -4.18314843e-01, -4.60188912e-01,
        5.81786462e-02,  5.03044650e-01,  3.41410047e-01, -1.80785051e-01,
       -4.54315489e-01, -2.50893896e-01,  1.85179582e-01,  4.48362694e-01,
        2.56044975e-01, -2.57199393e-01, -5.20664344e-01, -1.64834075e-01,
        4.00124138e-01,  4.75353321e-01,  5.39973211e-03, -4.19821139e-01,
       -3.69142728e-01,  3.69990213e-02,  3.98779887e-01,  3.71822410e-01,
       -6.95696794e-02, -4.85675917e-01, -3.53910868e-01,  2.19089956e-01])

    Z = np.array([ 8.40258556e+00   -0.j , -7.96826245e+00   -4.36309725j,
       -4.01697123e+00  +17.32625165j,  8.35199559e+00   +6.89722348j,
       -9.94420860e+00   +9.77323632j,  1.75243467e+00  +29.11030303j,
        8.16121325e+00  +14.12314797j, -9.65236165e+00  +25.16293279j,
        9.20981958e+00  +39.5867789j ,  7.74372776e+00  +22.14326272j,
       -6.57929800e+00  +41.89177955j,  1.81294768e+01  +48.52126895j,
        7.06006504e+00  +31.69182734j,  6.53370172e-02  +59.93321301j,
        2.80620218e+01  +55.65689284j,  6.28653013e+00  +43.89533983j,
        1.13678993e+01  +78.99874421j,  3.82126565e+01  +60.8878745j ,
        6.08108304e+00  +60.38796659j,  2.86879092e+01  +98.36272625j,
        4.73421849e+01  +64.56690094j,  7.99820878e+00  +83.43295714j,
        5.35962563e+01 +116.66671458j,  5.37140745e+01  +68.04094635j,
        1.51829240e+01 +116.10624836j,  8.78033978e+01 +131.70094661j,
        5.51211294e+01  +74.65872686j,  3.37192483e+01 +162.73614148j,
        1.33193153e+02 +140.12162181j,  4.90427545e+01  +92.00610249j,
        7.59212063e+01 +230.3070049j ,  1.92415359e+02 +136.88044972j,
        3.30284378e+01 +138.35146325j,  1.71233362e+02 +333.98668579j,
        2.72270721e+02 +113.10647294j,  5.62780103e+00 +270.12833299j,
        4.23024859e+02 +528.12272375j,  4.08769120e+02  +41.25927953j,
       -2.89426312e+01 +830.98817309j,  1.91386579e+03+1409.62242602j,
        2.94304243e+03-1486.05008603j, -1.43567339e+02-2929.8933438j ,
       -1.46772255e+03 -432.9899393j ,  1.40081389e+02 +153.20251989j,
       -1.72684851e+02 -717.47158833j, -5.24924408e+02 +108.49002547j,
        2.62149678e+02  +12.43220056j, -2.27880010e+02 -400.67133866j,
       -2.12371767e+02 +261.16589064j,  2.86118159e+02  -94.37767948j])

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
        assert np.allclose(wake.WP[::50], self.WP), "Wake potential samples failed"
        assert np.cumsum(np.abs(wake.WP))[-1] == pytest.approx(1325.6968037037557, 0.1), "Wake potential cumsum failed"

    def test_long_impedance(self):
        global wake
        assert np.allclose(np.abs(wake.Z)[::20], np.abs(self.Z)), "Abs Impedance samples failed"
        assert np.allclose(np.real(wake.Z)[::20], np.real(self.Z)), "Real Impedance samples failed"
        assert np.allclose(np.imag(wake.Z)[::20], np.imag(self.Z)), "Imag Impedance samples failed"
        assert np.cumsum(np.abs(wake.Z))[-1] == pytest.approx(372019.59123029554, 0.1), "Abs Impedance cumsum failed"
