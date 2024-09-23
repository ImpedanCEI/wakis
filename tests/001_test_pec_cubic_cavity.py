import os, sys
import numpy as np
import pyvista as pv

sys.path.append('../')

from wakis import SolverFIT3D
from wakis import GridFIT3D 
from wakis import WakeSolver

import pytest 

# ---------- Domain setup ---------
# Number of mesh cells
Nx = 50
Ny = 50
Nz = 150

# Embedded boundaries
stl_file = 'stl/001_cubic_cavity.stl' 
surf = pv.read(stl_file)

stl_solids = {'cavity': stl_file}
stl_materials = {'cavity': 'vacuum'}

# Domain bounds
xmin, xmax, ymin, ymax, zmin, zmax = surf.bounds
Lx, Ly, Lz = (xmax-xmin), (ymax-ymin), (zmax-zmin)

# set grid and geometry
grid = GridFIT3D(xmin, xmax, ymin, ymax, zmin, zmax, Nx, Ny, Nz, 
                stl_solids=stl_solids, 
                stl_materials=stl_materials)
    
# ------------ Beam source ----------------
# Beam parameters
beta = 1.          # beam beta
sigmaz = 18.5e-3*beta    #[m]
q = 1e-9            #[C]
xs = 0.             # x source position [m]
ys = 0.             # y source position [m]
xt = 0.             # x test position [m]
yt = 0.             # y test position [m]

wake = WakeSolver(q=q, sigmaz=sigmaz, beta=beta,
            xsource=xs, ysource=ys, xtest=xt, ytest=yt,
            save=False, logfile=False, Ez_file='001_Ez.h5')

# ----------- Solver & Simulation ----------
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
os.remove('001_Ez.h5')

# ----------- Assert -----------------------

# 00 - Longitudinal wake potential
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

assert np.allclose(wake.WP[::100], WP), "Wake potential samples failed"
assert np.cumsum(np.abs(wake.WP))[-1] == pytest.approx(2133.916823624629, 0.1), "Wake potential cumsum failed"

# 01 - Longitudinal impedance

