import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c
from tqdm import tqdm

sys.path.append('../')
import wakis

print("\n---------- Initializing simulation ------------------")
# Domain bounds and grid
xmin, xmax = -1., 1.
ymin, ymax = -1., 1.
zmin, zmax = 0., 10.

Nx, Ny = 20, 20
Nz = 200

grid = wakis.GridFIT3D(xmin, xmax, ymin, ymax, zmin, zmax, 
                       Nx, Ny, Nz)

# Source
amplitude = 1000.
nodes = 15
planeWave = wakis.sources.PlaneWave(xs=slice(0, Nx), ys=slice(0,Ny), zs=0, 
                                    nodes=nodes, beta=1.0, amplitude=amplitude)


# Boundary conditions and solver
bc_low = ['periodic', 'periodic', 'pec']
bc_high = ['periodic', 'periodic', 'pml']
n_pml = 10 # number of pml cells

solver = wakis.SolverFIT3D(grid, use_stl=False, 
                           bc_low=bc_low, bc_high=bc_high,
                           n_pml=n_pml)
# Simulation
Nt = int(2.0*(zmax-zmin)/c/solver.dt)

for n in tqdm(range(Nt)):

    if n < Nt/4: # Fill half the domain
        planeWave.update(solver, n*solver.dt)

    solver.one_step()

    if n == 1000:
        Hi = solver.H.copy()
        Ei = solver.E.copy()

    if n%200 == 0:
        solver.plot2D('Hy', plane='ZY', pos=0.5, cmap='rainbow', 
            interpolation='spline36', n=n, vmin=-amplitude, vmax=amplitude,
            off_screen=True, title='005_img/2Dplot_Hy') 
        
    if n == 4000:
        Hf = solver.H.copy()
        Ef = solver.E.copy()
