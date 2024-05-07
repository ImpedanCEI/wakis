import numpy as np
from scipy.constants import c as c_light, epsilon_0 as eps_0, mu_0 as mu_0
from matplotlib import pyplot as plt
import os, sys
import pprofile
import time

sys.path.append('../')

from solverFIT3D import SolverFIT3D
from solver3D import EMSolver3D
from grid3D import Grid3D
from conductors3d import noConductor, InCube, ConductorsAssembly
from field import Field 

# Computational domain length
L = 1.

# Number of mesh cells
N = 25
Nx = N
Ny = N
Nz = N
Lx = L
Ly = L
Lz = L
dx = L / Nx
dy = L / Ny
dz = L / Nz

# Domain limits
xmin = -Lx/2  + dx / 2
xmax = Lx/2 + dx / 2
ymin = - Ly/2 + dx / 2
ymax = Ly/2 + dx / 2
zmin = - Lz/2 + dx / 2
zmax = Lz/2 + dx / 2

#Embedded cube 
lx = Lx*0.4
ly = Ly*0.4
lz = Lz*0.4
x_cent = 0
y_cent = 0
z_cent = 0
cube = InCube(lx, ly, lz, x_cent, y_cent, z_cent) #noConductor() 
conductors = ConductorsAssembly([cube])
ti = time.time()

profiler = pprofile.Profile()
with profiler:

    # Grid setup
    grid = Grid3D(xmin, xmax, ymin, ymax, zmin, zmax, Nx, Ny, Nz, conductors, 'FIT')

    # Solver setup
    NCFL = 1
    solver = SolverFIT3D(grid, 'FIT', NCFL)

    # Iinitial condition
    solver.E[int(2*Nx/4), int(2*Ny/4), int(Nz/2), 'z'] = 1.0*c_light

    # Tiny profile it
    solver.one_step()

profiler.print_stats()
print('Time for first timestep:')
print(f'Elapsed time: {time.time()-ti}')

'''
# Inspect matrixes
plt.imshow(solver.Da.toarray(), cmap='bwr')
plt.title('Da matrix')
plt.show()

plt.imshow(solver.Ds.toarray(), cmap='bwr')
plt.title('Ds matrix')
plt.show()

plt.imshow(solver.C.toarray(), cmap='bwr')
plt.title('C matrix')
plt.show()
'''

#solver.L.inspect()
#solver.A.inspect()

