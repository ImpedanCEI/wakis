import numpy as np
from scipy.constants import c as c_light, epsilon_0 as eps_0, mu_0 as mu_0
import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import patches
import os, sys
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.append('../')

from solverFIT3D import SolverFIT3D
from solver3D import EMSolver3D
from grid3D import Grid3D
from conductors3d import noConductor, InCube, ConductorsAssembly
from scipy.special import jv
from field import Field 
import pprofile

Z0 = np.sqrt(mu_0 / eps_0)

L = 1.
# Number of mesh cells
N = 100
Nx = N
Ny = N
Nz = N
Lx = L
Ly = L
Lz = L
dx = L / Nx
dy = L / Ny
dz = L / Nz

xmin = -Lx/2  + dx / 2
xmax = Lx/2 + dx / 2
ymin = - Ly/2 + dx / 2
ymax = Ly/2 + dx / 2
zmin = - Lz/2 + dx / 2
zmax = Lz/2 + dx / 2

#Embedded cube 
lx = Lx*0.7
ly = Ly*0.7
lz = Lz*0.7
x_cent = 0
y_cent = 0
z_cent = 0
cube = InCube(lx, ly, lz, x_cent, y_cent, z_cent) #noConductor() 
conductors = ConductorsAssembly([cube])

NCFL = 1

profiler = pprofile.Profile()
with profiler:

    # set FIT solver
    gridFIT = Grid3D(xmin, xmax, ymin, ymax, zmin, zmax, Nx, Ny, Nz, conductors, 'FIT')
    tgridFIT = Grid3D(xmin + dx/2, xmax + dx/2, ymin + dy/2, ymax + dy/2, zmin + dz/2, zmax + dz/2, Nx, Ny, Nz, conductors, 'FIT')
    #solverFIT = SolverFIT3D(gridFIT)
    solverFIT = SolverFIT3D(gridFIT, tgridFIT)
    solverFIT.E[int(Nx/2), int(Ny/2), int(Nz/2), 'z'] = 1.0*c_light

    Nt = 50

    for n in tqdm(range(Nt), 'FIT:'):

        solverFIT.one_step()

profiler = pprofile.Profile()
with profiler:

    # set FDTD solver
    gridFDTD = Grid3D(xmin, xmax, ymin, ymax, zmin, zmax, Nx, Ny, Nz, conductors, 'FDTD')
    solverFDTD = EMSolver3D(gridFDTD, 'FDTD', NCFL)
    solverFDTD.Ez[int(Nx/2), int(Ny/2),  int(Nz/2)] = 1.0*c_light
    for n in tqdm(range(Nt), 'FDTD:'):

        solverFDTD.one_step()


