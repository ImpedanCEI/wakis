# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

import os
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import sys
sys.path.append('../')

from wakis import SolverFIT3D
from wakis import GridFIT3D 
from wakis import WakeSolver

from tqdm import tqdm

# ---------- MPI setup ------------
from mpi4py import MPI

comm = MPI.COMM_WORLD  # Get MPI communicator
rank = comm.Get_rank()  # Process ID
size = comm.Get_size()  # Total number of MPI processes

# ---------- Domain setup ---------

# Geometry & Materials
solid_1 = 'data/001_vacuum_cavity.stl'
solid_2 = 'data/001_lossymetal_shell.stl'

stl_solids = {'cavity': solid_1, 
              'shell': solid_2
              }

stl_materials = {'cavity': 'vacuum', 
                 'shell': [30, 1.0, 30] #[eps_r, mu_r, sigma[S/m]]
                 }

# Extract domain bounds from geometry
solids = pv.read(solid_1) + pv.read(solid_2)
xmin, xmax, ymin, ymax, ZMIN, ZMAX = solids.bounds

# Number of mesh cells
Nx = 80
Ny = 80
NZ = 140

# Adjust for MPI & compute local Z-slice range
NZ -= NZ%(size) 
dz = (ZMAX - ZMIN) / NZ
Z = np.linspace(ZMIN, ZMAX, NZ+1)[:-1] + dz/2

# Allocate mpi node cells
Nz = NZ // (size)
zmin = rank * Nz * dz + ZMIN
zmax = (rank+1) * Nz * dz + ZMIN

print(f"Process {rank}: Handling Z range {zmin} to {zmax} with {Nz} cells")

grid = GridFIT3D(xmin, xmax, ymin, ymax, zmin, zmax, 
                Nx, Ny, Nz, 
                stl_solids=stl_solids, 
                stl_materials=stl_materials,
                stl_scale=1.0,
                stl_rotate=[0,0,0],
                stl_translate=[0,0,0],
                verbose=1)

# ------------ Beam source & Wake ----------------
# Beam parameters
sigmaz = 10e-2      #[m] -> 2 GHz
q = 1e-9            #[C]
beta = 1.0          # beam beta 
xs = 0.             # x source position [m]
ys = 0.             # y source position [m]
xt = 0.             # x test position [m]
yt = 0.             # y test position [m]
# [DEFAULT] tinj = 8.53*sigmaz/c_light  # injection time offset [s] 

# Simualtion
wakelength = 10. # [m]
add_space = 10   # no. cells to skip from boundaries - removes BC artifacts

from wakis.sources import Beam
from scipy.constants import c
beam = Beam(q=q, sigmaz=sigmaz, beta=beta,
            xsource=xs, ysource=ys, ti=3*sigmaz/c)

results_folder = f'003_results_n{rank}/'

'''
wake = WakeSolver(q=q, 
                  sigmaz=sigmaz, 
                  beta=beta,
                  xsource=xs, ysource=ys, 
                  xtest=xt, ytest=yt,
                  add_space=add_space, 
                  results_folder=results_folder,
                  Ez_file=results_folder+'Ez.h5')
'''

# ----------- Solver & Simulation ----------
# boundary conditions
bc_low=['pec', 'pec', 'pec']
bc_high=['pec', 'pec', 'pec']

# Solver setup
solver = SolverFIT3D(grid,
                    bc_low=bc_low, 
                    bc_high=bc_high, 
                    use_stl=True, 
                    use_mpi=True, # Activate MPI
                    bg='pec' # Background material
                    )
def plot1D_field(field, name='E', y=Ny//2, n=None, results_folder='img/', ymin=None, ymax=None,):
    if ymin is None:
        ymin = -field.max()
    if ymax is None:
        ymax = field.max()

    fig, ax = plt.subplots()
    ax.plot(field[y, :], c='g')
    ax.set_title(f'{name} at timestep={n}')
    ax.set_xlabel('z [m]')
    ax.set_ylabel('Field amplitude')
    ax.set_ylim((ymin, ymax))

    #plot vertical lines at subdomain borders
    for r in range(size):
        ax.axvline(Nz*(r+1), c='red', alpha=0.5)
        ax.axvline(Nz*(r), c='blue', alpha=0.5)

    fig.tight_layout()
    fig.savefig(results_folder+name+'1d_'+str(n).zfill(4)+'.png')

    plt.clf()
    plt.close(fig)

def plot2D_field(field, name='E', n=None, results_folder='img/', vmin=None, vmax=None,):
    extent = (ZMIN, ZMAX, ymin, ymax)
    if vmin is None:
        vmin = -field.max()
    if vmax is None:
        vmax = field.max()

    fig, ax = plt.subplots()
    im = ax.imshow(field, cmap='bwr', extent=extent, vmin=vmin, vmax=vmax)
    fig.colorbar(im, cax=make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05))
    ax.set_title(f'{name} at timestep={n}')
    ax.set_xlabel('z [m]')
    ax.set_ylabel('y [m]')

    fig.tight_layout(h_pad=0.3)
    fig.savefig(results_folder+name+str(n).zfill(4)+'.png')

    plt.clf()
    plt.close(fig)

if rank == 0:
    img_folder = '003_img/'
    if not os.path.exists(img_folder): 
        os.mkdir(img_folder)

'''
# Check global material tensors
ieps = solver.mpi_gather(solver.ieps, 'z', x=Nx//2)
sigma = solver.mpi_gather(solver.sigma, 'z', x=Nx//2)

if rank == 0:
    plot_field(ieps, 'eps^-1z_', n=0, results_folder=img_folder)
    plot_field(sigma, 'sigmaz_', n=0, results_folder=img_folder)
'''

Nt = 3000

# Plot beam current vs time
# beam.plot(np.linspace(0, solver.dt*Nt, Nt+1))

for n in tqdm(range(Nt)):

    beam.mpi_update(solver, n*solver.dt)

    solver.mpi_one_step()

    if n%20 == 0:
        Ez = solver.mpi_gather('Ez', x=Nx//2)
        if rank == 0:
            #plot2D_field(Ez, 'Ez', n=n, results_folder=img_folder, vmin=-500, vmax=500)
            plot1D_field(Ez, 'Ez', n=n, results_folder=img_folder, ymin=-800, ymax=800)
        
# Run with:
# mpiexec -n 4 python 003_MPI_wakefield_simulation.py