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

print(f"Process {rank} of {size} is running")

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
xmin, xmax, ymin, ymax, zmin, zmax = solids.bounds

# Number of mesh cells
Nx = 80
Ny = 80
Nz = 141

# Adjust for MPI & ompute local Z-slice range
Nz += Nz%(size) #if rank 0 is common, then size-1
dz = (zmax - zmin) / Nz
z = np.linspace(zmin, zmax, Nz+1)

# Allocate mpi node cells
Nz_mpi = Nz // (size) 
zmin_mpi = rank * Nz_mpi * dz + zmin
zmax_mpi= (rank+1) * Nz_mpi * dz + zmin

print(f"Process {rank}: Handling Z range {zmin_mpi} to {zmax_mpi}")

grid = GridFIT3D(xmin, xmax, ymin, ymax, 
                zmin_mpi, zmax_mpi, 
                Nx, Ny, Nz_mpi, 
                stl_solids=stl_solids, 
                stl_materials=stl_materials,
                stl_scale=1.0,
                stl_rotate=[0,0,0],
                stl_translate=[0,0,0],
                verbose=1)

# ------------ Beam source & Wake ----------------
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

results_folder = f'001_results_n{rank}/'

'''
wake = WakeSolver(q=q, 
                  sigmaz=sigmaz, 
                  beta=beta,
                  xsource=xs, ysource=ys, 
                  xtest=xt, ytest=yt,
                  add_space=add_space, 
                  results_folder=results_folder,
                  Ez_file=results_folder+'001_Ez.h5')
'''

# ----------- Solver & Simulation ----------
# boundary conditions
bc_low=['pec', 'pec', 'pec']
bc_high=['pec', 'pec', 'pec']

if rank > 0:
    bc_low=['pec', 'pec', 'periodic']

if rank < size - 1:
    bc_high=['pec', 'pec', 'periodic']

# Solver setup
solver = SolverFIT3D(grid,
                    bc_low=bc_low, 
                    bc_high=bc_high, 
                    use_stl=True, 
                    bg='pec' # Background material
                    )
# Communication between ghost cells
def communicate_ghost_cells():
    if rank > 0:
        for d in ['x','y','z']:
            comm.Sendrecv(solver.E[:, :, 1,d], 
                        recvbuf=solver.E[:, :, 0,d],
                        dest=rank-1, sendtag=0,
                        source=rank-1, recvtag=1)

            comm.Sendrecv(solver.H[:, :, 1,d], 
                        recvbuf=solver.H[:, :, 0,d],
                        dest=rank-1, sendtag=0,
                        source=rank-1, recvtag=1)
            
            comm.Sendrecv(solver.J[:, :, 1,d], 
                        recvbuf=solver.J[:, :, 0,d],
                        dest=rank-1, sendtag=0,
                        source=rank-1, recvtag=1)
            
    if rank < size - 1:
        for d in ['x','y','z']:
            comm.Sendrecv(solver.E[:, :, -2,d], 
                          recvbuf=solver.E[:, :, -1, d], 
                          dest=rank+1, sendtag=1,
                          source=rank+1, recvtag=0)
            
            comm.Sendrecv(solver.H[:, :, -2,d], 
                          recvbuf=solver.H[:, :, -1, d], 
                          dest=rank+1, sendtag=1,
                          source=rank+1, recvtag=0)
            
            comm.Sendrecv(solver.J[:, :, -2,d], 
                          recvbuf=solver.J[:, :, -1, d], 
                          dest=rank+1, sendtag=1,
                          source=rank+1, recvtag=0)

def compose_field(field, d='z'):

    if field == 'E':
        local = solver.E[Nx//2, :, :,d].ravel()
    elif field == 'H':
        local = solver.H[Nx//2, :, :,d].ravel()
    elif field == 'J':
        local = solver.J[Nx//2, :, :,d].ravel()

    buffer = comm.gather(local, root=0)
    field = None

    if rank == 0:
        field = np.zeros((Ny, Nz))  # Reinitialize global array
        for r in range(size):
            field[:, r*Nz_mpi:(r+1)*Nz_mpi] = np.reshape(buffer[r], (Ny, Nz_mpi))
    
    return field

def plot_field(field, name='E', n=None, results_folder='img/'):
    extent = (zmin, zmax, ymin, ymax)
    fig, ax = plt.subplots()
    im = ax.imshow(field, cmap='bwr', extent=extent, vmin=-500, vmax=500)
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

Nt = 3000
for n in tqdm(range(Nt)):

    beam.update_mpi(solver, n*solver.dt, zmin)

    solver.one_step()

    #Communicate slices
    communicate_ghost_cells()

    if n%20 == 0:
        Ez = compose_field('E')
        if rank == 0:
            plot_field(Ez, 'Ez', n=n, results_folder=img_folder)

        
# Run with:
# mpiexec -n 4 python 003_MPI_wakefield_simulation.py
