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

#----- TE Funtions -----#
m = 0
n = 1
p = 1
theta = 0 #np.pi/8

# Analytic solution of cubic resonator
# Ref: http://faculty.pccu.edu.tw/~meng/new%20EM6.pdf pp.20/24

def analytic_sol_Hz(x, y, z, t):
    Rm = np.array([[np.cos(-theta), - np.sin(-theta), 0],[np.sin(-theta), np.cos(-theta), 0], [0, 0, 1]])
    [x_0, y_0, z_0] = np.dot(Rm, np.array([x, y, z]))

    return np.cos(m * np.pi / Lx * (x_0 - Lx/2)) * np.cos(n * np.pi / Ly * (y_0 - Ly/2)) * np.sin(
        p * np.pi / Lz * (z_0 - Lz/2)) * np.cos(np.sqrt(2) * np.pi / Lx * c_light * t)

def analytic_sol_Hy(x, y, z, t):
    Rm = np.array([[np.cos(-theta), - np.sin(-theta), 0],[np.sin(-theta), np.cos(-theta), 0], [0, 0, 1]])
    [x_0, y_0, z_0] = np.dot(Rm, np.array([x, y, z]))
    h_2 = (m * np.pi / Lx) ** 2 + (n * np.pi / Ly) ** 2 + (p * np.pi / Lz) ** 2

    return -2 / h_2 * (n * np.pi / Ly) * (p * np.pi / Lz) * np.cos(m * np.pi / Lx * (x_0 - Lx/2)) * np.sin(
        n * np.pi / Ly * (y_0 - Ly/2)) * np.cos(p * np.pi / Lz * (z_0 - Lz/2)) * np.cos(
        np.sqrt(2) * np.pi / Lx * c_light * t)

def analytic_sol_Hx(x, y, z, t):
    Rm = np.array([[np.cos(-theta), - np.sin(-theta), 0],[np.sin(-theta), np.cos(-theta), 0], [0, 0, 1]])
    [x_0, y_0, z_0] = np.dot(Rm, np.array([x, y, z]))
    h_2 = (m * np.pi / Lx) ** 2 + (n * np.pi / Ly) ** 2 + (p * np.pi / Lz) ** 2

    return -2 / h_2 * (m * np.pi / Lx) * (p * np.pi / Lz) * np.sin(m * np.pi / Lx * (x_0 - Lx/2)) * np.cos(
        n * np.pi / Ly * (y_0 - Ly/2)) * np.cos(p * np.pi / Lz * (z_0 - Lz/2)) * np.cos(
        np.sqrt(2) * np.pi / Lx * c_light * t)


#---- Domain definition ----#

Z0 = np.sqrt(mu_0 / eps_0)

L = 1. # Domain length
N = 30 # Number of mesh cells

Nx = N
Ny = N
Nz = N
Lx = L
Ly = L
Lz = L
dx = L / Nx
dy = L / Ny
dz = L / Nz

xmin = -Lx/2 + dx / 2
xmax = Lx/2 + dx / 2
ymin = - Ly/2 + dy / 2
ymax = Ly/2 + dy / 2
zmin = - Lz/2 + dz / 2
zmax = Lz/2 + dz / 2

# ---- Solver definitions ---------#

conductors = noConductor()
bc_low=['pec', 'pec', 'pec']
bc_high=['pec', 'pec', 'pec']

NCFL=1.0

gridFIT = Grid3D(xmin, xmax, ymin, ymax, zmin, zmax, Nx, Ny, Nz, conductors, 'FIT')
tgridFIT = Grid3D(xmin + dx/2, xmax + dx/2, ymin + dy/2, ymax + dy/2, zmin + dz/2, zmax + dz/2, Nx, Ny, Nz, conductors, 'FIT')
#solverFIT = SolverFIT3D(gridFIT, bc_low=bc_low, bc_high=bc_high)
solverFIT = SolverFIT3D(gridFIT, tgridFIT, bc_low=bc_low, bc_high=bc_high)

#---- Initial conditions ------------#

for ii in range(Nx):
    for jj in range(Ny):
        for kk in range(Nz):

            x = ii * dx + xmin
            y = jj * dy + ymin
            z = kk * dz + zmin
            solverFIT.H[ii, jj, kk, 'z'] = analytic_sol_Hz(x, y, z, -0.5 * solverFIT.dt)

            x = ii * dx + xmin
            y = jj * dy + ymin
            z = kk * dz + zmin
            solverFIT.H[ii, jj, kk, 'y'] = analytic_sol_Hy(x, y, z, -0.5 * solverFIT.dt)

            x = ii * dx + xmin
            y = jj * dy + ymin
            z = kk * dz + zmin
            solverFIT.H[ii, jj, kk, 'x'] = analytic_sol_Hx(x, y, z, -0.5 * solverFIT.dt)

#----- Time loop -----#

Nt = 50
for nt in tqdm(range(Nt)):
    solverFIT.one_step()

#----- Compare -----#

analytic = EMSolver3D(gridFIT, 'FDTD', NCFL)

for ii in range(Nx):
    for jj in range(Ny):
        for kk in range(Nz):

            x = (ii + 0.5) * dx + xmin
            y = (jj + 0.5) * dy + ymin
            z = kk * dz + zmin
            analytic.Hz[ii, jj, kk] = analytic_sol_Hz(x, y, z, (Nt-0.5) * analytic.dt)

            x = (ii + 0.5) * dx + xmin
            y = jj * dy + ymin
            z = (kk + 0.5) * dz + zmin
            analytic.Hy[ii, jj, kk] = analytic_sol_Hy(x, y, z, (Nt-0.5) * analytic.dt)

            x = ii*dx + xmin
            y = (jj + 0.5)*dy + ymin
            z = (kk + 0.5)*dz + zmin
            analytic.Hx[ii, jj, kk] = analytic_sol_Hx(x, y, z, (Nt-0.5) * analytic.dt)

# Plot fields
planes = ['XY', 'YZ', 'XZ']

fig, axs = plt.subplots(3,3, tight_layout=True, figsize=[8,6])
dims = {0:'x', 1:'y', 2:'z'}


for j, plane in enumerate(planes):
    if plane == 'YZ':
        xx, yy, zz = int(Nx//2), slice(0,Ny), slice(0,Nz) #plane YZ
        title = '(Nx/2,y,z)'
        xax, yax = 'z', 'y'

    if plane == 'XY':
        xx, yy, zz = slice(0,Nx), slice(0,Ny), int(Nz//2) #plane XY
        title = '(x,y,Nz/2)'
        xax, yax = 'y', 'x'

    if plane == 'XZ':
        xx, yy, zz = slice(0,Nx), int(Ny//2), slice(0,Nz)  #plane ZX
        title = '(x,Nz/2,z)'
        xax, yax = 'z', 'x'

    field_an = {0: analytic.Hx[xx, yy, zz],
              1: analytic.Hy[xx, yy, zz], 
              2: analytic.Hz[xx, yy, zz]
              }

    for i, ax in enumerate(axs[j,:]):

        field2plot = (solverFIT.H[xx, yy, zz, dims[i]] - field_an[i])/np.max(np.abs(field_an[i]))

        im = ax.imshow(field2plot, cmap='bwr', vmin=-1., vmax=1.)
        fig.colorbar(im, label=r'$\varepsilon_{rel}$', cax=make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05))
        ax.set_title(f'FIT H{dims[i]}{title}')
        ax.set_xlabel(xax)
        ax.set_ylabel(yax)

fig.suptitle(f'H field, timestep={Nt}')
fig.savefig(f'imgResH/diff_TE{m}{n}{p}.png')
plt.show()