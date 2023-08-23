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

#----- Funtions -----#
m = 1
n = 0
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

# Plotting functions
folder ='imgRes'
def plot_E_field(solverFIT, solverFDTD, n):

    fig, axs = plt.subplots(2,3, tight_layout=True, figsize=[8,6])
    dims = {0:'x', 1:'y', 2:'z'}
    vmin, vmax = -500., 500.

    #FIT
    for i, ax in enumerate(axs[0,:]):
        #vmin, vmax = -np.max(np.abs(solverFIT.E[:,:,int(Nz/2), dims[i]])), np.max(np.abs(solverFIT.E[:,:,int(Nz/2), dims[i]]))
        im = ax.imshow(solverFIT.E[:,:,int(Nz/2), dims[i]], cmap='rainbow', vmin=vmin, vmax=vmax)
        fig.colorbar(im, cax=make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05))
        ax.set_title(f'FIT E{dims[i]}(x,y,Nz/2)')

    #FDTD
    ax = axs[1,0]
    #vmin, vmax = -np.max(np.abs(solverFDTD.Ex[:,:,int(Nz/2)])), np.max(np.abs(solverFDTD.Ex[:,:,int(Nz/2)]))
    im = ax.imshow(solverFDTD.Ex[:,:,int(Nz/2)], cmap='rainbow', vmin=vmin, vmax=vmax)
    fig.colorbar(im, cax=make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05))
    ax.set_title(f'FDTD Ex(x,y,Nz/2)')

    ax = axs[1,1]
    #vmin, vmax = -np.max(np.abs(solverFDTD.Ey[:,:,int(Nz/2)])), np.max(np.abs(solverFDTD.Ey[:,:,int(Nz/2)]))
    im = ax.imshow(solverFDTD.Ey[:,:,int(Nz/2)], cmap='rainbow', vmin=vmin, vmax=vmax)
    fig.colorbar(im, cax=make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05))
    ax.set_title(f'FDTD Ey(x,y,Nz/2)')

    ax = axs[1,2]
    #vmin, vmax = -np.max(np.abs(solverFDTD.Ez[:,:,int(Nz/2)])), np.max(np.abs(solverFDTD.Ez[:,:,int(Nz/2)]))
    im = ax.imshow(solverFDTD.Ez[:,:,int(Nz/2)], cmap='rainbow', vmin=vmin, vmax=vmax)
    fig.colorbar(im, cax=make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05))
    ax.set_title(f'FDTD Ez(x,y,Nz/2)')

    fig.suptitle(f'E field, timestep={n}')
    fig.savefig(f'{folder}E/'+str(n).zfill(4)+'.png')
    plt.clf()
    plt.close(fig)

def plot_H_field(solverFIT, solverFDTD, n):

    fig, axs = plt.subplots(2,3, tight_layout=True, figsize=[8,6])
    dims = {0:'x', 1:'y', 2:'z'}
    vmin, vmax = -1., 1. 
    #FIT
    for i, ax in enumerate(axs[0,:]):
        #vmin, vmax = -np.max(np.abs(solverFIT.H[:,:,int(Nz/2), dims[i]])), np.max(np.abs(solverFIT.H[:,:,int(Nz/2), dims[i]]))
        im = ax.imshow(solverFIT.H[:,:,int(Nz/2), dims[i]], cmap='rainbow', vmin=vmin, vmax=vmax)
        fig.colorbar(im, cax=make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05))
        ax.set_title(f'FIT H{dims[i]}(x,y,Nz/2)')

    #FDTD
    ax = axs[1,0]
    #vmin, vmax = -np.max(np.abs(solverFDTD.Hx[:,:,int(Nz/2)])), np.max(np.abs(solverFDTD.Hx[:,:,int(Nz/2)]))
    im = ax.imshow(solverFDTD.Hx[:,:,int(Nz/2)], cmap='rainbow', vmin=vmin, vmax=vmax)
    fig.colorbar(im, cax=make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05))
    ax.set_title(f'FDTD Hx(x,y,Nz/2)')

    ax = axs[1,1]
    #vmin, vmax = -np.max(np.abs(solverFDTD.Hy[:,:,int(Nz/2)])), np.max(np.abs(solverFDTD.Hy[:,:,int(Nz/2)]))
    im = ax.imshow(solverFDTD.Hy[:,:,int(Nz/2)], cmap='rainbow', vmin=vmin, vmax=vmax)
    fig.colorbar(im, cax=make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05))
    ax.set_title(f'FDTD Hy(x,y,Nz/2)')

    ax = axs[1,2]
    #vmin, vmax = -np.max(np.abs(solverFDTD.Hz[:,:,int(Nz/2)])), np.max(np.abs(solverFDTD.Hz[:,:,int(Nz/2)]))
    im = ax.imshow(solverFDTD.Hz[:,:,int(Nz/2)], cmap='rainbow', vmin=vmin, vmax=vmax)
    fig.colorbar(im, cax=make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05))
    ax.set_title(f'FDTD Hz(x,y,Nz/2)')

    fig.suptitle(f'H field, timestep={n}')
    fig.savefig(f'{folder}H/'+str(n).zfill(4)+'.png')
    plt.clf()
    plt.close(fig)

#---- Domain definition ----#

Z0 = np.sqrt(mu_0 / eps_0)

L = 1. # Domain length
N = 50 # Number of mesh cells

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
ymin = - Ly/2 + dx / 2
ymax = Ly/2 + dx / 2
zmin = - Lz/2 + dx / 2
zmax = Lz/2 + dx / 2

#Embedded cube 
lx = Lx*0.8
ly = Ly*0.8
lz = Lz*0.8
x_cent = 0
y_cent = 0
z_cent = 0

cube = InCube(lx, ly, lz, x_cent, y_cent, z_cent) #noConductor() 
conductors = ConductorsAssembly([cube])
#conductors = noConductor()

gridFIT = Grid3D(xmin, xmax, ymin, ymax, zmin, zmax, Nx, Ny, Nz, conductors, 'FIT')
tgridFIT = Grid3D(xmin + dx/2, xmax + dx/2, ymin + dy/2, ymax + dy/2, zmin + dz/2, zmax + dz/2, Nx, Ny, Nz, conductors, 'FIT')
#solverFIT = SolverFIT3D(gridFIT, 'FIT', NCFL)
solverFIT = SolverFIT3D(gridFIT, tgridFIT)

gridFDTD = Grid3D(xmin, xmax, ymin, ymax, zmin, zmax, Nx, Ny, Nz, conductors, 'FDTD')
solverFDTD = EMSolver3D(gridFDTD, 'FDTD', NCFL)

#---- Initial conditions -----#
for ii in range(Nx):
    for jj in range(Ny):
        for kk in range(Nz):

            if gridFDTD.flag_int_cell_xy[ii, jj, kk]:
                x = (ii + 0.5) * dx + xmin
                y = (jj + 0.5) * dy + ymin
                z = kk * dz + zmin
                solverFDTD.Hz[ii, jj, kk] = analytic_sol_Hz(x, y, z, -0.5 * solverFDTD.dt)

            if solverFIT.A[ii, jj, kk, 'z']:
                x = ii * dx + xmin
                y = jj * dy + ymin
                z = kk * dz + zmin
                solverFIT.H[ii, jj, kk, 'z'] = analytic_sol_Hz(x, y, z, -0.5 * solverFDTD.dt)

            if gridFDTD.flag_int_cell_zx[ii, jj, kk]:
                x = (ii + 0.5) * dx + xmin
                y = jj * dy + ymin
                z = (kk + 0.5) * dz + zmin
                solverFDTD.Hy[ii, jj, kk] = analytic_sol_Hy(x, y, z, -0.5 * solverFDTD.dt)

            if solverFIT.A[ii, jj, kk, 'y']:
                x = ii * dx + xmin
                y = jj * dy + ymin
                z = kk * dz + zmin
                solverFIT.H[ii, jj, kk, 'y'] = analytic_sol_Hy(x, y, z, -0.5 * solverFIT.dt)

            if gridFDTD.flag_int_cell_yz[ii, jj, kk]:
                x = ii*dx + xmin
                y = (jj + 0.5)*dy + ymin
                z = (kk + 0.5)*dz + zmin
                solverFDTD.Hx[ii, jj, kk] = analytic_sol_Hx(x, y, z, -0.5 * solverFDTD.dt)

            if solverFIT.A[ii, jj, kk, 'x']:
                x = ii * dx + xmin
                y = jj * dy + ymin
                z = kk * dz + zmin
                solverFIT.H[ii, jj, kk, 'x'] = analytic_sol_Hx(x, y, z, -0.5 * solverFIT.dt)

#----- Time loop -----#

Nt = 50
for nt in tqdm(range(Nt)):

    solverFIT.one_step()
    solverFDTD.one_step()

    plot_E_field(solverFIT, solverFDTD, nt)
    plot_H_field(solverFIT, solverFDTD, nt)

#----- Compare -----#

analytic = EMSolver3D(gridFDTD, 'FDTD', NCFL)

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
fig, axs = plt.subplots(3,3, tight_layout=True, figsize=[8,6])
dims = {0:'x', 1:'y', 2:'z'}

#FIT
for i, ax in enumerate(axs[0,:]):
    vmin, vmax = -np.max(np.abs(solverFIT.H[:,:,int(Nz/2), dims[i]])), np.max(np.abs(solverFIT.H[:,:,int(Nz/2), dims[i]]))
    im = ax.imshow(solverFIT.H[:,:,int(Nz/2), dims[i]], cmap='rainbow', vmin=vmin, vmax=vmax)
    fig.colorbar(im, cax=make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05))
    ax.set_title(f'FIT H{dims[i]}(x,y,Nz/2)')

#FDTD
ax = axs[1,0]
vmin, vmax = -np.max(np.abs(solverFDTD.Hx[:,:,int(Nz/2)])), np.max(np.abs(solverFDTD.Hx[:,:,int(Nz/2)]))
im = ax.imshow(solverFDTD.Hx[:,:,int(Nz/2)], cmap='rainbow', vmin=vmin, vmax=vmax)
fig.colorbar(im, cax=make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05))
ax.set_title(f'FDTD Hx(x,y,Nz/2)')

ax = axs[1,1]
vmin, vmax = -np.max(np.abs(solverFDTD.Hy[:,:,int(Nz/2)])), np.max(np.abs(solverFDTD.Hy[:,:,int(Nz/2)]))
im = ax.imshow(solverFDTD.Hy[:,:,int(Nz/2)], cmap='rainbow', vmin=vmin, vmax=vmax)
fig.colorbar(im, cax=make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05))
ax.set_title(f'FDTD Hy(x,y,Nz/2)')

ax = axs[1,2]
vmin, vmax = -np.max(np.abs(solverFDTD.Hz[:,:,int(Nz/2)])), np.max(np.abs(solverFDTD.Hz[:,:,int(Nz/2)]))
im = ax.imshow(solverFDTD.Hz[:,:,int(Nz/2)], cmap='rainbow', vmin=vmin, vmax=vmax)
fig.colorbar(im, cax=make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05))
ax.set_title(f'FDTD Hz(x,y,Nz/2)')

#Analytic
ax = axs[2,0]
vmin, vmax = -np.max(np.abs(analytic.Hx[:,:,int(Nz/2)])), np.max(np.abs(analytic.Hx[:,:,int(Nz/2)]))
im = ax.imshow(analytic.Hx[:,:,int(Nz/2)], cmap='rainbow', vmin=vmin, vmax=vmax)
fig.colorbar(im, cax=make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05))
ax.set_title(f'Analytic Hx(x,y,Nz/2)')

ax = axs[2,1]
vmin, vmax = -np.max(np.abs(analytic.Hy[:,:,int(Nz/2)])), np.max(np.abs(analytic.Hy[:,:,int(Nz/2)]))
im = ax.imshow(analytic.Hy[:,:,int(Nz/2)], cmap='rainbow', vmin=vmin, vmax=vmax)
fig.colorbar(im, cax=make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05))
ax.set_title(f'Analytic Hy(x,y,Nz/2)')

ax = axs[2,2]
vmin, vmax = -np.max(np.abs(analytic.Hz[:,:,int(Nz/2)])), np.max(np.abs(analytic.Hz[:,:,int(Nz/2)]))
im = ax.imshow(analytic.Hz[:,:,int(Nz/2)], cmap='rainbow', vmin=vmin, vmax=vmax)
fig.colorbar(im, cax=make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05))
ax.set_title(f'Analytic Hz(x,y,Nz/2)')

fig.suptitle(f'H field, timestep={Nt}')
fig.savefig('analytic.png')
plt.clf()
plt.close(fig)
