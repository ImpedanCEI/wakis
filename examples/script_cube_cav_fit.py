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
from conductors3d import noConductor
from scipy.special import jv
from field import Field 

Z0 = np.sqrt(mu_0 / eps_0)

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

xmin = -Lx/2 + dx / 2
xmax = Lx/2 + dx / 2
ymin = - Ly/2 + dx / 2
ymax = Ly/2 + dx / 2
zmin = - Lz/2 + dx / 2
zmax = Lz/2 + dx / 2

conductors = noConductor()

# conductors = cube
sol_type = 'FIT'
NCFL = 1

grid = Grid3D(xmin, xmax, ymin, ymax, zmin, zmax, Nx, Ny, Nz, conductors, sol_type)
solverFIT = SolverFIT3D(grid, sol_type, NCFL)

gridFDTD = Grid3D(xmin, xmax, ymin, ymax, zmin, zmax, Nx, Ny, Nz, conductors, 'FDTD')
solverFDTD = EMSolver3D(gridFDTD, 'FDTD', NCFL)

solverFIT.J[int(Nx/2), int(Ny/2), int(Nz/2), 'z'] = 1.0*c_light
solverFDTD.Jz[int(Nx/2)+1, int(Ny/2)+1,  int(Nz/2)] = 1.0*c_light

Nt = 50

def plot_E_field(solverFIT, solverFDTD, n):

    fig, axs = plt.subplots(2,3, tight_layout=True, figsize=[8,6])
    dims = {0:'x', 1:'y', 2:'z'}

    #FIT
    for i, ax in enumerate(axs[0,:]):
        im = ax.imshow(solverFIT.E[:,:,int(Nz/2), dims[i]], cmap='rainbow')
        fig.colorbar(im, cax=make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05))
        ax.set_title(f'FIT E{dims[i]}(x,y,Nz/2)')

    #FDTD
    ax = axs[1,0]
    im = ax.imshow(solverFDTD.Ex[:,:,int(Nz/2)], cmap='rainbow')
    fig.colorbar(im, cax=make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05))
    ax.set_title(f'FDTD Ex(x,y,Nz/2)')

    ax = axs[1,1]
    im = ax.imshow(solverFDTD.Ey[:,:,int(Nz/2)], cmap='rainbow')
    fig.colorbar(im, cax=make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05))
    ax.set_title(f'FDTD Ey(x,y,Nz/2)')

    ax = axs[1,2]
    im = ax.imshow(solverFDTD.Ez[:,:,int(Nz/2)], cmap='rainbow')
    fig.colorbar(im, cax=make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05))
    ax.set_title(f'FDTD Ez(x,y,Nz/2)')

    fig.suptitle(f'E field, timestep={n}')
    fig.savefig('imgE/'+str(n).zfill(4)+'.png')
    plt.clf()
    plt.close(fig)

def plot_H_field(solverFIT, solverFDTD, n):

    fig, axs = plt.subplots(2,3, tight_layout=True, figsize=[8,6])
    dims = {0:'x', 1:'y', 2:'z'}

    #FIT
    for i, ax in enumerate(axs[0,:]):
        im = ax.imshow(solverFIT.H[:,:,int(Nz/2), dims[i]], cmap='rainbow')
        fig.colorbar(im, cax=make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05))
        ax.set_title(f'FIT H{dims[i]}(x,y,Nz/2)')

    #FDTD
    ax = axs[1,0]
    im = ax.imshow(solverFDTD.Hx[:,:,int(Nz/2)], cmap='rainbow')
    fig.colorbar(im, cax=make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05))
    ax.set_title(f'FDTD Hx(x,y,Nz/2)')

    ax = axs[1,1]
    im = ax.imshow(solverFDTD.Hy[:,:,int(Nz/2)], cmap='rainbow')
    fig.colorbar(im, cax=make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05))
    ax.set_title(f'FDTD Hy(x,y,Nz/2)')

    ax = axs[1,2]
    im = ax.imshow(solverFDTD.Hz[:,:,int(Nz/2)], cmap='rainbow')
    fig.colorbar(im, cax=make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05))
    ax.set_title(f'FDTD Hz(x,y,Nz/2)')

    fig.suptitle(f'H field, timestep={n}')
    fig.savefig('imgH/'+str(n).zfill(4)+'.png')
    plt.clf()
    plt.close(fig)

for n in tqdm(range(Nt)):
    solverFIT.one_step()
    solverFDTD.one_step()
    fig, (ax1, ax2) = plt.subplots(1,2, tight_layout=True)

    #solverFIT.H[int(Nx/2), int(Ny/2), :, 'z'] = solverFIT.H[int(Nx/2), int(Ny/2), :, 'z']+c_light
    #solverFDTD.Hz[int(Nx/2), int(Ny/2), :] += c_light

    #plot_field = Field(Nx, Ny, Nz)
    #plot_field.fromarray(solverFIT.dt*(solverFIT.C0.transpose()*solverFIT.H.toarray()-solverFIT.J.toarray()))

    plot_E_field(solverFIT, solverFDTD, n)
    plot_H_field(solverFIT, solverFDTD, n)

# Analytic solution
'''
m = 1
n = 0
p = 1


def analytic_sol_Hz(x, y, z, t):
    Rm = np.array([[np.cos(-theta), - np.sin(-theta), 0],[np.sin(-theta), np.cos(-theta), 0], [0, 0, 1]])
    [x_0, y_0, z_0] = np.dot(Rm, np.array([x, y, z]))

    return np.cos(m * np.pi / lx * (x_0 - lx/2)) * np.cos(n * np.pi / ly * (y_0 - ly/2)) * np.sin(
        p * np.pi / lz * (z_0 - lz/2)) * np.cos(np.sqrt(2) * np.pi / lx * c_light * t)


h_2 = (m * np.pi / lx) ** 2 + (n * np.pi / ly) ** 2 + (p * np.pi / lz) ** 2


def analytic_sol_Hy(x, y, z, t):
    Rm = np.array([[np.cos(-theta), - np.sin(-theta), 0],[np.sin(-theta), np.cos(-theta), 0], [0, 0, 1]])
    [x_0, y_0, z_0] = np.dot(Rm, np.array([x, y, z]))
    return -2 / h_2 * (n * np.pi / ly) * (p * np.pi / lz) * np.cos(m * np.pi / lx * (x_0 - lx/2)) * np.sin(
        n * np.pi / ly * (y_0 - ly/2)) * np.cos(p * np.pi / lz * (z_0 - lz/2)) * np.cos(
        np.sqrt(2) * np.pi / lx * c_light * t)
    # return -0.9994120256621584*np.cos(m*np.pi/lx*(x_0 + lx/2))*np.sin(n*np.pi/ly*(y_0 + ly/2))*np.cos(p*np.pi/lz*(z_0 + lz/2))*np.cos(np.sqrt(2)*np.pi/lx*c_light*t)


def analytic_sol_Hx(x, y, z, t):
    Rm = np.array([[np.cos(-theta), - np.sin(-theta), 0],[np.sin(-theta), np.cos(-theta), 0], [0, 0, 1]])
    [x_0, y_0, z_0] = np.dot(Rm, np.array([x, y, z]))

    return -2 / h_2 * (m * np.pi / lx) * (p * np.pi / lz) * np.sin(m * np.pi / lx * (x_0 - lx/2)) * np.cos(
        n * np.pi / ly * (y_0 - ly/2)) * np.cos(p * np.pi / lz * (z_0 - lz/2)) * np.cos(
        np.sqrt(2) * np.pi / lx * c_light * t)


k = 2.7437 / r_circ

for ii in range(Nx):
    for jj in range(Ny):
        for kk in range(Nz):
            if grid.flag_int_cell_xy[ii, jj, kk]:  # and not grid.flag_bound_cell_xy[ii, jj, kk]:
                x = (ii + 0.5) * dx + xmin
                y = (jj + 0.5) * dy + ymin
                z = kk * dz + zmin
                solver.Hz[ii, jj, kk] = analytic_sol_Hz(x, y, z, -0.5 * solver.dt)

            if grid.flag_int_cell_zx[ii, jj, kk]:
                x = (ii + 0.5) * dx + xmin
                y = jj * dy + ymin
                z = (kk + 0.5) * dz + zmin
                solver.Hy[ii, jj, kk] = analytic_sol_Hy(x, y, z, -0.5 * solver.dt)

            if grid.flag_int_cell_yz[ii, jj, kk]:
                x = ii*dx + xmin
                y = (jj + 0.5)*dy + ymin
                z = (kk + 0.5)*dz + zmin
                solver.Hx[ii, jj, kk] = analytic_sol_Hx(x, y, z, -0.5 * solver.dt)


a = lx*np.sqrt(2)/3
Tf = 1.1*np.sqrt(2)*(a/c_light)
Nt = int(Tf/solver.dt)

res_Hy = np.zeros(Nt)
res_Ex = np.zeros(Nt)
res_Hz = np.zeros(Nt)

fields_norms = True

for t in tqdm(range(Nt)):
    if fields_norms:
        fig, axs = plt.subplots(1, 2, figsize=(16, 8))
        fig.subplots_adjust(left=0.05, bottom=0.1, right=0.97, top=0.94, wspace=0.15)
        norme = np.sqrt(np.square(solver.Ex[:, :, k_s]) + np.square(solver.Ey[:, :, k_s]) + np.square(solver.Ez[:, :, k_s]))
        normh = np.sqrt(np.square(solver.Hx[:, :, k_s]) + np.square(solver.Hy[:, :, k_s]) + np.square(solver.Hz[:, :, k_s]))

        im1 = axs[0].imshow(norme, cmap='jet', vmax=530, vmin=-530)  # ,extent=[0, L , 0, L ])
        axs[0].set_xlabel('x [m]')
        axs[0].set_ylabel('y [m]')
        axs[0].set_title('||E|| [V/m]')
        fig.colorbar(im1, ax=axs[0])
        im1 = axs[1].imshow(normh, cmap='jet', vmax=1, vmin=-1)
        axs[1].set_xlabel('x [m]')
        axs[1].set_ylabel('y [m]')
        axs[1].set_title('||H|| [V/m]')
        fig.colorbar(im1, ax=axs[1])
    else:
        fig, axs = plt.subplots(2, 3, figsize=(16, 10))
        fig.subplots_adjust(left=0.05, bottom=0.1, right=0.97, top=0.94, wspace=0.15)

        im1 = axs[0, 0].imshow(solver.Ex[:, :, k_s], cmap='jet', vmax=530, vmin=-530)  # ,extent=[0, L , 0, L ])
        axs[0, 0].set_xlabel('x [m]')
        axs[0, 0].set_ylabel('y [m]')
        axs[0, 0].set_title('Ex [V/m]')
        fig.colorbar(im1, ax=axs[0, 0], )
        im1 = axs[0, 1].imshow(solver.Ey[:, :, k_s], cmap='jet', vmax=530, vmin=-530)
        axs[0, 1].set_xlabel('x [m]')
        axs[0, 1].set_ylabel('y [m]')
        axs[0, 1].set_title('Ey [V/m]')
        fig.colorbar(im1, ax=axs[0, 1])
        im1 = axs[0, 2].imshow(solver.Ez[:, :, k_s], cmap='jet', vmax=530, vmin=-530)
        axs[0, 2].set_xlabel('x [m]')
        axs[0, 2].set_ylabel('y [m]')
        axs[0, 2].set_title('Ez [V/m]')
        fig.colorbar(im1, ax=axs[0, 2])
        im1 = axs[1, 0].imshow(solver.Hx[:, :, k_s], cmap='jet', vmax=0.1, vmin=-0.1)  # ,extent=[0, L , 0, L ])
        axs[1, 0].set_xlabel('x [m]')
        axs[1, 0].set_ylabel('y [m]')
        axs[1, 0].set_title('Hx [A/m]')
        fig.colorbar(im1, ax=axs[1, 0], )
        im1 = axs[1, 1].imshow(solver.Hy[:, :, k_s], cmap='jet', vmax=0.17, vmin=-0.17)
        axs[1, 1].set_xlabel('x [m]')
        axs[1, 1].set_ylabel('y [m]')
        axs[1, 1].set_title('Hy [A/m]')
        fig.colorbar(im1, ax=axs[1, 1])
        im1 = axs[1, 2].imshow(solver.Hz[:, :, k_s], cmap='jet', vmax=1, vmin=-1)
        axs[1, 2].set_xlabel('x [m]')
        axs[1, 2].set_ylabel('y [m]')
        axs[1, 2].set_title('Hz [A/m]')
        fig.colorbar(im1, ax=axs[1, 2])
    plt.suptitle(str(solver.time))
    
    folder = sol_type + '_images_cube'
    if not os.path.exists(folder):
        os.mkdir(folder)

    filename = folder + '/%d.png' % t
    plt.savefig(filename)
    plt.close(fig) 

    res_Hy[t] = solver.Hy[i_s, j_s, 5]
    res_Hz[t] = solver.Hz[i_s, 5, k_s]
    res_Ex[t] = solver.Ex[i_s, j_s, k_s]

    solver.one_step()
    # solver.Jx[i_s, j_s, k_s] = solver.gauss(solver.time)
    # solver.Hz[i_s, j_s, k_s] += solver.gauss(solver.time)
    # solver.Jz[i_s, j_s, k_s] = solver.gauss(solver.time)

fig, axs = plt.subplots(2, 3, figsize=(16, 10))
fig.subplots_adjust(left=0.05, bottom=0.1, right=0.97, top=0.94, wspace=0.15)
axs[0, 0].imshow(grid.flag_int_cell_xy[:, :, k_s])
axs[0, 1].imshow(grid.Syz[:, :, k_s])
axs[0, 2].imshow(grid.Szx[:, :, k_s])
axs[1, 0].imshow(solver.Hz[:, :, k_s], cmap='jet')
axs[1, 1].imshow(solver.Hx[:, :, k_s], cmap='jet')
axs[1, 2].imshow(solver.Hy[:, :, k_s], cmap='jet')
'''
