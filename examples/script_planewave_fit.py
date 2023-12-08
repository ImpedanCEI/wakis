import numpy as np
import matplotlib.pyplot as plt
import os, sys
import pyvista as pv
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.constants import c as c_light, epsilon_0 as eps_0, mu_0 as mu_0

sys.path.append('../')

from solverFIT3D import SolverFIT3D
from gridFIT3D import GridFIT3D 
from field import Field 


# ---------- Domain setup ---------
# Number of mesh cells
Nx = 80
Ny = 80
Nz = 160

# Embedded boundaries
stl_file = 'stl/sphere.stl' 
surf = pv.read(stl_file)

stl_solids = {'Solid 1': stl_file}
stl_materials = {'Solid 1': 'dielectric'}
stl_rotate = [0, 0, 0]
stl_scale = 1e-3

surf = surf.rotate_x(stl_rotate[0])
surf = surf.rotate_y(stl_rotate[1])
surf = surf.rotate_z(stl_rotate[2])
surf = surf.scale(stl_scale)
#surf.plot()

# Domain bounds
xmin, xmax, ymin, ymax, zmin, zmax = surf.bounds
padx, pady, padz = (xmax-xmin)*0.2, (ymax-ymin)*0.2, (zmax-zmin)*1.0

xmin, ymin, zmin = (xmin-padx), (ymin-pady), (zmin-padz)
xmax, ymax, zmax = (xmax+padx), (ymax+pady), (zmax+padz)

Lx, Ly, Lz = (xmax-xmin), (ymax-ymin), (zmax-zmin)

# boundary conditions
bc_low=['pec', 'pec', 'pec']
bc_high=['pec', 'pec', 'pec']

# set FIT solver
grid = GridFIT3D(xmin, xmax, ymin, ymax, zmin, zmax, Nx, Ny, Nz, 
                stl_solids=stl_solids, 
                stl_rotate=stl_rotate,
                stl_scale=stl_scale,
                stl_materials=stl_materials)

solver = SolverFIT3D(grid, 
                     bc_low=bc_low, bc_high=bc_high, 
                     use_stl=True)

# ------------ Plot functions ---------------
plane = 'YZ'

if plane == 'XY':
    x, y, z = slice(0,Nx), slice(0,Ny), int(Nz//2) #plane XY
    title = '(x,y,Nz/2)'
    xax, yax = 'y', 'x'
    extent = [0,Ny,0,Nx]

if plane == 'YZ':
    x, y, z = int(Nx//2), slice(0,Ny), slice(0,Nz) #plane YZ
    title = '(Nx/2,y,z)'
    xax, yax = 'z', 'y'
    extent = [0,Nz,0,Ny]
    extent = [zmin, zmax, ymin, ymax]

def plot_E_field(solver, n, plot_patch=True):

    fig, ax = plt.subplots(1,1, figsize=[8,4])
    vmin, vmax = -2.e3, 2.e3

    #im = ax.imshow(solver.E.get_abs()[x, y, z], cmap='jet', norm='log', extent=extent, origin='lower', vmin=1e-2)#, vmax=vmax,)
    im = ax.imshow(solver.E[x, y, z, 'x'], cmap='jet', extent=extent, origin='lower', vmin=vmin, vmax=vmax,)
    fig.colorbar(im, cax=make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05))
    ax.set_title(f'FIT Ex{title}, timestep={n}')
    ax.set_xlabel(xax)
    ax.set_ylabel(yax)

    # Patch stl
    if plot_patch:
        mask = np.reshape(grid.grid['Solid 1'], (Nx, Ny, Nz))
        patch = np.ones((Nx, Ny, Nz))
        patch[np.logical_not(mask)] = np.nan
        ax.imshow(patch[x,y,z], cmap='Greys', extent=extent, origin='lower', alpha=0.2)

    #fig.suptitle(f'Abs(E) field, timestep={n}')
    fig.tight_layout()
    fig.savefig('imgE/'+str(n).zfill(4)+'.png')
    plt.clf()
    plt.close(fig)

def plot_H_field(solver, n, plot_patch=True):

    fig, ax = plt.subplots(1,1, figsize=[8,4])
    vmin, vmax = -1., 1.
    
    #im = ax.imshow(solver.H.get_abs()[x, y, z], cmap='jet', norm='log', extent=extent, origin='lower', vmin=1e-2)#, vmax=vmax)
    im = ax.imshow(solver.H[x, y, z, 'y'], cmap='jet', extent=extent, origin='lower', vmin=vmin, vmax=vmax)
    fig.colorbar(im, cax=make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05))
    ax.set_title(f'FIT Hy{title}, timestep={n}')
    ax.set_xlabel(xax)
    ax.set_ylabel(yax)

    # Patch stl
    if plot_patch:
        mask = np.reshape(grid.grid['Solid 1'], (Nx, Ny, Nz))
        patch = np.ones((Nx, Ny, Nz))
        patch[np.logical_not(mask)] = np.nan
        ax.imshow(patch[x,y,z], cmap='Greys', extent=extent, origin='lower', alpha=0.2)
    
    #fig.suptitle(f'Abs(H) field, timestep={n}')
    fig.tight_layout()
    fig.savefig('imgH/'+str(n).zfill(4)+'.png')
    plt.clf()
    plt.close(fig)

def plot3D_Hy_field(solver, n):
    # Add H field data
    grid.grid.cell_data['Hy'] = np.reshape(solver.H[:, :, :, 'y'], solver.N)
    points= grid.grid.cell_data_to_point_data() #interpolate
    clim = [-2., 2.]

    # Pyvista plot
    ac1 = pl.add_mesh(points.clip_box(bounds=[xmax-Lx/2, xmax, ymax-Ly/2, ymax, zmin, zmax]), scalars='Hy', cmap='jet', clim=clim)

    # Save
    #pl.save_graphic('imgH/'+str(n).zfill(4)+'_3D.pdf')
    pl.screenshot('imgH/'+str(n).zfill(4)+'_3D.png')
    pl.remove_actor(ac1)
    

# ------------ Time loop ----------------

# Initial conditions
def plane_wave(solver,t, Nt,f=None, beta=1.0):

    if f is None:
        f = 15 * 1/(solver.dt*(Nt-1))    # 15 nodes
    vp = beta*c_light       # wavefront velocity beta*c
    w = 2*np.pi*f           # ang. frequency  
    kz = w/c_light          # wave number 

    solver.H[:,:,0,'y'] = -1.0 * np.cos(w*t) 
    solver.E[:,:,0,'x'] = 1.0 * np.cos(w*t) /(kz/(mu_0*vp)) 


#Nt = 300
Nt = int((zmax-zmin)/(solver.dt*c_light))

'''
pl = pv.Plotter(off_screen=True)
pl.add_mesh(surf, color='white', opacity=0.3)
pl.add_axes()
#pl.enable_3_lights()
pl.camera_position = 'zx'
pl.camera.azimuth += 30
pl.camera.elevation += 30
pl.camera.zoom(0.5)
'''
Nt = 10

for n in tqdm(range(Nt)):

    # Initial condition
    plane_wave(solver, n*solver.dt, Nt)

    # Advance
    solver.one_step()

    # Plot
    #if n%5 == 0:
        #plot_E_field(solver, n)
        #plot_H_field(solver, n)
    
    # Plot 3D
    #if n%10 == 0:
        #plot3D_Hy_field(solver, n)

# Plot 3D built-in
solver.plot3D(field='H', component='y', clim=None,  hide_solids=None, show_solids=None, 
               add_stl='Solid 1', stl_opacity=0.1, stl_colors='white',
               title=None, cmap='rainbow', clip_volume=False, clip_normal='-y',
               clip_box=True, clip_bounds=None, off_screen=True, zoom=0.4, n=n)

solver.plot2D(field='E', component='x', plane='ZY', pos=0.5, norm='symlog', 
               vmin=None, vmax=None, figsize=[8,4], cmap='jet', patch_alpha=0.1,
               add_patch=False, title=None, off_screen=True, n=n, interpolation='spline36')