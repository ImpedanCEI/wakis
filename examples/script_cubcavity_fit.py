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
Nx = 50
Ny = 50
Nz = 150

# Embedded boundaries
stl_file = 'stl/cubcavity.stl' 
surf = pv.read(stl_file)

stl_solids = {'Cavity': stl_file}
stl_materials = {'Cavity': 'vacuum'}

# Domain bounds
xmin, xmax, ymin, ymax, zmin, zmax = surf.bounds
Lx, Ly, Lz = (xmax-xmin), (ymax-ymin), (zmax-zmin)

# boundary conditions
bc_low=['pec', 'pec', 'pec']
bc_high=['pec', 'pec', 'pec']

# set FIT solver
grid = GridFIT3D(xmin, xmax, ymin, ymax, zmin, zmax, Nx, Ny, Nz, 
                stl_solids=stl_solids, 
                stl_materials=stl_materials)

solver = SolverFIT3D(grid, 
                     bc_low=bc_low, bc_high=bc_high, 
                     use_stl=True, bg='pec')
    
# ------------ Beam source ----------------
# Beam parameters
sigmaz = 18.5e-3    #[m]
q = 1e-9            #[C]
beta = 1.0          # beam beta
xs = 0.             # x source position [m]
ys = 0.             # y source position [m]
xt = 0.             # x test position [m]
yt = 0.             # y test position [m]
tinj = 8.53*sigmaz/c_light  # injection time offset [s]

x, y, z = solver.x, solver.y, solver.z
ixs, iys = np.abs(x-xs).argmin(), np.abs(y-ys).argmin()
ixt, iyt = np.abs(x-xt).argmin(), np.abs(y-yt).argmin()

def beam(solver, t):
    # Define gaussian
    s0 = z.min() - c_light*tinj
    s = z-c_light*t
    profile = 1/np.sqrt(2*np.pi*sigmaz**2)*np.exp(-(s-s0)**2/(2*sigmaz**2))

    # Update current
    solver.J[ixs,iys,:,'z'] = q*c_light*beta*profile/solver.dx/solver.dy

# ------------ Time loop ----------------
# Define wake length
wakelength = 1*1e-3 #[m]

# Obtain simulation time
tmax = (wakelength + tinj*c_light + (zmax-zmin))/c_light #[s]
Nt = int(tmax/solver.dt)
t = np.arange(0, Nt*solver.dt, solver.dt)
ninj = int(tinj/solver.dt)

for n in tqdm(range(Nt)):

    # Initial condition
    beam(solver, n*solver.dt)

    # Advance
    solver.one_step()

    # Plot 2D
    if n>ninj and n%10 == 0:
        solver.plot2D(field='E', component='Abs', plane='ZY', pos=0.5, norm='linear', 
                      vmin=0, vmax=10e5, cmap='rainbow', patch_alpha=0.5, 
                      add_patch='Cavity', patch_reverse=True, title='imgCav/Eabs', off_screen=True,  
                      n=n, interpolation='spline36')
    
    # Save
    #if n == 0: Ezt = {}
    #Ezt['#'+str(n).zfill(5)]=solver.E[ixt, iyt, :, 'z'] 

# Plot 3D built-in
#solver.plot3D(field='H', component='y', clim=None,  hide_solids=None, show_solids=None, 
#               add_stl='Solid 1', stl_opacity=0.1, stl_colors='white',
#               title=None, cmap='rainbow', clip_volume=False, clip_normal='-y',
#               clip_box=True, clip_bounds=None, off_screen=True, zoom=0.4, n=n)

# 