import numpy as np
import matplotlib.pyplot as plt
import os, sys
import pyvista as pv
import h5py
from tqdm import tqdm
from scipy.constants import c as c_light

sys.path.append('../')

from solverFIT3D import SolverFIT3D
from gridFIT3D import GridFIT3D 

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
bc_low=['pec', 'pec', 'abc']
bc_high=['pec', 'pec', 'abc']

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
    current = q*c_light*beta*profile/solver.dx/solver.dy
    solver.J[ixs,iys,:,'z'] = current

# ------------ Time loop ----------------
# Define wake length
wakelength = 1*1e-3 #[m]

# Obtain simulation time
tmax = (wakelength + tinj*c_light + (zmax-zmin))/c_light #[s]
Nt = int(tmax/solver.dt)
t = np.arange(0, Nt*solver.dt, solver.dt)
ninj = int(tinj/solver.dt)

# Prepare save file
hf = h5py.File('hdf5/Ez_per-abs.h5', 'w')
hf2 = h5py.File('hdf5/Jz_per-abs.h5', 'w')
hf['x'], hf['y'], hf['z'], hf['t'] = x, y, z, t

plot = True

for n in tqdm(range(Nt)):

    # Initial condition
    beam(solver, n*solver.dt)

    # Advance
    solver.one_step()

    # Plot 2D
    if n>ninj and n%10 == 0 and plot:
        solver.plot2D(field='E', component='z', plane='ZY', pos=0.5, norm='linear', 
                      vmin=-1e5, vmax=1e5, cmap='rainbow', patch_alpha=0.5, 
                      add_patch='Cavity', patch_reverse=True, title='imgCav/Ez', off_screen=True,  
                      n=n, interpolation='spline36')
    
    # Save
    hf['#'+str(n).zfill(5)]=solver.E[ixt, iyt, :, 'z'] 
    hf2['#'+str(n).zfill(5)]=solver.J[ixt, iyt, :, 'z'] 

# Close save file
hf.close()
hf2.close()

# Plot 1D
'''
hf = h5py.File('hdf5/Ez_abc.h5', 'r')
hf2 = h5py.File('hdf5/Jz_abc.h5', 'r')

fig, ax = plt.subplots(1, 1, figsize=[6,5], dpi=140)

axx = ax.twinx()
keystoplot = [ninj+(Nt-ninj)*0.1, ninj+(Nt-ninj)*0.3, ninj+(Nt-ninj)*0.6, ninj+(Nt-ninj)*0.8, Nt-1]
colors = plt.cm.rainbow([0.2, 0.4, 0.6, 0.8, 1.])
colorsJ = [0.2, 0.4, 0.6, 0.8, 1.]

for i, key in enumerate(keystoplot):
    Etoplot = np.array(hf[list(hf.keys())[int(key)]])
    Jtoplot = np.array(hf2[list(hf2.keys())[int(key)]])
    ax.plot(z, Etoplot, color=colors[i], ls='--', label=f'timestep #{int(key)}', lw=2)
    axx.plot(z, Jtoplot, color='darkorange', label=f'timestep #{int(key)}', alpha=colorsJ[i])
    axx.fill_between(z, Jtoplot, color='darkorange', label=f'timestep #{int(key)}', alpha=colorsJ[i]*0.8)
hf.close()
hf2.close()

hf = h5py.File('hdf5/Ez_pec.h5', 'r')
hf2 = h5py.File('hdf5/Jz_pec.h5', 'r')

for i, key in enumerate(keystoplot):
    Etoplot = np.array(hf[list(hf.keys())[int(key)]])
    Jtoplot = np.array(hf2[list(hf2.keys())[int(key)]])
    ax.plot(z, Etoplot, color=colors[i], label=f'timestep #{int(key)}', lw=2)

ax.set_xlabel('$z$ [m]')
ax.set_ylabel('$E_z$ [V/m]', color='darkgreen', fontweight='bold')
lims = max(np.abs(plt.ylim()[0]), np.abs(plt.ylim()[0]))
ax.set_ylim(-1e6, 1e6)
axx.set_ylabel('Charge distribution [C/m]', color='r', fontweight='bold')
axx.set_ylim(ymin=-np.max(Jtoplot)*1.5, ymax=np.max(Jtoplot)*1.5)
ax.legend(loc=4, ncol=2)

fig.suptitle('$E_z(x_s,y_s,z)$ field and $J_z(x_s,y_s,z)$ for different timesteps')
fig.tight_layout()
plt.show()
'''


# Plot 3D built-in
#solver.plot3D(field='H', component='y', clim=None,  hide_solids=None, show_solids=None, 
#               add_stl='Solid 1', stl_opacity=0.1, stl_colors='white',
#               title=None, cmap='rainbow', clip_volume=False, clip_normal='-y',
#               clip_box=True, clip_bounds=None, off_screen=True, zoom=0.4, n=n)
