import os, sys
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

sys.path.append('../../')

from solverFIT3D import SolverFIT3D
from gridFIT3D import GridFIT3D 
from wake import Wake

# ---------- Domain setup ---------
# Number of mesh cells
Nx = 50
Ny = 50
Nz = 150

# Embedded boundaries
stl_file = 'cubcavity.stl' 
surf = pv.read(stl_file)

stl_solids = {'cavity': stl_file}
stl_materials = {'cavity': 'vacuum'}

# Domain bounds
xmin, xmax, ymin, ymax, zmin, zmax = surf.bounds
Lx, Ly, Lz = (xmax-xmin), (ymax-ymin), (zmax-zmin)

# set grid and geometry
grid = GridFIT3D(xmin, xmax, ymin, ymax, zmin, zmax, Nx, Ny, Nz, 
                stl_solids=stl_solids, 
                stl_materials=stl_materials)
    
# ------------ Beam source ----------------
# Beam parameters
sigmaz = 18.5e-3    #[m]
q = 1e-9            #[C]
beta = 1.0          # beam beta TODO
xs = 0.             # x source position [m]
ys = 0.             # y source position [m]
xt = 0.             # x test position [m]
yt = 0.             # y test position [m]
# [DEFAULT] tinj = 8.53*sigmaz/c_light  # injection time offset [s] 

wake = Wake(q=q, sigmaz=sigmaz, beta=beta,
            xsource=xs, ysource=ys, xtest=xt, ytest=yt,
            save=True, logfile=True)

# ----------- Solver & Simulation ----------
# boundary conditions
bc_low=['pec', 'pec', 'pec']
bc_high=['pec', 'pec', 'pec']

# set Solver object
solver = SolverFIT3D(grid, wake,
                     bc_low=bc_low, bc_high=bc_high, 
                     use_stl=True, bg='pec')

wakelength = 1. #[m]
add_space = 15  # no. cells

# Plot settings
if not os.path.exists('img/'): os.mkdir('img/')
plotkw = {'title':'img/Ez', 
            'add_patch':'cavity', 'patch_alpha':0.7,
            'vmin':-1e4, 'vmax':1e4,
            'plane': [int(Nx/2), slice(0, Ny), slice(add_space, -add_space)]}

# a - Run full electromagnetic time-domain simulation
if not os.path.exists('Ez.h5'):
    solver.wakesolve(wakelength=wakelength, #add_space=add_space,
                    plot=False, plot_every=30, save_J=True,
                    **plotkw)

# b - Wake computation from pre-computed Ez field
elif not os.path.exists('results/'): 
    wake.solve(Ez_file='Ez.h5')

# c -Load previous results
else:
    wake.s, wake.WP = wake.read_txt('results/WP.txt').values()
    wake.f, wake.Z = wake.read_txt('results/Z.txt').values()

#-------------- Compare with CST -------------

#--- Longitudinal wake and impedance ---
plot = False
if plot:
    # CST wake
    cstWP = wake.read_txt('cst/WP.txt')
    cstZ = wake.read_txt('cst/Z.txt')

    fig, ax = plt.subplots(1,2, figsize=[8,4], dpi=150)
    ax[0].plot(wake.s*1e3, wake.WP, c='r', lw=1.5, label='FIT+Wakis')
    ax[0].plot(cstWP[0], cstWP[1], c='k', ls='--', lw=1.5, label='CST')
    ax[0].set_xlabel('s [mm]')
    ax[0].set_ylabel('Longitudinal wake potential [V/pC]', color='r')
    ax[0].legend()

    ax[1].plot(np.abs(wake.f)*1e-9, np.abs(wake.Z), c='b', lw=1.5, label='FIT+Wakis')
    ax[1].plot(cstZ[0], cstZ[1], c='k', ls='--', lw=1.5, label='CST')
    ax[1].set_xlabel('f [GHz]')
    ax[1].set_ylabel('Longitudinal impedance [Abs][$\Omega$]', color='b')
    ax[1].legend()

    fig.suptitle('Benchmark with CST Wakefield Solver')
    fig.tight_layout()
    fig.savefig('benchmark.png')

    plt.show()

#--- 1d Ez field ---
plot = False
if plot:
    # E field
    d = wake.read_Ez('EzABC.h5',return_value=True)
    t, z = np.array(d['t']), np.array(d['z'])    
    dt = t[1]-t[0]
    steps = list(d.keys())

    # Beam J
    dd = wake.read_Ez('Jz.h5',return_value=True)

    # CST E field
    cstfiles = sorted(os.listdir('cst/1d/'))

    for n, step in enumerate(steps[:3750:30]):
        fig, ax = plt.subplots(1,1, figsize=[6,4], dpi=150)
        axx = ax.twinx()  

        #E field
        cst = wake.read_txt('cst/1d/'+cstfiles[n])
        ax.plot(cst[0]*1e-3, cst[1], c='k', lw=1.5, ls='--', label='Ez(0,0,z) CST')

        ax.plot(z, d[step][1,1,:], c='g', lw=1.5, label='Ez(0,0,z) FIT')
        ax.set_xlabel('z [m]')
        ax.set_ylabel('$E_z$ field amplitude [V/m]', color='g')
        ax.set_ylim(-4e4, 4e4)
        ax.set_xlim(z.min(), z.max())
        ax.legend(loc=1)

        axx.plot(z, dd[step][1,1,:], c='r', lw=1.0, label='lambda λ(z)')
        axx.set_ylabel('$J_z$ beam current [C/m]', color='r')
        axx.set_ylim(-7e6, 7e6)

        fig.suptitle('timestep='+str(n*30))
        fig.tight_layout()
        fig.savefig('img/Ez1d_'+str(n*30).zfill(6)+'.png')

        plt.clf()
        plt.close(fig)


#--- 3d Ez field ---
# TODO
'''
if not os.path.exists('cst/3d/Ez.h5'):
    wake.read_cst_3d('cst/3d/')
'''

#-------------- Compare .h5 files -------------
plot = True
if plot:
    # E field
    d1 = wake.read_Ez('EzABC.h5',return_value=True)
    d2 = wake.read_Ez('EzPEC.h5',return_value=True)
    d3 = wake.read_Ez('EzPECadd.h5',return_value=True)

    t = np.array(d1['t'])   
    dt = t[1]-t[0]
    steps = list(d1.keys())

    # Beam J
    dd = wake.read_Ez('Jz.h5',return_value=True)

    for n, step in enumerate(steps[:3750:30]):
        fig, ax = plt.subplots(1,1, figsize=[6,4], dpi=150)
        axx = ax.twinx()  

        # Beam current
        axx.plot(np.array(d1['z']), dd[step][1,1,:], c='r', lw=1.0, label='lambda λ(z)')
        axx.set_ylabel('$J_z$ beam current [C/m]', color='r')
        axx.set_ylim(-7e6, 7e6)

        # E field
        ax.plot(np.array(d1['z']) , d1[step][1,1,:], c='b', lw=1.5, label='Ez(0,0,z) ABC bc')
        ax.plot(np.array(d2['z']) , d2[step][1,1,:], c='g', lw=1.5, label='Ez(0,0,z) PEC bc')
        ax.plot(np.array(d3['z']) , d3[step][1,1,:], c='limegreen', lw=1.5, label='Ez(0,0,z) PEC+addspace 15')
        ax.set_xlabel('z [m]')
        ax.set_ylabel('$E_z$ field amplitude [V/m]', color='k')
        ax.set_ylim(-4e4, 4e4)
        ax.set_xlim(np.array(d1['z']).min(), np.array(d1['z']).max())
        ax.legend(loc=1)

        fig.suptitle('timestep='+str(n*30))
        fig.tight_layout()
        fig.savefig('img/Ez1d_'+str(n*30).zfill(6)+'.png')

        plt.clf()
        plt.close(fig)