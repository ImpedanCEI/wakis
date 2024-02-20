import os, sys
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

sys.path.append('../../')

from solverFIT3D import SolverFIT3D
from gridFIT3D import GridFIT3D 
from wakeSolver import WakeSolver

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

wake = WakeSolver(q=q, sigmaz=sigmaz, beta=beta,
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
add_space = 10  # no. cells

# Plot settings
if not os.path.exists('img/'): os.mkdir('img/')
plotkw = {'title':'img/Ez', 
            'add_patch':'cavity', 'patch_alpha':0.7,
            'vmin':-1e4, 'vmax':1e4,
            'plane': [int(Nx/2), slice(0, Ny), slice(add_space, -add_space)]}

# 1 - Run full electromagnetic time-domain simulation
if not os.path.exists('Ez.h5'):
    solver.wakesolve(wakelength=wakelength, add_space=add_space,
                    plot=False, plot_every=30, save_J=True,
                    **plotkw)

# 2 - Wake computation from pre-computed Ez field
elif not os.path.exists('results/'): 
    wake.solve(Ez_file='Ez.h5')

# Or Load previous results
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
    wake.f = np.abs(wake.f)

    fig, ax = plt.subplots(1,2, figsize=[10,4], dpi=150)
    ax[0].plot(wake.s*1e3, wake.WP, c='r', lw=1.5, label='FIT+Wakis')
    ax[0].plot(cstWP[0], cstWP[1], c='k', ls='--', lw=1.5, label='CST')
    ax[0].set_xlabel('s [mm]')
    ax[0].set_ylabel('Longitudinal wake potential [V/pC]', color='r')
    ax[0].legend()

    axx = ax[0].twinx()
    err = np.abs(wake.WP-np.interp(wake.s*1e3, cstWP[0], cstWP[1]))
    axx.plot(wake.s*1e3, err, c='grey', ls='-', lw=1.2, alpha=0.3)
    axx.set_ylabel('Absolute error [-]', color='grey')

    ax[1].plot(wake.f*1e-9, np.abs(wake.Z), c='b', lw=1.5, label='FIT+Wakis')
    ax[1].plot(cstZ[0], cstZ[1], c='k', ls='--', lw=1.5, label='CST')
    ax[1].set_xlabel('f [GHz]')
    ax[1].set_ylabel('Longitudinal impedance [Abs][$\Omega$]', color='b')
    ax[1].legend()

    axx = ax[1].twinx()
    err = np.abs(np.abs(wake.Z)-np.interp(wake.f*1e-9, cstZ[0], cstZ[1]))
    axx.plot(wake.f*1e-9, err, c='grey', ls='-', lw=1.2, alpha=0.3)
    axx.set_ylabel('Absolute error [-]', color='grey')

    fig.suptitle('Benchmark with CST Wakefield Solver')
    fig.tight_layout()
    fig.savefig('results/benchmark.png')

    plt.show()

#--- 1d Ez field ---
plot = True
if plot:
    # E field
    d = wake.read_Ez('Ez.h5',return_value=True)
    t, z = np.array(d['t']), np.array(d['z'])    
    dt = t[1]-t[0]
    steps = list(d.keys())

    # Beam J
    dd = wake.read_Ez('Jz.h5',return_value=True)

    # WarpX E field read
    try: warpx = wake.read_Ez('warpx/Ez.h5',return_value=True)
    except: pass

    for n, step in enumerate(steps[:3750:30]):
        fig, ax = plt.subplots(1,1, figsize=[6,4], dpi=150)
        axx = ax.twinx()  

        ax.plot(z, d[step][1,1,:], c='g', lw=1.5, label='Ez(0,0,z) FIT')
        ax.set_xlabel('z [m]')
        ax.set_ylabel('$E_z$ field amplitude [V/m]', color='g')
        ax.set_ylim(-4e4, 4e4)
        ax.set_xlim(z.min(), z.max())

        # CST E field
        try:    
            cstfiles = sorted(os.listdir('cst/1d/'))
            cst = wake.read_txt('cst/1d/'+cstfiles[n])
            ax.plot(cst[0]*1e-3, cst[1], c='k', lw=1.5, ls='--', label='Ez(0,0,z) CST')
        except:
            pass

        # WarpX E field
        sstep = '#'+str(int(n*30+300)).zfill(5)
        try: ax.plot(np.array(warpx['z']), warpx[sstep][1,1,:], c='b', lw=1.5, ls='dotted', label='Ez(0,0,z) WarpX')
        except: pass
        
        ax.legend(loc=1)

        # charge distribution
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

#------------- Compare with WarpX --------------
warpx = WakeSolver(q=q, sigmaz=sigmaz, beta=beta,
        xsource=xs, ysource=ys, xtest=xt, ytest=yt,
        save=True, results_folder='warpx/')

# Compute wake potential and impedance from Ez.h5 file
if not os.path.exists('warpx/WP.txt'):
    warpx.solve(Ez_file='warpx/Ez.h5', wakelength=1.)

plot = False
if plot:
    # warpx wake
    warpx.s, warpx.WP = wake.read_txt('warpx/WP.txt').values()
    warpx.f, warpx.Z = wake.read_txt('warpx/Z.txt').values()
    warpx.f = np.abs(warpx.f)

    # CST wake
    cstWP = wake.read_txt('cst/WP.txt')
    cstZ = wake.read_txt('cst/Z.txt')
    wake.f = np.abs(wake.f)

    # wake potential
    fig, ax = plt.subplots(1,2, figsize=[10,4], dpi=150)
    ax[0].plot(wake.s*1e3, wake.WP, c='r', lw=1.5, label='FIT+Wakis')
    ax[0].plot(warpx.s*1e3, warpx.WP, c='grey', ls='dotted', lw=1.5, label='WarpX+Wakis')
    ax[0].plot(cstWP[0], cstWP[1], c='k', ls='--', lw=1.5, label='CST')
    ax[0].set_xlabel('s [mm]')
    ax[0].set_ylabel('Longitudinal wake potential [V/pC]', color='r')
    ax[0].legend()

    # impedance
    ax[1].plot(wake.f*1e-9, np.abs(wake.Z), c='b', lw=1.5, label='FIT+Wakis')
    ax[1].plot(warpx.f*1e-9, np.abs(warpx.Z), c='grey', ls='dotted', lw=1.5, label='WarpX+Wakis')
    ax[1].plot(cstZ[0], cstZ[1], c='k', ls='--', lw=1.5, label='CST')
    ax[1].set_xlabel('f [GHz]')
    ax[1].set_ylabel('Longitudinal impedance [Abs][$\Omega$]', color='b')
    ax[1].legend()

    fig.suptitle('Benchmark with WarpX and CST Wakefield Solver')
    fig.tight_layout()
    fig.savefig('warpx/benchmark.png')

    plt.show()


#-------------- Compare .h5 files -------------
plot = False
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