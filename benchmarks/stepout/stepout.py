import os, sys
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from scipy.constants import c 

sys.path.append('../../')

from solverFIT3D import SolverFIT3D
from gridFIT3D import GridFIT3D 
from wakeSolver import WakeSolver

# ---------- Domain setup ---------
# Number of mesh cells
Nx = 67 
Ny = 67 
Nz = 72 + 20
#dt = 5.707829241e-12 # CST

# Embedded boundaries
stl_pipein = 'pipein.stl' 
stl_pipeout = 'pipeout.stl'
stl_taper = 'taper.stl'

# Materials
stl_solids = {'pipein': stl_pipein, 'pipeout': stl_pipeout, 'taper': stl_taper}
stl_materials = {'pipein': 'vacuum', 'pipeout': 'vacuum', 'taper': 'vacuum'}
background = 'pec' # lossy metal [ε_r, µ_r, σ]

# Domain bounds
surf = pv.read(stl_pipein) + pv.read(stl_pipeout) + pv.read(stl_taper)
#surf.plot()

xmin, xmax, ymin, ymax, zmin, zmax = surf.bounds
Lx, Ly, Lz = (xmax-xmin), (ymax-ymin), (zmax-zmin)

# Set grid and geometry
grid = GridFIT3D(xmin, xmax, ymin, ymax, zmin, zmax, Nx, Ny, Nz, 
                stl_solids=stl_solids, 
                stl_materials=stl_materials)
#grid.inspect()

# ------------ Beam source ----------------
# Beam parameters
beta = 1.0          # beam relativistic beta 
sigmaz = beta*4e-2  # [m] -> multiplied by beta to have f_max cte
q = 1e-9            # [C]
xs = 0.             # x source position [m]
ys = 0.             # y source position [m]
xt = 0.             # x test position [m]
yt = 0.             # y test position [m]
# [DEFAULT] tinj = 8.53*sigmaz/(beta*c)  # injection time offset [s] 

# Simualtion
wakelength = 10. #[m]
add_space = 15   # no. cells

wake = WakeSolver(q=q, sigmaz=sigmaz, beta=beta,
            xsource=xs, ysource=ys, xtest=xt, ytest=yt,
            add_space=add_space, save=True, logfile=True)

# ----------- Solver & Simulation ----------
# boundary conditions``
bc_low=['pec', 'pec', 'pml']
bc_high=['pec', 'pec', 'pml']

solver = SolverFIT3D(grid, wake, 
                     bc_low=bc_low, bc_high=bc_high, 
                     use_stl=True, bg=background)
# Plot settings
if not os.path.exists('img/'): os.mkdir('img/')
plotkw = {'title':'img/Ez', 
            'add_patch':'cavity', 'patch_alpha':0.3,
            'vmin':-1e4, 'vmax':1e4,
            'plane': [int(Nx/2), slice(0, Ny), slice(add_space, -add_space)]}

# Run wakefield time-domain simulation
run = True
if run:
    solver.wakesolve(wakelength=wakelength, add_space=add_space,
                    plot=False, plot_every=30, save_J=True,
                    use_etd=False,
                    **plotkw)

# Run only electromagnetic time-domain simulation
runEM = False
if runEM:
    from sources import Beam
    beam = Beam(q=q, sigmaz=sigmaz, beta=beta,
                xsource=xs, ysource=ys)

    solver.emsolve(Nt=500, source=beam, add_space=add_space,
                    plot=False, plot_every=30, save_J=False,
                    use_etd=True, **plotkw)
    
#-------------- Compare with CST -------------

#--- Longitudinal wake and impedance ---
plot = False
if plot:
    results = 'results/'
    wake.load_results(results)

    # CST wake
    cstWP = wake.read_txt('cst/W.txt')
    cstZ = wake.read_txt('cst/cZ.txt')
    cstZ[1] = np.abs(cstZ[1]+1.j*cstZ[2])
    wake.f = np.abs(wake.f)

    fig, ax = plt.subplots(1,2, figsize=[12,4], dpi=150)
    ax[0].plot(wake.s*1e2, wake.WP, c='r', lw=1.5, alpha=0.8, label='wakis ABC')
    ax[0].plot(cstWP[0], cstWP[1], c='k', ls='--', lw=1.2, label='CST PEC')
    ax[0].set_xlabel('s [mm]')
    ax[0].set_ylabel('Longitudinal wake potential [V/pC]', color='r')
    ax[0].legend()

    ax[1].plot(wake.f*1e-9, np.abs(wake.Z), c='b', alpha=0.8, lw=1.5, label='wakis ABC')
    ax[1].plot(cstZ[0], cstZ[1], c='k', ls='--', lw=1.2, label='CST')
    ax[1].set_xlabel('f [GHz]')
    ax[1].set_ylabel(r'Longitudinal impedance [Abs][$\Omega$]', color='b')
    ax[1].legend()

    fig.suptitle('Benchmark with CST Wakefield Solver')
    fig.tight_layout()
    fig.savefig(f'{results}benchmarkAbs.png')

    plt.show()

#--- Longitudinal wake and impedance ---
plot = False
if plot:
    results = 'results_pec/'
    wake.load_results(results)

    # CST wake
    cstWP = wake.read_txt('cst/W_pec.txt')
    cstZ = wake.read_txt('cst/Z_pec.txt')
    wake.f = np.abs(wake.f)

    fig, ax = plt.subplots(1,2, figsize=[12,4], dpi=150)
    ax[0].plot(wake.s*1e2, wake.WP, c='r', lw=1.5, alpha=0.8, label='wakis PEC')
    ax[0].plot(cstWP[0], cstWP[1], c='k', ls='--', lw=1.2, label='CST PEC')
    ax[0].set_xlabel('s [mm]')
    ax[0].set_ylabel('Longitudinal wake potential [V/pC]', color='r')
    ax[0].legend()

    ax[1].plot(wake.f*1e-9, np.abs(wake.Z), c='b', alpha=0.8, lw=1.5, label='wakis PEC')
    ax[1].plot(cstZ[0], cstZ[1], c='k', ls='--', lw=1.2, label='CST')
    ax[1].set_xlabel('f [GHz]')
    ax[1].set_ylabel(r'Longitudinal impedance [Abs][$\Omega$]', color='b')
    ax[1].legend()

    fig.suptitle('Benchmark with CST Wakefield Solver')
    fig.tight_layout()
    fig.savefig(f'{results}benchmarkAbs_pec.png')

    plt.show()

#--- Longitudinal wake and impedance ---
plot = False
if plot:
    results = 'results_pml_add15/'
    wake.load_results(results)

    # CST wake
    cstWP = wake.read_txt('cst/W.txt')
    cstZ = wake.read_txt('cst/cZ.txt')
    cstZ[1] = np.abs(cstZ[1]+1.j*cstZ[2])
    wake.f = np.abs(wake.f)

    fig, ax = plt.subplots(1,2, figsize=[12,4], dpi=150)
    ax[0].plot(wake.s*1e2, wake.WP, c='r', lw=1.5, alpha=0.8, label='wakis PML')
    ax[0].plot(cstWP[0], cstWP[1], c='k', ls='--', lw=1.2, label='CST PML')
    ax[0].set_xlabel('s [mm]')
    ax[0].set_ylabel('Longitudinal wake potential [V/pC]', color='r')
    ax[0].legend()

    ax[1].plot(wake.f*1e-9, np.abs(wake.Z), c='b', alpha=0.8, lw=1.5, label='wakis PML')
    ax[1].plot(cstZ[0], cstZ[1], c='k', ls='--', lw=1.2, label='CST PML')
    ax[1].set_xlabel('f [GHz]')
    ax[1].set_ylabel(r'Longitudinal impedance [Abs][$\Omega$]', color='b')
    ax[1].legend()

    fig.suptitle('Benchmark with CST Wakefield Solver')
    fig.tight_layout()
    fig.savefig(f'{results}benchmarkAbs.png')

    plt.show()

#--- Longitudinal wake and impedance ---
plot = False
if plot:
    results = 'results_pec/'
    wake.load_results(results)
    
    # CST wake
    cstWP = wake.read_txt('cst/W_pec.txt')
    cstZ = wake.read_txt('cst/cZ_pec.txt')
    wake.f = np.abs(wake.f)

    fig, ax = plt.subplots(1,2, figsize=[12,4], dpi=150)
    ax[0].plot(wake.s*1e2, wake.WP, c='r', lw=1.5, label='wakis PEC')
    ax[0].plot(cstWP[0], cstWP[1], c='k', ls='--', lw=1.2, label='CST PEC')
    ax[0].set_xlabel('s [mm]')
    ax[0].set_ylabel('Longitudinal wake potential [V/pC]', color='r')
    ax[0].legend()

    ax[1].plot(wake.f*1e-9, np.real(wake.Z), c='b', lw=1.5, label='Re(Z) wakis PEC')
    ax[1].plot(cstZ[0], cstZ[1], c='k', ls='--', lw=1.2, label='Re(Z) CST')
    ax[1].plot(wake.f*1e-9, np.imag(wake.Z), c='royalblue', lw=1.5, label='Im(Z) wakis PEC')
    ax[1].plot(cstZ[0], cstZ[2], c='k', ls=':', lw=1.2, label='Im(Z) CST')
    ax[1].set_xlabel('f [GHz]')
    ax[1].set_ylabel(r'Longitudinal impedance [$\Omega$]', color='b')
    ax[1].legend()

    fig.suptitle('Benchmark with CST Wakefield Solver')
    fig.tight_layout()
    fig.savefig(f'{results}benchmarkReIm_pec.png')

    plt.show()

#--- 1d Ez field ---
plot = True
if plot:
    # E field
    d = wake.read_Ez('results/Ez.h5',return_value=True)
    #dd = wake.read_Ez('results_pec/Ez.h5',return_value=True)
    t, z = np.array(d['t']), np.array(d['z'])    
    dt = t[1]-t[0]
    steps = list(d.keys())

    # Beam J
    current = wake.read_Ez('Jz.h5',return_value=True)

    for n, step in enumerate(steps[:1740:20]):
        fig, ax = plt.subplots(1,1, figsize=[6,4], dpi=150)
        axx = ax.twinx()  

        ax.plot(z, d[step][1,1,:], c='g', lw=1.5, label=r'Ez(0,0,z) PML')
        #ax.plot(z, dd[step][1,1,:], c='grey', lw=1.5, label='Ez(0,0,z) FIT | PEC')
        ax.set_xlabel('z [m]')
        ax.set_ylabel('$E_z$ field amplitude [V/m]', color='g')
        ax.set_ylim(-3e3, 3e3)
        ax.set_xlim(z.min(), z.max())
        
        # CST E field
        try:    
            cstfiles = sorted(os.listdir('cst/1d/'))
            cst = wake.read_txt('cst/1d/'+cstfiles[n])
            ax.plot(cst[0]*1e-2, cst[1], c='k', lw=1.5, ls='--', label=r'Ez(0,0,z) CST | $\sigma$ = 10 S/m')
        except:
            pass

        ax.legend(loc=1)

        # charge distribution
        axx.plot(z, current[step][1,1,:], c='r', lw=1.0, label='lambda λ(z)')
        axx.set_ylabel(r'$J_z$ beam current [C/m]', color='r')
        axx.set_ylim(-8e4, 8e4)

        fig.suptitle('timestep='+str(n*20))
        fig.tight_layout()
        fig.savefig('img/Ez1d_'+str(n*20).zfill(6)+'.png')

        plt.clf()
        plt.close(fig)

#-------------- Compare result files -------------

#--- Longitudinal wake and impedance ---

# compare BC
plot = True
if plot:
    # CST wake (PML)
    cstWP = wake.read_txt('cst/W.txt')
    cstZ = wake.read_txt('cst/cZ.txt')
    cstWP_pec = wake.read_txt('cst/W_pec.txt')
    cstZ_pec = wake.read_txt('cst/cZ_pec.txt')

    fig, ax = plt.subplots(1,2, figsize=[12,4], dpi=150)

    # Wakis wake
    keys = ['pec', 'pml_add15']
    res = {}
    for k in keys:
        res[k] = wake.copy()
        res[k].load_results(f'results_{k}/')

    ax[0].plot(res[keys[0]].s, res[keys[0]].WP, c='r', alpha=0.4,  lw=1.0, label='wakis PEC')
    ax[0].plot(res[keys[1]].s, res[keys[1]].WP, c='r', lw=1.5, label='wakis PML')

    ax[0].plot(cstWP_pec[0]*1e-2, cstWP_pec[1], c='k', ls='--', lw=1, alpha=0.4, label='CST PEC')
    ax[0].plot(cstWP[0]*1e-2, cstWP[1], c='k', ls='--', lw=1.5, label='CST PML')
    ax[0].set_xlabel('s [m]')
    ax[0].set_ylabel('Longitudinal wake potential [V/pC]', color='r')
    ax[0].legend()

    ax[1].plot(res[keys[0]].f*1e-9, np.real(res[keys[0]].Z), c='b', lw=1.0, alpha=0.4, label='wakis PEC')
    ax[1].plot(res[keys[1]].f*1e-9, np.real(res[keys[1]].Z), c='b', lw=1.5,  label='wakis PML')

    ax[1].plot(cstZ_pec[0], cstZ_pec[1], c='k', ls='--', alpha=0.4, lw=1, label='CST PEC')
    ax[1].plot(cstZ[0], cstZ[1], c='k', ls='-', lw=1.5, label='CST PML')
    ax[1].set_xlabel('f [GHz]')
    ax[1].set_ylabel(r'Longitudinal impedance [Re][$\Omega$]', color='b')
    ax[1].legend()

    fig.suptitle('Benchmark with CST Wakefield Solver')
    fig.tight_layout()
    fig.savefig('benchmark_BC_pml.png')

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
        axx.set_ylabel(r'$J_z$ beam current [C/m]', color='r')
        axx.set_ylim(-7e6, 7e6)

        # E field
        ax.plot(np.array(d1['z']) , d1[step][1,1,:], c='b', lw=1.5, label='Ez(0,0,z) ABC bc')
        ax.plot(np.array(d2['z']) , d2[step][1,1,:], c='g', lw=1.5, label='Ez(0,0,z) PEC bc')
        ax.plot(np.array(d3['z']) , d3[step][1,1,:], c='limegreen', lw=1.5, label='Ez(0,0,z) PEC+addspace 15')
        ax.set_xlabel('z [m]')
        ax.set_ylabel(r'$E_z$ field amplitude [V/m]', color='k')
        ax.set_ylim(-4e4, 4e4)
        ax.set_xlim(np.array(d1['z']).min(), np.array(d1['z']).max())
        ax.legend(loc=1)

        fig.suptitle('timestep='+str(n*30))
        fig.tight_layout()
        fig.savefig('img/Ez1d_'+str(n*30).zfill(6)+'.png')

        plt.clf()
        plt.close(fig)