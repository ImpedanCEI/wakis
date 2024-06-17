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
Nx = 55
Ny = 55
Nz = 108
#dt = 5.707829241e-12 # CST

# Embedded boundaries
stl_cavity = 'cavity.stl' 
stl_pipe = 'beampipe.stl'

# Materials
stl_solids = {'cavity': stl_cavity, 'pipe': stl_pipe}
stl_materials = {'cavity': 'vacuum', 'pipe':  'vacuum'}
background = [100, 1.0, 100] # lossy metal [ε_r, µ_r, σ]

#Geometry
stl_scale = {'cavity': 1.0, 'pipe': [1.0, 1.0, 2.0]}

# Domain bounds
surf = pv.read(stl_cavity) + pv.read(stl_pipe).scale([1.0, 1.0, 2.0])
xmin, xmax, ymin, ymax, zmin, zmax = surf.bounds
Lx, Ly, Lz = (xmax-xmin), (ymax-ymin), (zmax-zmin)

# Set grid and geometry
grid = GridFIT3D(xmin, xmax, ymin, ymax, zmin, zmax, Nx, Ny, Nz, 
                stl_solids=stl_solids, 
                stl_materials=stl_materials,
                stl_scale=stl_scale)
grid.inspect()

# ------------ Beam source ----------------
# Beam parameters
beta = 0.5          # beam relativistic beta 
sigmaz = beta*6e-2  # [m] -> multiplied by beta to have f_max cte
q = 1e-9            # [C]
xs = 0.             # x source position [m]
ys = 0.             # y source position [m]
xt = 0.             # x test position [m]
yt = 0.             # y test position [m]
# [DEFAULT] tinj = 8.53*sigmaz/(beta*c)  # injection time offset [s] 

# Simualtion
wakelength = 21  #[m]
add_space = 10   # no. cells

wake = WakeSolver(q=q, sigmaz=sigmaz, beta=beta,
            xsource=xs, ysource=ys, xtest=xt, ytest=yt,
            add_space=add_space, save=True, logfile=True)

# ----------- Solver & Simulation ----------
# boundary conditions``
bc_low=['pec', 'pec', 'pec']
bc_high=['pec', 'pec', 'pec']

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
run = False
if run:
    solver.wakesolve(wakelength=wakelength, add_space=add_space,
                    plot=False, plot_every=30, save_J=False,
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
    

# Run only wake solve
runWake = True
wake.solve(Ez_file = 'Ez.h5')

#-------------- Compare with CST -------------

#--- Longitudinal wake and impedance ---
plot = True
if plot:

    # CST wake
    #cstWP = wake.read_txt('cst/WP.txt')
    #cstZ = wake.read_txt('cst/Z.txt')
    wake.f = np.abs(wake.f)

    fig, ax = plt.subplots(1,2, figsize=[12,4], dpi=150)
    ax[0].plot(wake.s*1e2, wake.WP, c='r', lw=1.5, label='FIT+Wakis')
    #ax[0].plot(cstWP[0], cstWP[1], c='k', ls='--', lw=1.5, label='CST')
    ax[0].set_xlabel('s [mm]')
    ax[0].set_ylabel('Longitudinal wake potential [V/pC]', color='r')
    ax[0].legend()

    ax[1].plot(wake.f*1e-9, np.abs(wake.Z), c='b', lw=1.5, label='FIT+Wakis')
    #ax[1].plot(cstZ[0], cstZ[1], c='k', ls='--', lw=1.5, label='CST')
    ax[1].set_xlabel('f [GHz]')
    ax[1].set_ylabel(r'Longitudinal impedance [Abs][$\Omega$]', color='b')
    ax[1].legend()

    fig.suptitle('Benchmark with CST Wakefield Solver')
    fig.tight_layout()
    fig.savefig('results/benchmark.png')

    plt.show()

#--- Longitudinal wake and impedance ---
plot = False
if plot:
    key = '_beta0.6_wl5'
    results = f'results{key}/'
    wake.load_results(results)
    # CST wake
    cstWP = wake.read_txt(f'cst/W{key}.txt')
    cstZ = wake.read_txt(f'cst/cZ{key}.txt')
    wake.f = np.abs(wake.f)

    fig, ax = plt.subplots(1,2, figsize=[12,4], dpi=150)
    ax[0].plot(wake.s*1e2, wake.WP, c='r', lw=1.5, label='wakis')
    ax[0].plot(cstWP[0], cstWP[1], c='k', ls='--', lw=1.2, label='CST')
    ax[0].set_xlabel('s [mm]')
    ax[0].set_ylabel('Longitudinal wake potential [V/pC]', color='r')
    ax[0].legend()

    ax[1].plot(wake.f*1e-9, np.real(wake.Z), c='b', lw=1.5, label='Re(Z) wakis')
    ax[1].plot(wake.f*1e-9, np.imag(wake.Z), c='royalblue', lw=1.3, label='Im(Z) wakis')
    ax[1].plot(cstZ[0], cstZ[1], c='k', ls='--', lw=1.2, label='Re(Z) CST')
    ax[1].plot(cstZ[0], cstZ[2], c='k', ls=':', lw=1.2, label='Im(Z) CST')
    ax[1].set_xlabel('f [GHz]')
    ax[1].set_ylabel(r'Longitudinal impedance [Abs][$\Omega$]', color='b')
    ax[1].legend()

    fig.suptitle('Benchmark with CST Wakefield Solver')
    fig.tight_layout()
    fig.savefig(f'{results}benchmarkReIm.png')

    plt.show()

#--- 1d Ez field ---
plot = False
if plot:
    # E field
    d = wake.read_Ez('results_sigma5/Ez.h5',return_value=True)
    dd = wake.read_Ez('results_pec/Ez.h5',return_value=True)
    t, z = np.array(d['t']), np.array(d['z'])    
    dt = t[1]-t[0]
    steps = list(d.keys())

    # Beam J
    current = wake.read_Ez('Jz.h5',return_value=True)

    for n, step in enumerate(steps[:1740:20]):
        fig, ax = plt.subplots(1,1, figsize=[6,4], dpi=150)
        axx = ax.twinx()  

        ax.plot(z, d[step][1,1,:], c='g', lw=1.5, label=r'Ez(0,0,z) FIT | $\sigma$ = 5 S/m')
        ax.plot(z, dd[step][1,1,:], c='grey', lw=1.5, label='Ez(0,0,z) FIT | PEC')
        ax.set_xlabel('z [m]')
        ax.set_ylabel(r'$E_z$ field amplitude [V/m]', color='g')
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

# compare beta
plot = False
if plot:

    fig, ax = plt.subplots(1,2, figsize=[12,4.5], dpi=170)

    # Read data
    keys = ['beta1_wl5', 'beta0.8_wl5', 'beta0.6_wl5']
    res, cstWP, cstZ = {}, {}, {}
    for k in keys:
        # Wakis wake
        res[k] = wake.copy()
        res[k].load_results(f'results_{k}/')
        # CST wake
        cstWP[k] = wake.read_txt(f'cst/W_{k}.txt')
        cstZ[k] = wake.read_txt(f'cst/cZ_{k}.txt')
        cstZ[k][1] = np.abs(cstZ[k][1] + 1.j*cstZ[k][2])

    ax[0].plot(res[keys[0]].s, res[keys[0]].WP, c='tab:red', lw=2., alpha=0.5, label=r'$\beta = 1.0$')
    ax[0].plot(res[keys[1]].s, res[keys[1]].WP, c='tab:blue', lw=2., alpha=0.5, label=r'$\beta = 0.8$')
    ax[0].plot(res[keys[2]].s, res[keys[2]].WP, c='tab:green', lw=2., alpha=0.5, label=r'$\beta = 0.6$')

    ax[0].plot(cstWP[keys[0]][0]*1e-2, cstWP[keys[0]][1], c='tab:red', ls='--', lw=1.5, label=r'CST $\beta = 1.0$')
    ax[0].plot(cstWP[keys[1]][0]*1e-2, cstWP[keys[1]][1], c='tab:blue', ls='--', lw=1.5, label=r'CST $\beta = 0.8$')
    ax[0].plot(cstWP[keys[2]][0]*1e-2, cstWP[keys[2]][1], c='tab:green', ls='--', lw=1.5, label=r'CST $\beta = 0.6$')

    ax[0].set_xlabel('s [m]')
    ax[0].set_ylabel('Longitudinal wake potential [V/pC]', color='k')
    ax[0].legend()
    ax[0].margins(x=0.01, tight=True)

    ax[1].plot(res[keys[0]].f*1e-9, np.abs(res[keys[0]].Z), c='tab:red', lw=2., alpha=0.5, label=r'$\beta = 1.0$')
    ax[1].plot(res[keys[1]].f*1e-9, np.abs(res[keys[1]].Z), c='tab:blue', lw=2., alpha=0.5, label=r'$\beta = 0.8$')
    ax[1].plot(res[keys[2]].f*1e-9, np.abs(res[keys[2]].Z), c='tab:green', lw=2., alpha=0.5, label=r'$\beta = 0.6$')

    ax[1].plot(cstZ[keys[0]][0], cstZ[keys[0]][1], c='tab:red', ls='--', lw=1.5, label=r'CST $\beta = 1.0$')
    ax[1].plot(cstZ[keys[1]][0], cstZ[keys[1]][1], c='tab:blue', ls='--', lw=1.5, label=r'CST $\beta = 0.8$')
    ax[1].plot(cstZ[keys[2]][0], cstZ[keys[2]][1], c='tab:green', ls='--', lw=1.5, label=r'CST $\beta = 0.6$')
    
    ax[1].set_xlabel('f [GHz]')
    ax[1].set_ylabel(r'Longitudinal impedance [Abs][$\Omega$]', color='k')
    ax[1].legend()
    ax[1].margins(x=0.01, tight=True)

    fig.suptitle('Benchmark with CST Wakefield Solver')
    fig.tight_layout()
    
    fig.savefig('benchmark_beta.png')
    plt.show()


#-------------- Compare .h5 files -------------
plot = False
if plot:
    # E field
    keys = ['beta1_wl5', 'beta0.8_wl5', 'beta0.6_wl5']
    d1 = wake.read_Ez(f'results_{keys[0]}/Ez.h5',return_value=True)
    d2 = wake.read_Ez(f'results_{keys[1]}/Ez.h5',return_value=True)
    d3 = wake.read_Ez(f'results_{keys[2]}/Ez.h5',return_value=True)

    t = np.array(d1['t'])   
    dt = t[1]-t[0]
    steps = list(d1.keys())

    # Beam J
    dd1 = wake.read_Ez(f'results_{keys[0]}/Jz.h5',return_value=True)
    dd2 = wake.read_Ez(f'results_{keys[1]}/Jz.h5',return_value=True)
    dd3 = wake.read_Ez(f'results_{keys[2]}/Jz.h5',return_value=True)

    for n, step in enumerate(steps[:3660:30]):
        fig, ax = plt.subplots(1,1, figsize=[7,4], dpi=200)
        axx = ax.twinx()  

        # Beam current
        axx.plot(np.array(d1['z']), dd1[step][1,1,:], ls=':', c='k', lw=1.5, label='Current $J_z$')
        axx.plot(np.array(d1['z']), dd1[step][1,1,:], ls=':', c='tab:red', lw=1.5, )
        axx.plot(np.array(d2['z']), dd2[step][1,1,:], ls=':', c='tab:blue', lw=1.5, )
        axx.plot(np.array(d3['z']), dd3[step][1,1,:], ls=':', c='tab:green', lw=1.5, )
        axx.set_ylabel('$J_z$ beam current [C/m]', color='k')
        axx.set_ylim(-3e5, 3e5)
        axx.legend(loc=2, frameon=False)

        # E field
        ax.plot(np.array(d1['z']) , d1[step][1,1,:], c='tab:red', lw=2.5, alpha=0.6, label=r'Ez(0,0,z) $\beta=1.0$')
        ax.plot(np.array(d2['z']) , d2[step][1,1,:], c='tab:blue', lw=2.5, alpha=0.6, label=r'Ez(0,0,z) $\beta=0.8$')
        ax.plot(np.array(d3['z']) , d3[step][1,1,:], c='tab:green', lw=2.5, alpha=0.6, label=r'Ez(0,0,z) $\beta=0.6$')
        ax.set_xlabel('z [m]')
        ax.set_ylabel(r'$E_z$ field amplitude [V/m]', color='k')
        ax.set_ylim(-1e4, 1e4)
        ax.set_xlim(np.array(d1['z']).min(), np.array(d1['z']).max())
        ax.legend(loc=1, frameon=False)

        fig.suptitle('timestep='+str(n*30))
        fig.tight_layout()
        fig.savefig('img/Ez1d_'+str(n*30).zfill(6)+'.png')

        plt.clf()
        plt.close(fig)