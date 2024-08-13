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
Nx = 25
Ny = 25
Nz = 150
#dt = 5.707829241e-12 # CST

# Embedded boundaries
stl_pipe = 'squarepipe.stl'

# Materials
stl_solids = {'pipe': stl_pipe}
stl_materials = {'pipe':  'vacuum'}
background = 'pec' # lossy metal [ε_r, µ_r, σ]

#Geometry
stl_scale = {'pipe': [1.0, 1.0, 1.0]}

# Domain bounds
surf = pv.read(stl_pipe).scale(stl_scale['pipe'])
xmin, xmax, ymin, ymax, zmin, zmax = surf.bounds
Lx, Ly, Lz = (xmax-xmin), (ymax-ymin), (zmax-zmin)

# Set grid and geometry
grid = GridFIT3D(xmin, xmax, ymin, ymax, zmin, zmax, Nx, Ny, Nz, 
                stl_solids=stl_solids, 
                stl_materials=stl_materials,
                stl_scale=stl_scale)
#grid.inspect()

# ------------ Beam source ----------------
# Beam parameters
beta = 0.4        # beam relativistic beta 
sigmaz = beta*18.5e-3 # [m] -> multiplied by beta to have f_max cte
q = 1e-9            # [C]
xs = 0.             # x source position [m]
ys = 0.             # y source position [m]
xt = 0.             # x test position [m]
yt = 0.             # y test position [m]
tinj = 8.54*sigmaz/(np.sqrt(beta)*beta*c)  # injection time offset [s] 

# Simualtion
wakelength = 1  #[m]
add_space = 10   # no. cells
results_folder = f'results_beta{beta}_add{add_space}_inj/'

wake = WakeSolver(q=q, sigmaz=sigmaz, beta=beta, ti=tinj,
            xsource=xs, ysource=ys, xtest=xt, ytest=yt,
            add_space=add_space, results_folder=results_folder,
            Ez_file='Ez.h5', save=True, logfile=False)

# ----------- Solver & Simulation ----------
# boundary conditions``
bc_low=['pec', 'pec', 'pec']
bc_high=['pec', 'pec', 'pec']

solver = SolverFIT3D(grid, wake, 
                     bc_low=bc_low, bc_high=bc_high, 
                     use_stl=True, bg=background)
# Plot settings
img_folder = results_folder+'img/'
if not os.path.exists(img_folder): os.mkdir(img_folder)
plotkw = {'title': img_folder+'Ez', 
            #'add_patch':'pipe', 'patch_alpha':0.3,
            'vmin':-1e4, 'vmax':1e4,
            'plane': [int(Nx/2), slice(0, Ny), slice(0, Nz)]}

# Run wakefield time-domain simulation
run = True
if run:
    solver.wakesolve(wakelength=wakelength, add_space=add_space,
                    plot=False, plot_every=50, plot_until=7000,
                    save_J=False,
                    **plotkw)    

# Run only wake solve
runWake = False
if runWake:
    wake.solve(Ez_file = 'Ez.h5')

#-------------- Compare with CST -------------

#--- Longitudinal wake and impedance ---
plot = True
if plot:
    #results_folder = f'results_beta{beta}_add0_inj/'
    #wake.load_results(results_folder)

    # CST wake
    cstWP = wake.read_txt(f'cst/WP_beta{beta}.txt')
    cstZ = wake.read_txt(f'cst/Z_beta{beta}.txt')
    wake.f = np.abs(wake.f)

    fig, ax = plt.subplots(1,2, figsize=[12,4], dpi=150)
    ax[0].plot(wake.s*1e3, wake.WP, c='r', lw=1.5, label='wakis')
    ax[0].plot(cstWP[0], cstWP[1], c='k', ls='--', lw=1.2, label='CST')
    ax[0].set_xlabel('s [mm]')
    ax[0].set_ylabel('Longitudinal wake potential [V/pC]', color='r')
    ax[0].legend()

    ax[1].plot(wake.f*1e-9, np.real(wake.Z), c='b', lw=1.5, label='Re(Z) wakis')
    ax[1].plot(wake.f*1e-9, np.imag(wake.Z), c='cyan', lw=1.3, label='Im(Z) wakis')
    ax[1].plot(wake.f*1e-9, np.abs(wake.Z), c='b', lw=1.5, alpha=0.5, label='Abs(Z) wakis')
    ax[1].plot(cstZ[0], cstZ[1], c='k', ls='--', lw=1.2, label='Re(Z) CST')
    ax[1].plot(cstZ[0], cstZ[2], c='k', ls=':', lw=1.2, label='Im(Z) CST')
    ax[1].plot(cstZ[0], np.abs(cstZ[1]+1.j*cstZ[2]), c='k', ls='-', lw=1.2, alpha=0.5, label='Abs(Z) CST')
    ax[1].set_xlabel('f [GHz]')
    ax[1].set_ylabel(r'Longitudinal impedance [Abs][$\Omega$]', color='b')
    ax[1].legend()

    fig.suptitle('Benchmark with CST Wakefield Solver')
    fig.tight_layout()
    fig.savefig(f'{results_folder}benchmarkReImAbs.png')

    plt.show()

#--- Longitudinal wake and impedance w/ error ---
plot = True
if plot:
    #results_folder = f'results_beta{beta}_add0_inj/'
    #wake.load_results(results_folder)

    # CST wake
    cstWP = wake.read_txt(f'cst/WP_beta{beta}.txt')
    cstZ = wake.read_txt(f'cst/Z_beta{beta}.txt')
    wake.f = np.abs(wake.f)

    fig, ax = plt.subplots(1,2, figsize=[12,4], dpi=150)
    ax[0].plot(wake.s*1e3, wake.WP, c='r', lw=1.5, label='wakis')
    ax[0].plot(cstWP[0], cstWP[1], c='k', ls='--', lw=1.2, label='CST')
    ax[0].set_xlabel('s [mm]')
    ax[0].set_ylabel('Longitudinal wake potential [V/pC]', color='r')
    

    ax[1].plot(wake.f*1e-9, np.real(wake.Z), c='b', lw=1.5, label='Re(Z) wakis')
    ax[1].plot(wake.f*1e-9, np.imag(wake.Z), c='cyan', lw=1.3, label='Im(Z) wakis')
    ax[1].plot(wake.f*1e-9, np.abs(wake.Z), c='b', lw=1.5, alpha=0.5, label='Abs(Z) wakis')
    ax[1].plot(cstZ[0], cstZ[1], c='k', ls='--', lw=1.2, label='Re(Z) CST')
    ax[1].plot(cstZ[0], cstZ[2], c='k', ls=':', lw=1.2, label='Im(Z) CST')
    ax[1].plot(cstZ[0], np.abs(cstZ[1]+1.j*cstZ[2]), c='k', ls='-', lw=1.2, alpha=0.5, label='Abs(Z) CST')
    ax[1].set_xlabel('f [GHz]')
    ax[1].set_ylabel(r'Longitudinal impedance [Abs][$\Omega$]', color='b')

    err = wake.copy()
    err.WP = np.interp(wake.s, cstWP[0]*1e-3, cstWP[1]) - wake.WP
    err.lambdas = None
    err.calc_long_Z()

    ax[0].plot(err.s*1e3, err.WP, c='g', lw=1.5, label='error')
    ax[1].plot(err.f*1e-9, np.real(err.Z), c='g', lw=1.5, label='Re(Z) error')
    ax[1].plot(err.f*1e-9, np.imag(err.Z), c='limegreen', lw=1.3, label='Im(Z) error')
    ax[1].plot(err.f*1e-9, np.abs(err.Z), c='g', lw=1.5, alpha=0.5, label='Abs(Z) error')

    ax[0].legend()
    ax[1].legend()

    fig.suptitle('Benchmark with CST Wakefield Solver')
    fig.tight_layout()
    fig.savefig(f'{results_folder}benchmarkReImAbs_error.png')

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

    fig, ax = plt.subplots(1,2, figsize=[12,4.5], dpi=150)

    # Read data
    beta = '0.4'
    keys = ['1', '10', '20', '30', '40', '50']
    colors = plt.cm.jet(np.linspace(0.1,0.9,len(keys)))
    res = {} 
    for i, k in enumerate(keys):
        # Wakis wake
        res[k] = wake.copy()
        res[k].load_results(f'results_beta{beta}_add{k}/')

        ax[0].plot(res[k].s, res[k].WP, c=colors[i], lw=1.5, alpha=0.8, label=r'$\beta=0.4$'+f' add={k}')
        ax[1].plot(res[k].f*1e-9, np.real(res[k].Z), c=colors[i], lw=1.5, alpha=0.8, label=r'Re: $\beta=0.4$'+f' add={k}')
        ax[1].plot(res[k].f*1e-9, np.imag(res[k].Z), c=colors[i], lw=1.5, ls=':', alpha=1.0, label=r'Im: $\beta=0.4$'+f' add={k}')
    
    # CST wake
    cstWP = wake.read_txt(f'cst/WP_beta{beta}.txt')
    cstZ = wake.read_txt(f'cst/Z_beta{beta}.txt')

    ax[0].plot(cstWP[0]*1e-3, cstWP[1], c='k', ls='--', lw=1.5, label=r'CST $\beta = 0.4$')
    ax[1].plot(cstZ[0], cstZ[1], c='k', ls='--', lw=1.5, label=r'CST Re: $\beta = 0.4$')
    ax[1].plot(cstZ[0], cstZ[2], c='k', ls=':', lw=1.5, label=r'CST Im: $\beta = 0.4$')
    
    ax[0].set_xlabel('s [m]')
    ax[0].set_ylabel('Longitudinal wake potential [V/pC]', color='k')
    #ax[0].set_yscale('symlog')
    ax[0].set_ylim(-30, 30)
    ax[0].legend()
    ax[0].margins(x=0.01, tight=True)

    ax[1].set_xlabel('f [GHz]')
    ax[1].set_ylabel(r'Longitudinal impedance [Re/Im][$\Omega$]', color='k')
    #ax[1].set_yscale('symlog')
    box = ax[1].get_position()
    ax[1].set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax[1].margins(x=0.01, tight=True)

    fig.suptitle('Benchmark with CST Wakefield Solver')
    #fig.tight_layout()
    
    fig.savefig(f'benchmark_addspace_beta{beta}.png')
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