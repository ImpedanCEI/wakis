import sys,os
import numpy as np
from scipy.constants import c, mu_0, epsilon_0
from tqdm import tqdm

sys.path.append('../')
import wakis

import pytest

flag_interactive = False # Set to true to run plot tests

class TestPML:

    def test_timestep(self):
        #TODO: check for dt < relaxation time
        pass

    def test_reflection_planewave(self):
        print("\n---------- Initializing simulation ------------------")
        # Domain bounds and grid
        xmin, xmax = -1., 1.
        ymin, ymax = -1., 1.
        zmin, zmax = 0., 10.

        Nx, Ny = 20, 20
        Nz = 200

        grid = wakis.GridFIT3D(xmin, xmax, ymin, ymax, zmin, zmax, 
                            Nx, Ny, Nz)
        
        # Boundary conditions and solver
        bc_low = ['periodic', 'periodic', 'pec']
        bc_high = ['periodic', 'periodic', 'pml']


        solver = wakis.SolverFIT3D(grid, use_stl=False,
                                bc_low=bc_low, bc_high=bc_high,
                                n_pml=10)

        # Source
        amplitude = 100.
        nodes = 7
        f = 15/((solver.z.max()-solver.z.min())/c)
        planeWave = wakis.sources.PlaneWave(xs=slice(0, Nx), ys=slice(0,Ny), zs=0, 
                                            beta=1.0, amplitude=amplitude,
                                            f=f, nodes=nodes, phase=np.pi/2)
        solver.dt = 1/f/200 #ensure right amplitude

        # Simulation
        Nt = int(2.0*(zmax-zmin-solver.n_pml*solver.dz)/c/solver.dt)

        for n in tqdm(range(Nt)):
            planeWave.update(solver, n*solver.dt)
            solver.one_step()

            if flag_interactive and n%int(Nt/100) == 0:
                solver.plot1D('Hy', ylim=(-amplitude, amplitude), pos=[0.5, 0.35, 0.2, 0.1],
                               off_screen=True, title=f'005_Hy', n=n)
                solver.plot1D('Ex', ylim=(-amplitude*c*mu_0, amplitude*c*mu_0), pos=[0.5, 0.35, 0.2, 0.1],
                               off_screen=True, title=f'005_Ex', n=n)                
            
        reflectionH = solver.H.get_abs()[Nx//2,Ny//2,:].max()
        reflectionE = solver.E.get_abs()[Nx//2,Ny//2,:].max()/(mu_0*c)
        assert reflectionH == pytest.approx(10, 0.1), "PML H reflection >10%"
        assert reflectionE == pytest.approx(12, 0.2), "PML E reflection >10%"

        if flag_interactive:
            os.system(f'convert -delay 10 -loop 0 005_Hy*.png 005_Hy_planewave.gif')
            os.system(f'convert -delay 10 -loop 0 005_Ex*.png 005_Ex_planewave.gif')
            os.system(f'rm 005_Hy*.png')
            os.system(f'rm 005_Ex*.png')

            solver.plot2D('Ex', plane='ZX', pos=0.5, cmap='bwr', 
                interpolation='spline36', n=n, vmin=-amplitude*c*mu_0, vmax=amplitude*c*mu_0,
                off_screen=True, title=f'005_Ex2d') 

            solver.plot2D('Hy', plane='ZX', pos=0.5, cmap='bwr', 
                interpolation='spline36', n=n, vmin=-amplitude, vmax=amplitude,
                off_screen=True, title=f'005_Hy2d') 