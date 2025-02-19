import os, sys
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from scipy.constants import c

sys.path.append('../wakis')
                
from wakis import SolverFIT3D
from wakis import GridFIT3D 
from wakis.sources import PlaneWave

import pytest 

flag_interactive = False # Set to true to run PyVista tests

class TestPlanewave:
    def test_simulation(self):
        print("\n---------- Initializing simulation ------------------")
        # Number of mesh cells
        Nx = 60
        Ny = 60
        Nz = 120

        # Embedded boundaries
        stl_file = 'tests/stl/003_sphere.stl' 
        surf = pv.read(stl_file)

        stl_solids = {'Sphere': stl_file}
        stl_materials = {'Sphere': [10.0, 1.0]} #dielectric [eps_r, mu_r]
        stl_rotate = [0, 0, 0]
        stl_scale = 1e-3

        surf = surf.rotate_x(stl_rotate[0])
        surf = surf.rotate_y(stl_rotate[1])
        surf = surf.rotate_z(stl_rotate[2])
        surf = surf.scale(stl_scale)

        # Domain bounds and grid
        xmin, xmax, ymin, ymax, zmin, zmax = surf.bounds
        padx, pady, padz = (xmax-xmin)*0.2, (ymax-ymin)*0.2, (zmax-zmin)*1.0

        xmin, ymin, zmin = (xmin-padx), (ymin-pady), (zmin-padz)
        xmax, ymax, zmax = (xmax+padx), (ymax+pady), (zmax+padz)

        grid = GridFIT3D(xmin, xmax, ymin, ymax, zmin, zmax, Nx, Ny, Nz, 
                stl_solids=stl_solids, 
                stl_rotate=stl_rotate,
                stl_scale=stl_scale,
                stl_materials=stl_materials)

        # Boundary conditions and 
        bc_low=['periodic', 'periodic', 'pec']
        bc_high=['periodic', 'periodic', 'pml']
        
        # simulation
        global solver
        solver = SolverFIT3D(grid, use_stl=True, bc_low=bc_low, bc_high=bc_high)
        
        # source
        f = 15/((solver.z.max()-solver.z.min())/c)
        source = PlaneWave(xs=slice(1, Nx-1), ys=slice(1,Ny-1), zs=1, 
                           f=f, beta=1.0)
        
        Nt = int(1.0*(solver.z.max()-solver.z.min())/c/solver.dt)
        solver.emsolve(Nt, source)

    def test_plot2D(self):
        global solver
        solver.plot2D('Ex', plane='ZY', pos=0.5, cmap='rainbow', 
                    add_patch='Sphere', patch_alpha=0.3, 
                    off_screen=False)
        if not flag_interactive:
            plt.close()
            
    def test_plot2D_offscreen(self):
        global solver
        solver.plot2D('Hy', plane='ZY', pos=0.5, cmap='bwr', 
                    add_patch='Sphere', patch_alpha=0.1, interpolation='spline36',
                    off_screen=True, n=solver.Nt, title='003_2Dplot_Hy') 
        if not flag_interactive:
            os.remove(f'003_2Dplot_Hy_{str(solver.Nt).zfill(6)}.png')
    
    @pytest.mark.skipif(not flag_interactive, reason="Requires interactive plotting")
    def test_plot3D_interactive(self):
        global solver
        solver.plot3D(field='E', component='x', cmap='jet',
            add_stl='Sphere', stl_opacity=0.1, stl_colors='white',
            clip_interactive=True, clip_normal='-y',
            off_screen=False)  
        
    @pytest.mark.skipif(not flag_interactive, reason="Requires Xserver connection for plotting")     
    def test_plot3D_offscreen(self):
        global solver
        solver.plot3D(field='H', component='y', cmap='bwr',
                      add_stl='Sphere', stl_opacity=0.1, stl_colors='white',
                      clip_box=True, clip_bounds=None, 
                      off_screen=True, title='003_3Dplot_Hy')
        if not flag_interactive:
            os.remove('003_3Dplot_Hy.png')
    
    @pytest.mark.skipif(not flag_interactive, reason="Requires interactive plotting")
    def test_plot3DonSTL_interactive(self):
        global solver
        solver.plot3DonSTL('Ex', cmap='jet', 
                           stl_with_field='Sphere', field_opacity=1.,
                           stl_transparent='Sphere', stl_opacity=0.3, stl_colors='white',
                           clip_interactive=True, clip_normal='-y',
                           off_screen=False, zoom=1.0)
        
    @pytest.mark.skipif(not flag_interactive, reason="Requires Xserver connection for plotting")
    def test_plot3DonSTL_offscreen(self):
        global solver
        solver.plot3DonSTL('Ex', cmap='jet', 
                           stl_with_field='Sphere', field_opacity=1.,
                           stl_transparent=None, stl_opacity=0., stl_colors=None,
                           off_screen=True, zoom=1.0, title='003_3DplotOnSTL_Hy')
        if not flag_interactive:
            os.remove('003_3DplotOnSTL_Hy.png')
        
