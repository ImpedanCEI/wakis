import os, sys
import numpy as np
import pyvista as pv

sys.path.append('../wakis')

from wakis import SolverFIT3D
from wakis import GridFIT3D 
from wakis import WakeSolver

import pytest 

@pytest.mark.slow
class TestSmartMesh:

    zedges = [-0.05,-0.0075,0.0075,0.05]
    xedges = [-0.025,-0.005, 0.005, 0.025]
    yedges = [-0.025,-0.005, 0.005, 0.025]

    def test_grid_generation(self):
        print("\n---------- Initializing simulation ------------------")
        # Number of mesh cells
        Nx = 50
        Ny = 50
        Nz = 150

        # Embedded boundaries
        stl_file = 'tests/stl/001_cubic_cavity.stl' 
        surf = pv.read(stl_file)

        stl_solids = {'cavity': stl_file}
        stl_materials = {'cavity': 'vacuum'}

        # Domain bounds
        xmin, xmax, ymin, ymax, zmin, zmax = surf.bounds

        refinement_tol=1e-8
        snap_tol=1e-2
        # set grid and geometry
        grid = GridFIT3D(xmin, xmax, ymin, ymax, zmin, zmax, Nx, Ny, Nz, 
                        stl_solids=stl_solids, 
                        stl_materials=stl_materials,
                        use_mesh_refinement=True,
                        snap_tol=snap_tol,
                        refinement_tol=refinement_tol)
        for edg in self.zedges:
            diff = np.min(np.abs(grid.z-edg))
            assert diff <= refinement_tol + snap_tol, "Mesh not inside the tolerance at the z edges"

        for edg in self.yedges:
            diff = np.min(np.abs(grid.y-edg))
            assert diff <= refinement_tol + snap_tol, "Mesh not inside the tolerance at the y edges"

        for edg in self.xedges:
            diff = np.min(np.abs(grid.x-edg))
            assert diff <= refinement_tol + snap_tol, "Mesh not inside the tolerance at the x edges"