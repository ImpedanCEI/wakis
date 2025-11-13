# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

import numpy as np
import pyvista as pv
from functools import partial
from scipy.optimize import least_squares

from .field import Field

try:
    from mpi4py import MPI
    imported_mpi = True
except ImportError:
    imported_mpi = False

class GridFIT3D:
    """
    Class holding the grid information and 
    stl importing handling using PyVista

    Parameters
    ----------
    xmin, xmax, ymin, ymax, zmin, zmax: float
        extent of the domain.
    Nx, Ny, Nz: int
        number of cells per direction
    stl_solids: dict, optional
        stl files to import in the domain.
        {'Solid 1': stl_1, 'Solid 2': stl_2, ...}
        If stl files are not in the same folder,
        add the path to the file name.
    stl_materials: dict, optional
        Material properties associated with stl
        {'Solid 1': [eps1, mu1],
         'Solid 2': [eps1, mu1], 
         ...}
    stl_rotate: list or dict, optional
        Angle of rotation to apply to the stl models: [rot_x, rot_y, rot_z]
        - if list, it will be applied to all stls in `stl_solids`
        - if dict, it must contain the same keys as `stl_solids`, 
          indicating the rotation angle per stl
    stl_scale: float or dict, optional
        Scaling value to apply to the stl model to convert to [m]
        - if float, it will be applied to all stl in `stl_solids`
        - if dict, it must contain the same keys as `stl_solids` 
    tol: float, default 1e-3
        Tolerance factor for stl import, used in grid.select_enclosed_points.
        Importing tolerance is computed by: tol*min(dx,dy,dz). 
    verbose: int or bool, default 1
        Enable verbose ouput on the terminal
    """

    def __init__(self, xmin, xmax, ymin, ymax, zmin, zmax, 
                Nx, Ny, Nz, 
                x=None, y=None, z=None, 
                use_mpi=False, 
                use_mesh_refinement=False, refinement_method='insert', refinement_tol=1e-8,
                snap_points=None, snap_tol=1e-5, snap_solids=None,
                stl_solids=None, stl_materials=None, 
                stl_rotate=[0., 0., 0.], stl_translate=[0., 0., 0.], stl_scale=1.0,
                stl_colors=None, verbose=1, stl_tol=1e-3):
        
        if verbose: print('Generating grid...')
        self.verbose = verbose
        self.use_mpi = use_mpi
        self.use_mesh_refinement = use_mesh_refinement

        # domain limits
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.zmin = zmin
        self.zmax = zmax
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.dx = (xmax - xmin) / Nx
        self.dy = (ymax - ymin) / Ny
        self.dz = (zmax - zmin) / Nz
        
        # stl info
        self.stl_solids = stl_solids
        self.stl_materials = stl_materials
        self.stl_rotate = stl_rotate
        self.stl_translate = stl_translate
        self.stl_scale = stl_scale
        self.stl_colors = stl_colors
        if stl_solids is not None:
            self._prepare_stl_dicts()

        # MPI subdivide domain 
        if self.use_mpi: 
            self.ZMIN = None
            self.ZMAX = None
            self.NZ = None
            self.Z = None
            if imported_mpi:
                self.mpi_initialize()
                if self.verbose: print(f"MPI initialized for {self.rank} of {self.size}")
            else:
                raise ImportError("*** mpi4py is required when use_mpi=True but was not found")
            
        # primal Grid G base axis x, y, z
        self.x = x
        self.y = y
        self.z = z
        self.refinement_method = refinement_method
        self.snap_points = snap_points
        self.snap_tol = snap_tol
        self.snap_solids = snap_solids # if None, use all stl_solids

        if self.x is not None and self.y is not None and self.z is not None:
            # allow user to set the grid axis manually
            self.Nx = len(self.x) - 1
            self.Ny = len(self.y) - 1
            self.Nz = len(self.z) - 1
            self.dx = np.min(np.diff(self.x)) 
            self.dy = np.min(np.diff(self.y))  
            self.dz = np.min(np.diff(self.z))

        elif self.use_mesh_refinement:
            if verbose: print('Applying mesh refinement...')
            if self.snap_points is None and stl_solids is not None:
                self.compute_snap_points(snap_solids=snap_solids, snap_tol=snap_tol)
            self.refine_xyz_axis(method=refinement_method, tol=refinement_tol)  # obtain self.x, self.y, self.z
        else:
            self.x = np.linspace(self.xmin, self.xmax, self.Nx+1)
            self.y = np.linspace(self.ymin, self.ymax, self.Ny+1)
            self.z = np.linspace(self.zmin, self.zmax, self.Nz+1)
            
        # grid G and tilde grid ~G, lengths and inverse areas
        self.compute_grid()

        # tolerance for stl import tol*min(dx,dy,dz)
        if verbose: print('Importing STL solids...')
        self.tol = stl_tol 
        if stl_solids is not None:
            self.mark_cells_in_stl()
            if stl_colors is None:
                self.assign_colors()

    def compute_grid(self):
        X, Y, Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        self.grid = pv.StructuredGrid(X.transpose(), Y.transpose(), Z.transpose())

        self.L = Field(self.Nx, self.Ny, self.Nz)
        self.L.field_x = X[1:, 1:, 1:] - X[:-1, :-1, :-1]
        self.L.field_y = Y[1:, 1:, 1:] - Y[:-1, :-1, :-1]
        self.L.field_z = Z[1:, 1:, 1:] - Z[:-1, :-1, :-1]

        self.iA = Field(self.Nx, self.Ny, self.Nz)
        self.iA.field_x = np.divide(1.0, self.L.field_y * self.L.field_z)
        self.iA.field_y = np.divide(1.0, self.L.field_x * self.L.field_z)
        self.iA.field_z = np.divide(1.0, self.L.field_x * self.L.field_y)

        # tilde grid ~G
        self.tx = (self.x[1:]+self.x[:-1])/2 
        self.ty = (self.y[1:]+self.y[:-1])/2
        self.tz = (self.z[1:]+self.z[:-1])/2

        self.tx = np.append(self.tx, self.tx[-1])
        self.ty = np.append(self.ty, self.ty[-1])
        self.tz = np.append(self.tz, self.tz[-1])

        tX, tY, tZ = np.meshgrid(self.tx, self.ty, self.tz, indexing='ij')

        self.tL = Field(self.Nx, self.Ny, self.Nz)
        self.tL.field_x = tX[1:, 1:, 1:] - tX[:-1, :-1, :-1]
        self.tL.field_y = tY[1:, 1:, 1:] - tY[:-1, :-1, :-1]
        self.tL.field_z = tZ[1:, 1:, 1:] - tZ[:-1, :-1, :-1]

        self.itA = Field(self.Nx, self.Ny, self.Nz)
        aux = self.tL.field_y * self.tL.field_z
        self.itA.field_x = np.divide(1.0, aux, out=np.zeros_like(aux), where=aux!=0)
        aux = self.tL.field_x * self.tL.field_z
        self.itA.field_y = np.divide(1.0, aux, out=np.zeros_like(aux), where=aux!=0)
        aux = self.tL.field_x * self.tL.field_y
        self.itA.field_z = np.divide(1.0, aux, out=np.zeros_like(aux), where=aux!=0)
        del aux
    
    def mpi_initialize(self):
        comm = MPI.COMM_WORLD  # Get MPI communicator
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size() 

        # Error handling for Nz < size
        if self.Nz < self.size:
            raise ValueError(f"Nz ({self.Nz}) must be greater than or equal to the number of MPI processes ({self.size}).")
        
        # global z quantities [ALLCAPS]
        self.ZMIN = self.zmin
        self.ZMAX = self.zmax
        self.NZ = self.Nz - self.Nz%(self.size) # ensure multiple of MPI size
        self.Z = np.linspace(self.ZMIN, self.ZMAX, self.NZ+1)[:-1] + self.dz/2

        if self.verbose and self.rank==0: 
            print(f"Global grid ZMIN={self.ZMIN}, ZMAX={self.ZMAX}, NZ={self.NZ}")

        # MPI subdomain quantities 
        self.Nz = self.NZ // (self.size) 
        self.dz = (self.ZMAX - self.ZMIN) / self.NZ
        self.zmin = self.rank * self.Nz * self.dz + self.ZMIN
        self.zmax = (self.rank+1) * self.Nz * self.dz + self.ZMIN 

        if self.verbose: print(f"MPI rank {self.rank} of {self.size} initialized with \
                                zmin={self.zmin}, zmax={self.zmax}, Nz={self.Nz}")
        # Add ghost cells
        self.n_ghosts = 1
        if self.rank > 0:
            self.zmin += - self.n_ghosts * self.dz
            self.Nz += self.n_ghosts
        if self.rank < (self.size-1):
            self.zmax += self.n_ghosts * self.dz
            self.Nz += self.n_ghosts

        # Support for single core
        if self.rank == 0 and self.size == 1: 
            self.zmax += self.n_ghosts * self.dz
            self.Nz += self.n_ghosts
            
    def mpi_gather_asGrid(self):
        _grid = None
        if self.rank == 0:
            print(f"Generating global grid from {self.ZMIN} to {self.ZMAX}")
            _grid = GridFIT3D(self.xmin, self.xmax, 
                            self.ymin, self.ymax,
                            self.ZMIN, self.ZMAX,
                            self.Nx, self.Ny, self.NZ,
                            use_mpi=False,
                            stl_solids=self.stl_solids,
                            stl_materials=self.stl_materials,
                            stl_scale=self.stl_scale,
                            stl_rotate=self.stl_rotate,
                            stl_translate=self.stl_translate,
                            stl_colors=self.stl_colors,
                            verbose=self.verbose,
                            stl_tol=self.tol,
                            )
        return _grid

    def _prepare_stl_dicts(self):
        if type(self.stl_solids) is not dict:
            if type(self.stl_solids) is str:
                self.stl_solids = {'Solid 1' : self.stl_solids}
            else:
                raise Exception('Attribute `stl_solids` must contain a string or a dictionary')

        if type(self.stl_rotate) is not dict:
            # if not a dict, the same values will be applied to all solids
            stl_rotate = {}
            for key in self.stl_solids.keys():
                stl_rotate[key] = self.stl_rotate
            self.stl_rotate = stl_rotate

        if type(self.stl_scale) is not dict:
            # if not a dict, the same values will be applied to all solids
            stl_scale = {}
            for key in self.stl_solids.keys():
                stl_scale[key] = self.stl_scale
            self.stl_scale = stl_scale

        if type(self.stl_translate) is not dict:
            # if not a dict, the same values will be applied to all solids
            stl_translate = {}
            for key in self.stl_solids.keys():
                stl_translate[key] = self.stl_translate
            self.stl_translate = stl_translate

    def mark_cells_in_stl(self):
        # Obtain masks with grid cells inside each stl solid
        stl_tolerance = np.min([self.dx, self.dy, self.dz])*self.tol
        for key in self.stl_solids.keys():

            surf = self.read_stl(key)

            # mark cells in stl [True == in stl, False == out stl]
            try:
                select = self.grid.select_enclosed_points(surf, tolerance=stl_tolerance)
            except:
                select = self.grid.select_enclosed_points(surf, tolerance=stl_tolerance, check_surface=False)
            self.grid[key] = select.point_data_to_cell_data()['SelectedPoints'] > stl_tolerance

    def read_stl(self, key):
        # import stl
        surf = pv.read(self.stl_solids[key])

        # rotate
        surf = surf.rotate_x(self.stl_rotate[key][0])  
        surf = surf.rotate_y(self.stl_rotate[key][1])  
        surf = surf.rotate_z(self.stl_rotate[key][2])  

        # translate
        surf = surf.translate(self.stl_translate[key])

        # scale
        surf = surf.scale(self.stl_scale[key]) 

        return surf
    
    def compute_snap_points(self, snap_solids=None, snap_tol=1e-8):
        if self.verbose: print('* Calculating snappy points...')
        # Support for user-defined stl_keys as list
        if snap_solids is None:
            snap_solids = self.stl_solids.keys()

        # Union of all the surfaces
        # [TODO]: should use | for union instead or +?
        model = None
        for key in snap_solids:
            solid = self.read_stl(key)
            if model is None:
                model = solid
            else:
                model = model + solid  
    
        edges = model.extract_feature_edges(boundary_edges=True, manifold_edges=False)

        # Extract points lying in the X-Z plane (Y ≈ 0)
        xz_plane_points = edges.points[np.abs(edges.points[:, 1]) < snap_tol]
        # Extract points lying in the Y-Z plane (X ≈ 0)
        yz_plane_points = edges.points[np.abs(edges.points[:, 0]) < snap_tol]
        # Extract points lying in the X-Y plane (Z ≈ 0)
        xy_plane_points = edges.points[np.abs(edges.points[:, 2]) < snap_tol]

        self.snap_points = np.r_[xz_plane_points, yz_plane_points, xy_plane_points]

        # get the unique x, y, z coordinates
        x_snaps = np.unique(np.round(self.snap_points[:, 0], 5))
        y_snaps = np.unique(np.round(self.snap_points[:, 1], 5))
        z_snaps = np.unique(np.round(self.snap_points[:, 2], 5))

        # Include simulation domain bounds
        self.x_snaps = np.unique(np.concatenate(([self.xmin], x_snaps, [self.xmax])))
        self.y_snaps = np.unique(np.concatenate(([self.ymin], y_snaps, [self.ymax])))
        self.z_snaps = np.unique(np.concatenate(([self.zmin], z_snaps, [self.zmax])))

    def plot_snap_points(self, snap_solids=None, snap_tol=1e-8):
        # TODO
        # Support for user-defined stl_keys as list
        if snap_solids is None:
            snap_solids = self.stl_solids.keys()

        # Union of all the surfaces
        # [TODO]: should use | for union instead or +?
        model = None
        for key in snap_solids:
            solid = self.read_stl(key)
            if model is None:
                model = solid
            else:
                model = model + solid  
    
        edges = model.extract_feature_edges(boundary_edges=True, manifold_edges=False)

        # Extract points lying in the X-Z plane (Y ≈ 0)
        xz_plane_points = edges.points[np.abs(edges.points[:, 1]) < snap_tol]
        # Extract points lying in the Y-Z plane (X ≈ 0)
        yz_plane_points = edges.points[np.abs(edges.points[:, 0]) < snap_tol]
        # Extract points lying in the X-Y plane (Z ≈ 0)
        xy_plane_points = edges.points[np.abs(edges.points[:, 2]) < snap_tol]

        xz_cloud = pv.PolyData(xz_plane_points)
        yz_cloud = pv.PolyData(yz_plane_points)
        xy_cloud = pv.PolyData(xy_plane_points)

        pl = pv.Plotter()
        pl.add_mesh(model, color='white', opacity=0.5, label='base STL')
        pl.add_mesh(edges, color='black', line_width=5, opacity=0.8,)
        pl.add_mesh(xz_cloud, color='green', point_size=20, render_points_as_spheres=True, label='XZ plane points')
        pl.add_mesh(yz_cloud, color='orange', point_size=20, render_points_as_spheres=True, label='YZ plane points')
        pl.add_mesh(xy_cloud, color='magenta', point_size=20, render_points_as_spheres=True, label='XY plane points')
        pl.add_legend()
        pl.show()

    def refine_axis(self, xmin, xmax, Nx, x_snaps, 
                    method='insert', tol=1e-12):

        # Loss function to minimize cell size spread
        def loss_function(x, x0, is_snap):
            # avoid moving snap points
            penalty_snap = np.sum((x[is_snap] - x0[is_snap])**2) * 1000
            # avoid gaps < uniform gap
            dx = np.diff(x)
            threshold = 1/(len(x)-1) # or a hardcoded `min_spacing`
            penalty_small_gaps = np.sum((threshold - dx[dx < threshold])**2) 
            # avoid large spread in gap length
            dx = np.diff(x)
            penalty_variance = np.std(dx) * 10
            #return penalty_snap + penalty_small_gaps + penalty_variance
            return np.hstack([penalty_snap, penalty_small_gaps, penalty_variance])

        # Uniformly distributed points as initial guess
        x_snaps = (x_snaps-xmin)/(xmax-xmin) # normalize to [0,1]

        if method == 'insert':
            x0 = np.unique(np.append(x_snaps,  np.linspace(0, 1, Nx - len(x_snaps))))

        elif method == 'neighbor':
            x = np.linspace(0, 1, Nx)
            dx = np.diff(x)[0]
            mask = np.zeros_like(x, dtype=bool)
            i=0
            for s in x_snaps:
                m = np.isclose(x, s, rtol=0.0, atol=dx/2)
                if np.sum(m)>0:
                    x[np.argmax(m)] = s
            x0 = x.copy()
        
        elif method == 'subdivision':
            # x = snaps
            while len(x) < Nx:
                #idx of segments sorted min -> max
                idx_max_diffs = np.argsort(np.diff(x))[-1] # take bigger

                print(f"Bigger segment starts at {x[idx_max_diffs]}")
                # compute new point in the middle of the segment
                val = x[idx_max_diffs] + (x[idx_max_diffs + 1] - x[idx_max_diffs]) / 2

                # insert the new point
                x = np.insert(x, idx_max_diffs+1, val)
                x = np.unique(x)
                print(f"Inserted point {val} at index {idx_max_diffs}")
            x0 = x.copy()
        else:
            raise ValueError(f"Method {method} not supported. Use 'insert', 'neighbor' or 'subdivision'.")

        # minimize segment length spread for the test points 
        is_snap = np.isin(x0, x_snaps)
        result = least_squares(loss_function,
                                x0=x0.copy(),
                                bounds=(0,1),#(zmin, zmax),
                                jac='3-point',
                                method='dogbox',
                                loss='arctan',
                                gtol=tol,
                                ftol=tol,
                                xtol=tol,
                                verbose=1,
                                args=(x0.copy(), is_snap.copy()),
                                )
        # transform back to [xmin, xmax]
        return result.x*(xmax-xmin)+xmin 

    def refine_xyz_axis(self, method='insert', tol=1e-6):
        '''Refine the grid in the x, y, z axis
        using the snap points extracted from the stl solids.
        The snap points are used to refine the grid '''

        if self.verbose: print(f'* Refining x axis with {len(self.x_snaps)} snaps...')
        self.x = self.refine_axis(self.xmin, self.xmax, self.Nx+1, self.x_snaps,
                                  method=method, tol=tol)
        
        if self.verbose: print(f'* Refining y axis with {len(self.y_snaps)} snaps...')
        self.y = self.refine_axis(self.ymin, self.ymax, self.Ny+1, self.y_snaps,
                                  method=method, tol=tol)
        
        if self.verbose: print(f'* Refining z axis with {len(self.z_snaps)} snaps...')
        self.z = self.refine_axis(self.zmin, self.zmax, self.Nz+1, self.z_snaps,
                                  method=method, tol=tol)

        self.Nx = len(self.x) - 1
        self.Ny = len(self.y) - 1       
        self.Nz = len(self.z) - 1
        self.dx = np.min(np.diff(self.x))  #TODO: should this be an array?
        self.dy = np.min(np.diff(self.y))  
        self.dz = np.min(np.diff(self.z))

        print(f"Refined grid: Nx = {len(self.x)}, Ny ={len(self.y)}, Nz = {len(self.z)}")

    def assign_colors(self):
        '''Classify colors assigned to each solid
        based on the categories in `material_colors` dict
        inside `materials.py`
        '''
        self.stl_colors = {}

        for key in self.stl_solids:
            mat = self.stl_materials[key]
            if type(mat) is str:
                self.stl_colors[key] = mat
            elif len(mat) == 2: 
                if mat[0] is np.inf:  #eps_r
                    self.stl_colors[key] = 'pec'
                elif mat[0] > 1.0:    #eps_r
                    self.stl_colors[key] = 'dielectric'
                else:
                    self.stl_colors[key] = 'vacuum'
            elif len(mat) == 3:
                self.stl_colors[key] = 'lossy metal'
            else:
                self.stl_colors[key] = 'other'

    def plot_solids(self, bounding_box=False, show_grid=False, anti_aliasing=None,
                    opacity=1.0, specular=0.5, offscreen=False, **kwargs):
        """
        Generates a 3D visualization of the imported STL geometries using PyVista.

        Parameters:
        -----------
        bounding_box : bool, optional
            If True, adds a bounding box around the plotted geometry (default: False).

        show_grid : bool, optional
            If True, adds the grid's mesh wireframe to the display (default: False).   

        anti_aliasing : str or None, optional
            Enables anti-aliasing if provided. Valid values depend on PyVista settings (default: None).
        
        opacity : float, optional
            Controls the transparency of the plotted solids. A value of 1.0 is fully opaque, 
            while 0.0 is fully transparent (default: 1.0).
        
        specular : float, optional
            Adjusts the specular lighting effect on the surface. Higher values increase shininess (default: 0.5).
        
        **kwargs : dict
            Additional keyword arguments passed to `pyvista.add_mesh()`, allowing customization of the mesh rendering.
        
        Notes:
        ------
        - Colors are determined by the `GridFIT3D.stl_colors` attribute dictionary if not None
        - Solids labeled as 'vacuum' are rendered with a default opacity of 0.3 for visibility.e.
        - The camera is positioned at an angle to provide better depth perception.
        - If `bounding_box=True`, a bounding box is drawn around the model.
        - If `anti_aliasing` is specified, it is enabled to improve rendering quality.
        
        """

        from .materials import material_colors
        
        pl = pv.Plotter()
        pl.add_mesh(self.grid, opacity=0., name='grid', show_scalar_bar=False)
        for key in self.stl_solids:
            try:
                color = material_colors[self.stl_colors[key]] # match library e.g. 'vacuum'
            except: 
                color = self.stl_colors[key] # specifies color e.g. 'tab:red'

            if self.stl_colors[key] == 'vacuum' or self.stl_materials[key] == 'vacuum':
                _opacity = 0.3
            else:
                _opacity = opacity
            pl.add_mesh(self.read_stl(key), color=color, 
                        opacity=_opacity, specular=specular, smooth_shading=True,
                        **kwargs)
        
        pl.set_background('mistyrose', top='white')
        try: pl.add_logo_widget('../docs/img/wakis-logo-pink.png')
        except: pass
        pl.camera_position = 'zx'
        pl.camera.azimuth += 30
        pl.camera.elevation += 30
        pl.add_axes()

        if anti_aliasing is not None:
            pl.enable_anti_aliasing(anti_aliasing)

        if bounding_box:
            pl.add_bounding_box()
            
        if show_grid:
            pl.add_mesh(self.grid, style='wireframe', color='grey', opacity=0.3, name='grid')

        if offscreen:
            pl.export_html('grid_plot_solids.html')
        else:
            pl.show()

    def plot_stl_mask(self, stl_solid, cmap='viridis', bounding_box=True, show_grid=True,
                      add_stl='all', stl_opacity=0., stl_colors=None,
                      xmax=None, ymax=None, zmax=None,
                      anti_aliasing='ssaa', offscreen=False):
        
        """
        Interactive 3D visualization of the structured grid mask and imported STL geometries.

        This routine uses PyVista to display the grid scalar field corresponding to a
        chosen STL mask. It provides interactive slider widgets to clip the domain
        along the X, Y, and Z directions. At each slider position, the clipped
        scalar field is shown with a colormap while the grid structure is shown
        as a 2D slice in wireframe. Optionally, one or more STL geometries can
        be added to the scene, along with a bounding box of the simulation domain.

        Parameters
        ----------
        stl_solid : str
            Key name of the `stl_solids` dictionary to retrieve the mask for
            visualization (used as the scalar field).
        cmap : str, default 'viridis'
            Colormap used to visualize the clipped scalar values.
        bounding_box : bool, default True
            If True, add a static wireframe bounding box of the simulation domain.
        show_grid : bool, default True
            If True, adds the computational grid overlay on the clipped slice
        add_stl : {'all', str, list[str]}, default 'all'
            STL geometries to add:
            * 'all' → add all STL solids found in `self.stl_solids`
            * str   → add a single STL solid by key
            * list  → add a list of STL solids by key
            If None, no STL surfaces are added.
        stl_opacity : float, default 0.0
            Opacity of the STL surfaces (0 = fully transparent, 1 = fully opaque).
        stl_colors : str, list[str], dict, or None, default None
            Color(s) of the STL surfaces:
            * str   → single color for all STL surfaces
            * list  → per-solid colors, in order
            * dict  → mapping from STL key to color (using `material_colors`)
            * None  → use default colors from `self.stl_colors`
        xmax, ymax, zmax : float, optional
            Initial clipping positions along each axis. If None, use the
            maximum domain extent.
        anti_aliasing : {'ssaa', 'fxaa', None}, default 'ssaa'
            Anti-aliasing mode passed to `pl.enable_anti_aliasing`.
        offscreen : bool, default False
            If True, render offscreen and export the scene to
            ``grid_stl_mask_<stl_solid>.html``. If False, open an interactive window.

        Notes
        -----
        * Three sliders (X, Y, Z) control clipping of the scalar field by a box
        along the respective axis. The clipped scalar field is shown with the
        given colormap. A simultaneous 2D slice of the grid is displayed in
        wireframe at the clip location.
        * STL solids can be visualized in transparent mode to show the relation
        between the structured grid and the geometry.
        * A static domain bounding box can be added for reference.
        * Camera, background, and lighting are pre-configured for clarity
        """
        from .materials import material_colors
        if stl_colors is None:
            stl_colors = self.stl_colors

        if xmax is None: xmax = self.xmax
        if ymax is None: ymax = self.ymax
        if zmax is None: zmax = self.zmax

        pv.global_theme.allow_empty_mesh = True
        pl = pv.Plotter()
        vals = {'x':xmax, 'y':ymax, 'z':zmax}

        # --- Update function ---
        def update_clip(val, axis="x"):
            vals[axis] = val
            # define bounds dynamically
            if axis == "x":
                slice_obj = self.grid.slice(normal="x", origin=(val, 0, 0))
            elif axis == "y":
                slice_obj = self.grid.slice(normal="y", origin=(0, val, 0))
            else:  # z
                slice_obj = self.grid.slice(normal="z", origin=(0, 0, val))

            # add clipped volume (scalars)
            pl.add_mesh(
                self.grid.clip_box(bounds=(self.xmin, vals['x'], 
                                           self.ymin, vals['y'], 
                                           self.zmin, vals['z']), invert=False),
                scalars=stl_solid,
                cmap=cmap,
                name="clip",
            )

            # add slice wireframe (grid structure)
            if show_grid:
                pl.add_mesh(slice_obj, style="wireframe", color="grey", name="slice")

        # Plot stl surface(s)
        if add_stl is not None: 
            if type(add_stl) is str: #add all stl solids
                if add_stl.lower() == 'all':
                    for i, key in enumerate(self.stl_solids):
                        surf = self.read_stl(key)
                        if type(stl_colors) is dict:
                            if type(stl_colors[key]) is str:
                                pl.add_mesh(surf, color=material_colors[stl_colors[key]], opacity=stl_opacity, silhouette=dict(color=material_colors[stl_colors[key]]), name=key)
                            else:
                                pl.add_mesh(surf, color=stl_colors[key], opacity=stl_opacity, silhouette=dict(color=stl_colors[key]), name=key)
                        elif type(stl_colors) is list:
                            pl.add_mesh(surf, color=stl_colors[i], opacity=stl_opacity, silhouette=dict(color=stl_colors[i]), name=key)
                        else:
                            pl.add_mesh(surf, color='white', opacity=stl_opacity, silhouette=True, name=key)
                else: #add 1 selected stl solid
                    key = add_stl
                    surf = self.read_stl(key)
                    if type(stl_colors[key]) is str:
                        pl.add_mesh(surf, color=material_colors[stl_colors[key]], opacity=stl_opacity, silhouette=dict(color=material_colors[stl_colors[key]]), name=key)
                    else:
                        pl.add_mesh(surf, color=stl_colors[key], opacity=stl_opacity, silhouette=dict(color=stl_colors[key]), name=key)

            elif type(add_stl) is list: #add selected list of stl solids
                for i, key in enumerate(add_stl):
                    surf = self.read_stl(key)
                    if type(stl_colors[key]) is dict:
                        if type(stl_colors) is str:
                            pl.add_mesh(surf, color=material_colors[stl_colors[key]], opacity=stl_opacity, silhouette=dict(color=material_colors[stl_colors[key]]), name=key)
                        else:
                            pl.add_mesh(surf, color=stl_colors[key], opacity=stl_opacity, silhouette=dict(color=stl_colors[key]), name=key)
                    elif type(stl_colors) is list:
                        pl.add_mesh(surf, color=stl_colors[i], opacity=stl_opacity, silhouette=dict(color=stl_colors[i]), name=key)
                    else:
                        pl.add_mesh(surf, color='white', opacity=stl_opacity, silhouette=True, name=key)

        # --- Sliders (placed side-by-side vertically) ---
        pl.add_slider_widget(
            lambda val: update_clip(val, "x"),
            [self.xmin, self.xmax],
            value=xmax, title="X Clip",
            pointa=(0.8, 0.8), pointb=(0.95, 0.8),  # top-right
            style='modern',
        )

        pl.add_slider_widget(
            lambda val: update_clip(val, "y"),
            [self.ymin, self.ymax],
            value=ymax, title="Y Clip",
            pointa=(0.8, 0.6), pointb=(0.95, 0.6),  # middle-right
            style='modern',
        )

        pl.add_slider_widget(
            lambda val: update_clip(val, "z"),
            [self.zmin, self.zmax],
            value=zmax, title="Z Clip",
            pointa=(0.8, 0.4), pointb=(0.95, 0.4),  # lower-right
            style='modern',
        )

        # Camera orientation
        pl.camera_position = 'zx'
        pl.camera.azimuth += 30
        pl.camera.elevation += 30
        pl.set_background('mistyrose', top='white')
        try: pl.add_logo_widget('../docs/img/wakis-logo-pink.png')
        except: pass
        pl.add_axes()
        pl.enable_3_lights()
        pl.enable_anti_aliasing(anti_aliasing)

        if bounding_box:
            pl.add_mesh(pv.Box(bounds=(self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax)),
            style="wireframe", color="black", line_width=2, name="domain_box")

        if offscreen:
            pl.export_html(f'grid_stl_mask_{stl_solid}.html')
        else:
            pl.show()

    def inspect(self, add_stl=None, stl_opacity=0.5, stl_colors=None,
                anti_aliasing='ssaa', offscreen=False):
        
        '''3D plot using pyvista to visualize 
        the structured grid and
        the imported stl geometries

        Parameters
        ---
        add_stl: str or list, optional
            List or str of stl solids to add to the plot by `pv.add_mesh`
        stl_opacity: float, default 0.1
            Opacity of the stl surfaces (0 - Transparent, 1 - Opaque)
        stl_colors: str or list of str, default 'white'
            Color of the stl surfaces
        '''
        from .materials import material_colors
        if stl_colors is None:
            stl_colors = self.stl_colors

        pv.global_theme.allow_empty_mesh = True
        pl = pv.Plotter()
        pl.add_mesh(self.grid, show_edges=True, cmap=['white', 'white'], name='grid')
        def clip(widget):
            # Plot structured grid
            b = widget.bounds
            x = self.x[np.logical_and(self.x>=b[0], self.x<=b[1])]
            y = self.y[np.logical_and(self.y>=b[2], self.y<=b[3])]
            z = self.z[np.logical_and(self.z>=b[4], self.z<=b[5])]
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            grid = pv.StructuredGrid(X.transpose(), Y.transpose(), Z.transpose())

            pl.add_mesh(grid, show_edges=True, cmap=['white', 'white'], name='grid')
            # Plot stl surface(s)
            if add_stl is not None: #add 1 selected stl solid
                if type(add_stl) is str:
                    key = add_stl
                    surf = self.read_stl(key)
                    surf = surf.clip_box(widget.bounds, invert=False)
                    if type(stl_colors[key]) is str:
                        pl.add_mesh(surf, color=material_colors[self.stl_colors[key]], opacity=stl_opacity, silhouette=True, smooth_shading=True, name=key)
                    else:
                        pl.add_mesh(surf, color=stl_colors[key], opacity=stl_opacity, silhouette=True, smooth_shading=True, name=key)

                elif type(add_stl) is list: #add selected list of stl solids
                    for i, key in enumerate(add_stl):
                        surf = self.read_stl(key)
                        surf = surf.clip_box(widget.bounds, invert=False)
                        if type(stl_colors) is dict:
                            if type(stl_colors[key]) is str:
                                pl.add_mesh(surf, color=material_colors[stl_colors[key]], opacity=stl_opacity, silhouette=True, smooth_shading=True, name=key)
                            else:
                                pl.add_mesh(surf, color=stl_colors[key], opacity=stl_opacity, silhouette=True, smooth_shading=True, name=key)
                        elif type(stl_colors) is list:
                            pl.add_mesh(surf, color=stl_colors[i], opacity=stl_opacity, silhouette=True, smooth_shading=True, name=key)
                        else:
                            pl.add_mesh(surf, color='white', opacity=stl_opacity, silhouette=True, smooth_shading=True, name=key)
            
            else: #add all stl solids
                for i, key in enumerate(self.stl_solids):
                    surf = self.read_stl(key)
                    surf = surf.clip_box(widget.bounds, invert=False)
                    if type(stl_colors) is dict:
                        if type(stl_colors[key]) is str:
                            pl.add_mesh(surf, color=material_colors[stl_colors[key]], opacity=stl_opacity, silhouette=True, smooth_shading=True, name=key)
                        else:
                            pl.add_mesh(surf, color=stl_colors[key], opacity=stl_opacity, silhouette=True, smooth_shading=True, name=key)
                    elif type(stl_colors) is list:
                        pl.add_mesh(surf, color=stl_colors[i], opacity=stl_opacity, silhouette=True, smooth_shading=True, name=key)
                    else:
                        pl.add_mesh(surf, color='white', opacity=stl_opacity, silhouette=True, smooth_shading=True, name=key)

        _ = pl.add_box_widget(callback=clip, rotation_enabled=False)

        # Camera orientation
        pl.camera_position = 'zx'
        pl.camera.azimuth += 30
        pl.camera.elevation += 30
        pl.set_background('mistyrose', top='white')
        try: pl.add_logo_widget('../docs/img/wakis-logo-pink.png')
        except: pass
        #pl.camera.zoom(zoom)
        pl.add_axes()
        pl.enable_3_lights()
        pl.enable_anti_aliasing(anti_aliasing)

        if offscreen:
            pl.export_html('grid_inspect.html')
        else:
            pl.show()