# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

import numpy as np
import pyvista as pv

from .field import Field

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
                Nx, Ny, Nz, stl_solids=None, stl_materials=None, 
                stl_rotate=[0., 0., 0.], stl_translate=[0., 0., 0.], stl_scale=1.0,
                stl_colors=None, verbose=1, tol=1e-3):
        
        self.verbose = verbose

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

        # Compatibility with FDTD grid obj
        self.nx, self.ny, self.nz = Nx, Ny, Nz
        
        # stl info
        self.stl_solids = stl_solids
        self.stl_materials = stl_materials
        self.stl_rotate = stl_rotate
        self.stl_translate = stl_translate
        self.stl_scale = stl_scale
        self.stl_colors = stl_colors

        # primal Grid G
        self.x = np.linspace(self.xmin, self.xmax, self.Nx+1)
        self.y = np.linspace(self.ymin, self.ymax, self.Ny+1)
        self.z = np.linspace(self.zmin, self.zmax, self.Nz+1)

        #tolerance for stl import tol*min(dx,dy,dz)
        self.tol = tol 

        # grid
        if verbose: print('Generating grid...')
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
        
        if stl_solids is not None:
            self.mark_cells_in_stl()
            if stl_colors is None:
                self.assign_colors()

    def mark_cells_in_stl(self):

        if self.verbose: print('Importing stl solids...')

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

        tol = np.min([self.dx, self.dy, self.dz])*self.tol
        for key in self.stl_solids.keys():

            surf = self.read_stl(key)

            # mark cells in stl [True == in stl, False == out stl]
            try:
                select = self.grid.select_enclosed_points(surf, tolerance=tol)
            except:
                select = self.grid.select_enclosed_points(surf, tolerance=tol, check_surface=False)
            self.grid[key] = select.point_data_to_cell_data()['SelectedPoints'] > tol

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

    def plot_solids(self, bounding_box=False, anti_aliasing=None,
                    opacity=1.0, specular=0.5,**kwargs):
        """
        Generates a 3D visualization of the imported STL geometries using PyVista.

        Parameters:
        -----------
        bounding_box : bool, optional
            If True, adds a bounding box around the plotted geometry (default: False).
        
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

            if self.stl_colors[key] == 'vacuum' or self.stl_material[key] == 'vacuum':
                opacity = 0.3
            else:
                opacity = opacity
            pl.add_mesh(self.read_stl(key), color=color, 
                        opacity=opacity, specular=specular, smooth_shading=True,
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
            
        pl.show()

    def inspect(self, add_stl=None, stl_opacity=0.5, stl_colors=None,
                anti_aliasing='ssaa'):
        
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
                    if stl_colors is None:
                        pl.add_mesh(surf, color=material_colors[self.stl_colors[key]], opacity=stl_opacity, silhouette=True, smooth_shading=True, name=key)
                    else:
                        pl.add_mesh(surf, color=stl_colors, opacity=stl_opacity, silhouette=True, smooth_shading=True, name=key)

                elif type(add_stl) is list: #add selected list of stl solids
                    for i, key in enumerate(add_stl):
                        surf = self.read_stl(key)
                        surf = surf.clip_box(widget.bounds, invert=False)
                        if type(stl_colors) is dict:
                            pl.add_mesh(surf, color=material_colors[self.stl_colors[key]], opacity=stl_opacity, silhouette=True, smooth_shading=True, name=key)
                        elif type(stl_colors) is list:
                            pl.add_mesh(surf, color=stl_colors[i], opacity=stl_opacity, silhouette=True, smooth_shading=True, name=key)
                        else:
                            pl.add_mesh(surf, color='white', opacity=stl_opacity, silhouette=True, smooth_shading=True, name=key)
            
            else: #add all stl solids
                for i, key in enumerate(self.stl_solids):
                    surf = self.read_stl(key)
                    surf = surf.clip_box(widget.bounds, invert=False)
                    if type(stl_colors) is dict:
                        pl.add_mesh(surf, color=material_colors[self.stl_colors[key]], opacity=stl_opacity, silhouette=True, smooth_shading=True, name=key)
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
        pl.show()