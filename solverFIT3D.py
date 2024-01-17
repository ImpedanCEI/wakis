import numpy as np
from scipy.constants import c as c_light, epsilon_0 as eps_0, mu_0 as mu_0
from scipy.sparse import csc_matrix as sparse_mat
from scipy.sparse import diags, block_diag, hstack, vstack
from scipy.sparse.linalg import inv

from field import Field
from materials import material_lib

class SolverFIT3D:

    def __init__(self, grid, cfln=0.5,
                 bc_low=['Periodic', 'Periodic', 'Periodic'],
                 bc_high=['Periodic', 'Periodic', 'Periodic'],
                 use_conductors=False, use_stl=False,
                 bg=[1.0, 1.0]):

        # Grid 
        self.grid = grid
        self.cfln = cfln
        self.dt = cfln / (c_light * np.sqrt(1 / self.grid.dx ** 2 + 1 / self.grid.dy ** 2 +
                                            1 / self.grid.dz ** 2))
        self.use_conductors = use_conductors
        self.use_stl = use_stl

        if use_stl:
            self.use_conductors = False

        self.Nx = self.grid.nx
        self.Ny = self.grid.ny
        self.Nz = self.grid.nz
        self.N = self.Nx*self.Ny*self.Nz

        self.dx = self.grid.dx
        self.dy = self.grid.dy
        self.dz = self.grid.dz

        self.x = self.grid.x[:-1]+self.dx/2
        self.y = self.grid.y[:-1]+self.dy/2
        self.z = self.grid.z[:-1]+self.dz/2

        self.L = self.grid.L
        self.iA = self.grid.iA
        self.tL = self.grid.tL
        self.itA = self.grid.itA

        # Fields
        self.E = Field(self.Nx, self.Ny, self.Nz)
        self.H = Field(self.Nx, self.Ny, self.Nz)
        self.J = Field(self.Nx, self.Ny, self.Nz)

        # Matrices
        N = self.N

        self.Px = diags([-1, 1], [0, 1], shape=(N, N), dtype=np.int8)
        self.Py = diags([-1, 1], [0, self.Nx], shape=(N, N), dtype=np.int8)
        self.Pz = diags([-1, 1], [0, self.Nx*self.Ny], shape=(N, N), dtype=np.int8)

        # original grid
        self.Ds = diags(self.L.toarray(), shape=(3*N, 3*N), dtype=float)
        self.iDa = diags(self.iA.toarray(), shape=(3*N, 3*N), dtype=float)

        # tilde grid
        self.tDs = diags(self.tL.toarray(), shape=(3*N, 3*N), dtype=float)
        self.itDa = diags(self.itA.toarray(), shape=(3*N, 3*N), dtype=float)

        # Curl matrix
        self.C = vstack([
                            hstack([sparse_mat((N,N)), -self.Pz, self.Py]),
                            hstack([self.Pz, sparse_mat((N,N)), -self.Px]),
                            hstack([-self.Py, self.Px, sparse_mat((N,N))])
                        ])
                
        # Boundaries
        self.bc_low = bc_low
        self.bc_high = bc_high
        self.activate_abc = False

        self.apply_bc_to_C() 

        # Materials 
        if type(bg) is str:
            bg = material_lib[bg.lower()]

        self.eps_bg, self.mu_bg = bg[0]*eps_0, bg[1]*mu_0
        self.ieps = Field(self.Nx, self.Ny, self.Nz, use_ones=True)*(1./self.eps_bg) 
        self.imu = Field(self.Nx, self.Ny, self.Nz, use_ones=True)*(1./self.mu_bg) 

        if self.use_stl:
            self.apply_stl()

        if self.use_conductors:
            self.apply_conductors()

        self.iDeps = diags(self.ieps.toarray(), shape=(3*N, 3*N), dtype=float)
        self.iDmu = diags(self.imu.toarray(), shape=(3*N, 3*N), dtype=float)

        # Pre-computing
        self.tDsiDmuiDaC = self.tDs * self.iDmu * self.iDa * self.C 
        self.itDaiDepsDstC = self.itDa * self.iDeps * self.Ds * self.C.transpose()

        # Flags
        self.step_0 = True
        self.plotter_active = False

        self.attrcleanup()

    def one_step(self):

        if self.step_0:
            self.set_ghosts_to_0()
            self.step_0 = False

            #if self.use_conductors:
                #self.set_field_in_conductors_to_0()

        self.H.fromarray(self.H.toarray() -
                         self.dt*self.tDsiDmuiDaC*self.E.toarray()
                         )

        self.E.fromarray(self.E.toarray() +
                         self.dt*(self.itDaiDepsDstC * self.H.toarray() - self.iDeps*self.J.toarray())
                         )
        
        #update ABC
        if self.activate_abc:
            self.update_abc()
        
    def apply_bc_to_C(self):
        '''
        Modifies rows or columns of C and tDs and itDa matrices
        according to bc_low and bc_high
        '''
        xlo, ylo, zlo = 1., 1., 1.
        xhi, yhi, zhi = 1., 1., 1.

        # Perodic: out == in
        if any(True for x in self.bc_low if x.lower() == 'periodic'):
            if self.bc_low[0].lower() == 'periodic' and self.bc_high[0].lower() == 'periodic':
                self.tL[-1, :, :, 'x'] = self.L[0, :, :, 'x']
                self.itA[-1, :, :, 'y'] = self.iA[0, :, :, 'y']
                self.itA[-1, :, :, 'z'] = self.iA[0, :, :, 'z']

            if self.bc_low[1].lower() == 'periodic' and self.bc_high[1].lower() == 'periodic':
                self.tL[:, -1, :, 'y'] = self.L[:, 0, :, 'y']
                self.itA[:, -1, :, 'x'] = self.iA[:, 0, :, 'x']
                self.itA[:, -1, :, 'z'] = self.iA[:, 0, :, 'z']

            if self.bc_low[2].lower() == 'periodic' and self.bc_high[2].lower() == 'periodic':
                self.tL[:, :, -1, 'z'] = self.L[:, :, 0, 'z']
                self.itA[:, :, -1, 'x'] = self.iA[:, :, 0, 'x']
                self.itA[:, :, -1, 'y'] = self.iA[:, :, 0, 'y']

            self.tDs = diags(self.tL.toarray(), shape=(3*self.N, 3*self.N), dtype=float)
            self.itDa = diags(self.itA.toarray(), shape=(3*self.N, 3*self.N), dtype=float)

        # Dirichlet PEC: tangential E field = 0 at boundary
        if any(True for x in self.bc_low if x.lower() == 'electric' or x.lower() == 'pec'):
    
            if self.bc_low[0].lower() == 'electric' or self.bc_low[0].lower() == 'pec':
                xlo = 0
            if self.bc_low[1].lower() == 'electric' or self.bc_low[1].lower() == 'pec':
                ylo = 0    
            if self.bc_low[2].lower() == 'electric' or self.bc_low[2].lower() == 'pec':
                zlo = 0   
            if self.bc_high[0].lower() == 'electric' or self.bc_high[0].lower() == 'pec':
                xhi = 0
            if self.bc_high[1].lower() == 'electric' or self.bc_high[1].lower() == 'pec':
                yhi = 0
            if self.bc_high[2].lower() == 'electric' or self.bc_high[2].lower() == 'pec':
                zhi = 0

            # Assemble matrix
            self.BC = Field(self.Nx, self.Ny, self.Nz, dtype=np.int8, use_ones=True)

            for d in ['x', 'y', 'z']: #tangential to zero
                if d != 'x':
                    self.BC[0, :, :, d] = xlo
                    self.BC[-1, :, :, d] = xhi
                if d != 'y':
                    self.BC[:, 0, :, d] = ylo
                    self.BC[:, -1, :, d] = yhi
                if d != 'z':
                    self.BC[:, :, 0, d] = zlo
                    self.BC[:, :, -1, d] = zhi
            
            self.Dbc = diags(self.BC.toarray(),
                            shape=(3*self.N, 3*self.N), 
                            dtype=np.int8
                            )

            # Update C (columns)
            self.C = self.C*self.Dbc


        # Dirichlet PMC: tangential H field = 0 at boundary
        if any(True for x in self.bc_low if x.lower() == 'magnetic' or x.lower() == 'pmc'):

            if self.bc_low[0].lower() == 'magnetic' or self.bc_low[0] == 'pmc':
                xlo = 0
            if self.bc_low[1].lower() == 'magnetic' or self.bc_low[1] == 'pmc':
                ylo = 0    
            if self.bc_low[2].lower() == 'magnetic' or self.bc_low[2] == 'pmc':
                zlo = 0   
            if self.bc_high[0].lower() == 'magnetic' or self.bc_high[0] == 'pmc':
                xhi = 0
            if self.bc_high[1].lower() == 'magnetic' or self.bc_high[1] == 'pmc':
                yhi = 0
            if self.bc_high[2].lower() == 'magnetic' or self.bc_high[2] == 'pmc':
                zhi = 0

            # Assemble matrix
            self.BC = Field(self.Nx, self.Ny, self.Nz, dtype=np.int8, use_ones=True)

            for d in ['x', 'y', 'z']: #tangential to zero
                if d != 'x':
                    self.BC[0, :, :, d] = xlo
                    self.BC[-1, :, :, d] = xhi
                if d != 'y':
                    self.BC[:, 0, :, d] = ylo
                    self.BC[:, -1, :, d] = yhi
                if d != 'z':
                    self.BC[:, :, 0, d] = zlo
                    self.BC[:, :, -1, d] = zhi

            self.Dbc = diags(self.BC.toarray(),
                            shape=(3*self.N, 3*self.N), 
                            dtype=np.int8
                            )

            # Update C (rows)
            self.C = self.Dbc*self.C

        # Absorbing boundary conditions ABC
        if any(True for x in self.bc_low if x.lower() == 'abc'):
            self.activate_abc = True

    def update_abc(self):
        '''
        Apply ABC algo to the selected BC, 
        to be applied after each timestep
        '''

        if self.bc_low[0].lower() == 'abc':
            for d in ['x', 'y', 'z']:
                self.E[0, :, :, d] = self.E[1, :, :, d]
                self.H[0, :, :, d] = self.H[1, :, :, d]  

        if self.bc_low[1].lower() == 'abc':
            for d in ['x', 'y', 'z']:
                self.E[:, 0, :, d] = self.E[:, 1, :, d]
                self.H[:, 0, :, d] = self.H[:, 1, :, d]
                   
        if self.bc_low[2].lower() == 'abc':
            for d in ['x', 'y', 'z']:
                self.E[:, :, 0, d] = self.E[:, :, 1, d]
                self.H[:, :, 0, d] = self.H[:, :, 1, d]  

        if self.bc_high[0].lower() == 'abc':
            for d in ['x', 'y', 'z']:
                self.E[-1, :, :, d] = self.E[-2, :, :, d]
                self.H[-1, :, :, d] = self.H[-2, :, :, d] 

        if self.bc_high[1].lower() == 'abc':
            for d in ['x', 'y', 'z']:
                self.E[:, -1, :, d] = self.E[:, -2, :, d]
                self.H[:, -1, :, d] = self.H[:, -2, :, d] 

        if self.bc_high[2].lower() == 'abc':
            for d in ['x', 'y', 'z']:
                self.E[:, :, -1, d] = self.E[:, :, -2, d]
                self.H[:, :, -1, d] = self.H[:, :, -2, d] 

    def set_ghosts_to_0(self):
        '''
        Cleanup for initial conditions if they are 
        accidentally applied to the ghost cells
        '''    
        # Set H ghost quantities to 0
        for d in ['x', 'y', 'z']: #tangential to zero
            if d != 'x':
                self.H[-1, :, :, d] = 0.
            if d != 'y':
                self.H[:, -1, :, d] = 0.
            if d != 'z':
                self.H[:, :, -1, d] = 0.

        # Set E ghost quantities to 0
        self.E[-1, :, :, 'x'] = 0.
        self.E[:, -1, :, 'y'] = 0.
        self.E[:, :, -1, 'z'] = 0.

    def apply_conductors(self):
        '''
        Set the 1/epsilon values inside the PEC conductors to zero
        '''
        self.flag_in_conductors = self.grid.flag_int_cell_yz[:-1,:,:]  \
                        + self.grid.flag_int_cell_zx[:,:-1,:] \
                        + self.grid.flag_int_cell_xy[:,:,:-1]

        self.ieps *= self.flag_in_conductors

    def set_field_in_conductors_to_0(self):
        '''
        Cleanup for initial conditions if they are 
        accidentally applied to the conductors
        '''    
        self.flag_cleanup = self.grid.flag_int_cell_yz[:-1,:,:]  \
                        + self.grid.flag_int_cell_zx[:,:-1,:]    \
                        + self.grid.flag_int_cell_xy[:,:,:-1]

        self.H *= self.flag_cleanup
        self.E *= self.flag_cleanup
        
    def apply_stl(self):
        '''
        Mask the cells inside the stl and assing the material
        defined by the user

        * Note: stl material should contain **relative** epsilon and mu
        ** Note 2: when assigning the stl material, the default values
                   1./eps_0 and 1./mu_0 are substracted
        '''
        grid = self.grid.grid
        self.stl_solids = self.grid.stl_solids
        self.stl_materials = self.grid.stl_materials

        for key in self.stl_solids.keys():

            mask = np.reshape(grid[key], (self.Nx, self.Ny, self.Nz)).astype(int)
            
            if type(self.stl_materials[key]) is str:
                # Retrieve from material library
                mat_key = self.stl_materials[key].lower()

                eps = material_lib[mat_key][0]*eps_0
                mu = material_lib[mat_key][1]*mu_0

                # Setting to zero
                self.ieps += self.ieps * (-1.0*mask) 
                self.imu += self.imu * (-1.0*mask)

                # Adding new values
                self.ieps += mask * 1./eps 
                self.imu += mask * 1./mu

            else:
                # From input
                eps = self.stl_materials[key][0]*eps_0
                mu = self.stl_materials[key][1]*mu_0

                # Setting to zero
                self.ieps += self.ieps * (-1.0*mask) 
                self.imu += self.imu * (-1.0*mask)

                # Adding new values
                self.ieps += mask * 1./eps
                self.imu += mask * 1./mu

    def attrcleanup(self):

        # Fields
        del self.L, self.tL, self.iA, self.itA
        del self.BC

        # Matrices
        del self.Px, self.Py, self.Pz
        del self.Ds, self.iDa, self.tDs, self.itDa
        del self.C
        del self.Dbc

    def plot3D(self, field='E', component='z', clim=None, hide_solids=None,
               show_solids=None, add_stl=None, stl_opacity=0.1, stl_colors='white',
               title=None, cmap='jet', clip_volume=True, clip_normal='-y',
               clip_box=False, clip_bounds=None, off_screen=False, zoom=0.5, 
               nan_opacity=1.0, n=None):
        '''
        Built-in 3D plotting using PyVista
        
        Parameters:
        ----------
        field: str, default 'E'
            3D field magnitude ('E', 'H', or 'J') to plot
            To plot a component 'Ex', 'Hy' is also accepted
        component: str, default 'z'
            3D field compoonent ('x', 'y', 'z', 'Abs') to plot. It will be overriden
            if a component is defined in field
        clim: list, optional
            Colorbar limits for the field plot [min, max]
        hide_solids: bool, optional
            Mask the values inside solid to np.nan. NaNs will be shown in gray, 
            since there is a bug with the nan_opacity parameter
        show_solids: bool, optional
            Mask the values outside solid to np.nan. 
        add_stl: str or list, optional
            List or str of stl solids to add to the plot by `pv.add_mesh`
        stl_opacity: float, default 0.1
            Opacity of the stl surfaces (0 - Transparent, 1 - Opaque)
        stl_colors: str or list of str, default 'white'
            Color of the stl surfaces
        title: str, optional
            Title used to save the screenshot of the 3D plot (Path+Name) if off_screen=True
        cmap: str, default 'jet'
            Colormap name to use in the field display
        clip_volume: bool, default True
            Enable an interactive widget to clip out part of the domain, plane normal is defined by 
            `clip_normal` parameter
        clip_normal: str, default '-y'
            Normal direction of the clip_volume interactive plane
        clip box: bool, default False
            Enable a box clipping of the domain. The box bounds are defined by `clip_bounds` parameter
        off_screen: bool, default False
            Enable plot rendering off screen, for gif frames generation. 
            Plot will not be rendered if set to True.
        n: int, optional
            Timestep number to be added to the plot title and figsave title.
        '''
        import pyvista as pv

        if len(field) == 2: #support for e.g. field='Ex'
            component = field[1]
            field = field[0]

        if title is None:
            title = field + component +'3d'

        if not self.plotter_active:

            pl = pv.Plotter(off_screen=off_screen)

            # Plot stl surface(s)
            if add_stl is not None:
                if type(add_stl) is str:
                    key = add_stl
                    surf = self.grid.read_stl(key)
                    pl.add_mesh(surf, color=stl_colors, opacity=stl_opacity, smooth_shading=True)

                elif type(add_stl) is list:
                    for i, key in enumerate(add_stl):
                        surf = self.grid.read_stl(key)
                        if type(stl_colors) is list:
                            pl.add_mesh(surf, color=stl_colors[i], opacity=stl_opacity, smooth_shading=True)
                        else:
                            pl.add_mesh(surf, color=stl_colors, opacity=stl_opacity, smooth_shading=True)
                else:
                    print("`add_stl` format not supported")

            pl.camera_position = 'zx'
            pl.camera.azimuth += 30
            pl.camera.elevation += 30
            #pl.background_color = "grey"
            pl.camera.zoom(zoom)
            pl.add_axes()
            pl.enable_3_lights()

            if off_screen:
                self.plotter_active = True
        else: 
            pl = self.pl


        # Plot field
        if field == 'E':
            if component == 'Abs':
                self.grid.grid.cell_data[field+component] = np.reshape(self.E.get_abs()[:, :, :], self.N)
            else:
                self.grid.grid.cell_data[field+component] = np.reshape(self.E[:, :, :, component], self.N)

        elif field == 'H':
            if component == 'Abs':
                self.grid.grid.cell_data[field+component] = np.reshape(self.H.get_abs()[:, :, :], self.N)
            else:
                self.grid.grid.cell_data[field+component] = np.reshape(self.H[:, :, :, component], self.N)

        elif field == 'J':
            if component == 'Abs':
                self.grid.grid.cell_data[field+component] = np.reshape(self.J.get_abs()[:, :, :], self.N)
            else:
                self.grid.grid.cell_data[field+component] = np.reshape(self.J[:, :, :, component], self.N)
        else:
            print("`field` value not valid")

        points = self.grid.grid.cell_data_to_point_data() #interpolate
        
        if hide_solids is not None:
            tol = np.min([self.dx, self.dy, self.dz])*1e-3
            if type(hide_solids) is str:
                surf = self.grid.read_stl(hide_solids)
                select = self.grid.grid.select_enclosed_points(surf, tolerance=tol)
                mask = select['SelectedPoints'] > 0

            elif type(hide_solids) is list:
                for i, solid in enumerate(hide_solids):
                    surf = self.grid.read_stl(solid)
                    select = self.grid.grid.select_enclosed_points(surf, tolerance=tol)
                    if i == 0:
                        mask = select['SelectedPoints'] > 0
                    else:
                        mask += select['SelectedPoints'] > 0

            points[field+component][mask] = np.nan 

        if show_solids is not None:
            tol = np.min([self.dx, self.dy, self.dz])*1e-3
            if type(show_solids) is str:
                surf = self.grid.read_stl(show_solids)
                select = self.grid.grid.select_enclosed_points(surf, tolerance=tol)
                mask = select['SelectedPoints'] > 0

            elif type(show_solids) is list:
                for solid in show_solids:
                    surf = self.grid.read_stl(solid)
                    select = self.grid.grid.select_enclosed_points(surf, tolerance=tol)
                    if i == 0:
                        mask = select['SelectedPoints'] > 0
                    else:
                        mask += select['SelectedPoints'] > 0

            points[field+component][np.logical_not(mask)] = np.nan 

        if clip_box:
            if clip_bounds is None:
                Lx, Ly = (self.grid.xmax-self.grid.xmin), (self.grid.ymax-self.grid.ymin)
                clip_bounds = [self.grid.xmax-Lx/2, self.grid.xmax,
                               self.grid.ymax-Ly/2, self.grid.ymax,
                               self.grid.zmin, self.grid.zmax]
                
            ac1 = pl.add_mesh(points.clip_box(bounds=clip_bounds), opacity=nan_opacity,
                              scalars=field+component, cmap=cmap, 
                              clim=clim)

        elif clip_volume:
            ac1 = pl.add_mesh_clip_plane(points, normal=clip_normal, opacity=1.0,
                                         scalars=field+component, cmap=cmap, clim=clim, 
                                         normal_rotation=False, nan_opacity=nan_opacity)
        if n is not None:
            pl.add_title(field+component+f' field, timestep={n}', font='times', font_size=12)
            title += '_'+str(n).zfill(6)

        # Save
        if off_screen:
            pl.screenshot(title+'.png')
            pl.remove_actor(ac1)
            self.pl = pl
        else:
            pl.show(full_screen=True)

    def plot2D(self, field='E', component='z', plane='ZY', pos=0.5, norm='linear', 
               vmin=None, vmax=None, figsize=[8,4], cmap='jet', patch_alpha=0.1, 
               patch_reverse=False, add_patch=False, title=None, off_screen=False, n=None,
               interpolation='antialiased'):
        '''
        Built-in 2D plotting of a field slice using matplotlib
        
        Parameters:
        ----------
        field: str, default 'E'
            Field magnitude ('E', 'H', or 'J') to plot
            To plot a component 'Ex', 'Hy' is also accepted
        component: str, default 'z'
            Field compoonent ('x', 'y', 'z', 'Abs') to plot. It will be overriden
            if a component is defined in field
        plane: str, default 'XZ'
            Plane where to plot the 2d field cut: 'XY', 'ZY' or 'ZX'
        pos: float, default 0.5
            Position of the cutting plane, as a franction of the plane's normal dimension
            e.g. plane 'XZ' wil be sitting at y=pos*(ymax-ymin)
        norm: str, default 'linear'
            Plotting scale to pass to matplotlib imshow: 'linear', 'log', 'symlog'
        vmin: list, optional
            Colorbar min limit for the field plot
        vmax: list, optional
            Colorbar max limit for the field plot       
        figsize: list, default [8,4]    
            Figure size to pass to the plot initialization
        add_patch: str or list, optional
            List or str of stl solids to add to the plot by `pv.add_mesh`
        patch_alpha: float, default 0.1
            Value for the transparency of the patch if `add_patch = True`
        title: str, optional
            Title used to save the screenshot of the 3D plot (Path+Name) if off_screen=True.
            If n is provided, 'str(n).zfill(6)' will be added to the title.
        cmap: str, default 'jet'
            Colormap name to use in the field display
        off_screen: bool, default False
            Enable plot rendering off screen, for gif frames generation. 
            Plot will not be rendered if set to True.
        n: int, optional
            Timestep number to be added to the plot title and figsave title.
        interpolation: str, default 'antialiased'
            Interpolation method to pass to matplotlib imshow e.g., 'none',
            'antialiased', 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36',
        '''
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        xmin, xmax = self.grid.xmin, self.grid.xmax 
        ymin, ymax = self.grid.ymin, self.grid.ymax
        zmin, zmax = self.grid.zmin, self.grid.zmax
        
        if len(field) == 2: #support for e.g. field='Ex'
            component = field[1]
            field = field[0]
        
        if title is None:
            title = field + component +'2d'
            
        if plane == 'XY':
            x, y, z = slice(0,Nx), slice(0,Ny), int(Nz*pos) #plane XY
            cut = f'(x,y,a) a={round(pos*(zmax-zmin)+zmin,3)}'
            xax, yax = 'y', 'x'
            extent = [ymin, ymax, xmin, xmax]

        elif plane == 'ZY':
            x, y, z = int(Nx*pos), slice(0,Ny), slice(0,Nz) #plane ZY
            cut = f'(a,y,z) a={round(pos*(xmax-xmin)+xmin,3)}'
            xax, yax = 'z', 'y'
            extent = [zmin, zmax, ymin, ymax]
        
        elif plane == 'ZX':
            x, y, z = slice(0,Nx),  int(Ny*pos), slice(0,Nz) #plane XZ
            cut = f'(x,a,z) a={round(pos*(ymax-ymin)+ymin,3)}'
            xax, yax = 'z', 'x'
            extent = [zmin, zmax, xmin, xmax]
        
        else:
            print("Plane needs to be one of the following: 'XY', 'ZY', 'ZX'")

        fig, ax = plt.subplots(1,1, figsize=figsize)

        if field == 'E':
            if component == 'Abs':
                im = ax.imshow(self.E.get_abs()[x, y, z], cmap=cmap,  norm=norm, 
                               extent=extent, origin='lower', vmin=vmin, vmax=vmax,
                               interpolation=interpolation)
            else:
                im = ax.imshow(self.E[x, y, z, component], cmap=cmap,  norm=norm, 
                               extent=extent, origin='lower', vmin=vmin, vmax=vmax,
                               interpolation=interpolation)
        if field == 'H':
            if component == 'Abs':
                im = ax.imshow(self.H.get_abs()[x, y, z], cmap=cmap,  norm=norm, 
                               extent=extent, origin='lower', vmin=vmin, vmax=vmax,
                               interpolation=interpolation)
            else:
                im = ax.imshow(self.H[x, y, z, component], cmap=cmap,  norm=norm, 
                               extent=extent, origin='lower', vmin=vmin, vmax=vmax,
                               interpolation=interpolation)
        if field == 'J':
            if component == 'Abs':
                im = ax.imshow(self.J.get_abs()[x, y, z], cmap=cmap,  norm=norm, 
                               extent=extent, origin='lower', vmin=vmin, vmax=vmax,
                               interpolation=interpolation)
            else:
                im = ax.imshow(self.J[x, y, z, component], cmap=cmap,  norm=norm, 
                               extent=extent, origin='lower', vmin=vmin, vmax=vmax,
                               interpolation=interpolation)  
                              
        fig.colorbar(im, cax=make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05))
        ax.set_title(f'FIT {field}{component}{cut}')
        ax.set_xlabel(xax)
        ax.set_ylabel(yax)

        # Patch stl
        if add_patch is not None:
            if type(add_patch) is str:
                mask = np.reshape(self.grid.grid[add_patch], (Nx, Ny, Nz))
                patch = np.ones((Nx, Ny, Nz))
                if patch_reverse:
                    patch[mask] = np.nan 
                else:
                    patch[np.logical_not(mask)] = np.nan 
                ax.imshow(patch[x,y,z], cmap='Greys', extent=extent, origin='lower', alpha=patch_alpha)

            elif type(add_patch) is list:
                for solid in add_patch:
                    mask = np.reshape(self.grid.grid[solid], (Nx, Ny, Nz))
                    patch = np.ones((Nx, Ny, Nz))
                    patch[np.logical_not(mask)] = np.nan
                    ax.imshow(patch[x,y,z], cmap='Greys', extent=extent, origin='lower', alpha=patch_alpha)

        if n is not None:
            fig.suptitle('$'+str(field)+'_{'+str(component)+'}$ field, timestep='+str(n))
            title += '_'+str(n).zfill(6)

        fig.tight_layout()

        if off_screen:
            fig.savefig(title+'.png')
            plt.clf()
            plt.close(fig)

        else:
            plt.show()