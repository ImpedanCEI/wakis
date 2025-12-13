# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt

class PlotMixin:

    def plot3D(self, field='E', component='z', clim=None, hide_solids=None,
               show_solids=None, add_stl=None, stl_opacity=0.1, stl_colors='white',
               title=None, cmap='jet',
               clip_interactive=False, clip_normal='-y',
               clip_box=False, clip_bounds=None,
               off_screen=False, zoom=0.5, camera_position=None,
               nan_opacity=1.0, n=None):
        '''
        Built-in 3D plotting using PyVista

        Parameters:
        -----------
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
        clip_interactive: bool, default False
            Enable an interactive widget to clip out part of the domain, plane normal is defined by
            `clip_normal` parameter
        clip_normal: str, default '-y'
            Normal direction of the clip_volume interactive plane
        clip_box: bool, default False
            Enable a static box clipping of the domain. The box bounds are defined by `clip_bounds` parameter
        clip_bounds: Default None
            List of bounds [xmin, xmax, ymin, ymax, zmin, zmax] of the box to clip if clip_box is active.
        field_on_stl : bool, default False
            Samples the field on the stl file specified in `add_stl`.
        field_opacity : optional, default 1.0
            Sets de opacity of the `field_on_stl` plot
        off_screen: bool, default False
            Enable plot rendering off screen, for gif frames generation.
            Plot will not be rendered if set to True.
        n: int, optional
            Timestep number to be added to the plot title and figsave title.
        '''
        if self.use_mpi:
            print('[!] plot3D is not supported when `use_mpi=True`')
            return

        if len(field) == 2: #support for e.g. field='Ex'
            component = field[1]
            field = field[0]

        if title is None:
            title = field + component +'3d'

        if self.plotter_active and not off_screen:
            self.plotter_active = False

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
                    key = self.grid.stl_solids.keys()[0]
                    surf = self.grid.read_stl(key)
                    pl.add_mesh(surf, color=stl_colors, opacity=stl_opacity, smooth_shading=True)

            if camera_position is None:
                pl.camera_position = 'zx'
                pl.camera.azimuth += 30
                pl.camera.elevation += 30
            else:
                pl.camera_position = camera_position

            pl.set_background('mistyrose', top='white')
            self._add_logo_widget(pl)
            pl.camera.zoom(zoom)
            pl.add_axes()
            pl.enable_3_lights()

            if off_screen:
                self.plotter_active = True
        else:
            pl = self.pl


        # Plot field
        if field == 'E':
            self.grid.grid.cell_data[field+component] = np.reshape(self.E[:, :, :, component], self.N)
        elif field == 'H':
            self.grid.grid.cell_data[field+component] = np.reshape(self.H[:, :, :, component], self.N)
        elif field == 'J':
            self.grid.grid.cell_data[field+component] = np.reshape(self.J[:, :, :, component], self.N)
        else:
            print("`field` value not valid")

        points = self.grid.grid.cell_data_to_point_data() #interpolate

        # Mask the values inside solid to np.nan
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

        # Mask the values outside solid to np.nan
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

        # Clip a rectangle of the domain
        if clip_box:
            if clip_bounds is None:
                Lx, Ly = (self.grid.xmax-self.grid.xmin), (self.grid.ymax-self.grid.ymin)
                clip_bounds = [self.grid.xmax-Lx/2, self.grid.xmax,
                               self.grid.ymax-Ly/2, self.grid.ymax,
                               self.grid.zmin, self.grid.zmax]

            ac1 = pl.add_mesh(points.clip_box(bounds=clip_bounds), opacity=nan_opacity,
                              scalars=field+component, cmap=cmap, clim=clim)

        # Enable an interactive widget to clip out part of the domain with a plane, with clip_normal
        elif clip_interactive:
            ac1 = pl.add_mesh_clip_plane(points, normal=clip_normal, opacity=1.0,
                                         scalars=field+component, cmap=cmap, clim=clim,
                                         normal_rotation=False, nan_opacity=nan_opacity)
        else:
            print('Plotting option inconsistent')

        # Save
        if n is not None:
            pl.add_title(field+component+f' field, timestep={n}', font='times', font_size=12)
            title += '_'+str(n).zfill(6)
        if off_screen:
            pl.screenshot(title+'.png')
            pl.remove_actor(ac1)
            self.pl = pl
        else:
            pl.show(full_screen=False)

    def plot3DonSTL(self, field='E', component='z', clim=None, cmap='jet', log_scale=False,
                    stl_with_field=None, field_opacity=1.0, tolerance=None,
                    stl_transparent=None, stl_opacity=0.1, stl_colors='white',
                    clip_plane = False, clip_interactive=False,
                    clip_normal='-x', clip_origin=[0,0,0],
                    clip_box=False, clip_bounds=None,
                    title=None, off_screen=False, n=None,
                    zoom=0.5, camera_position=None, **kwargs):
        '''
        Built-in 3D plotting using PyVista

        Parameters:
        -----------
        field: str, default 'E'
            3D field magnitude ('E', 'H', or 'J') to plot
            To plot a component 'Ex', 'Hy' is also accepted
        component: str, default 'z'
            3D field compoonent ('x', 'y', 'z', 'Abs') to plot. It will be overriden
            if a component is defined in field
        clim: list, optional
            Colorbar limits for the field plot [min, max]
        cmap: str or cmap obj, default 'jet'
            Colormap to use for the field plot
        log_scale: bool, default False
            Turns on logarithmic scale colorbar
        stl_with_field : list or str
            STL str name or list of names to samples the selected field on
        field_opacity : optional, default 1.0
            Sets de opacity of the `field_on_stl` plot
        tolerance : float, default None
            Tolerance to apply to PyVista's sampling algorithm
        stl_transparent: list or str, default None
            STL name or list of names to add to the scene with the selected transparency and color
        stl_opacity: float, default 0.1
            Opacity of the STL solids without field
        stl_colors: list or str, default 'white'
            str or list of colors to use for each STL solid
        clip_interactive: bool, default False
            Enable an interactive widget to clip out part of the domain, plane normal is defined by
            `clip_normal` parameter
        clip_plane: bool, default False
            Clip stl_with_field surface with a plane and show field on such plane
        clip_normal: str, default '-y'
            Normal direction of the clip_volume interactive plane and the clip_plane
        clip_origin: list, default [0,0,0]
            Origin of the clipping plane for the clip_plane option
        clip_box: bool, default False
            Enable a static box clipping of the domain. The box bounds are defined by `clip_bounds` parameter
        clip_bounds: Default None
            List of bounds [xmin, xmax, ymin, ymax, zmin, zmax] of the box to clip if clip_box is active.
        off_screen: bool, default False
            Enable plot rendering off screen, for gif frames generation.
            Plot will not be rendered if set to True.
        title: str
            Name to use in the .png savefile with off_screen is True
        n: int, optional
            Timestep number to be added to the plot title and figsave title.
        **kwargs: optional
            PyVista's add_mesh optional arguments:
            https://docs.pyvista.org/api/plotting/_autosummary/pyvista.plotter.add_mesh
        '''
        if self.use_mpi:
            print('[!] plot3D is not supported when `use_mpi=True`')
            return

        if len(field) == 2: #support for e.g. field='Ex'
            component = field[1]
            field = field[0]

        if title is None:
            title = field + component +'3d'

        if self.plotter_active and not off_screen:
            self.plotter_active = False

        if not self.plotter_active:
            pl = pv.Plotter(off_screen=off_screen, lighting='none')
            light = pv.Light(light_type='headlight')
            pl.add_light(light)

            # Plot stl surface(s)
            if stl_transparent is not None:
                if type(stl_transparent) is str:
                    key = stl_transparent
                    surf = self.grid.read_stl(key)
                    pl.add_mesh(surf, color=stl_colors, opacity=stl_opacity, smooth_shading=True)

                elif type(stl_transparent) is list:
                    for i, key in enumerate(stl_transparent):
                        surf = self.grid.read_stl(key)
                        if type(stl_colors) is list:
                            pl.add_mesh(surf, color=stl_colors[i], opacity=stl_opacity, smooth_shading=True)
                        else:
                            pl.add_mesh(surf, color=stl_colors, opacity=stl_opacity, smooth_shading=True)

            if off_screen:
                self.plotter_active = True
        else:
            pl = self.pl

        # Plot field
        if field == 'E':
            self.grid.grid.cell_data[field+component] = np.reshape(self.E[:, :, :, component], self.N)

        elif field == 'H':
            self.grid.grid.cell_data[field+component] = np.reshape(self.H[:, :, :, component], self.N)

        elif field == 'J':
            self.grid.grid.cell_data[field+component] = np.reshape(self.J[:, :, :, component], self.N)
        else:
            print("`field` value not valid")

        points = self.grid.grid.cell_data_to_point_data() #interpolate

        # Interpolate fields on stl
        if stl_with_field is not None:
            if type(stl_with_field) is str:
                key = stl_with_field
                surf = self.grid.read_stl(key)
                if clip_plane:
                    try:
                        surf = surf.clip_closed_surface(normal=clip_normal, origin=clip_origin).subdivide_adaptive(max_edge_len=3*self.dz)
                    except Exception:
                        print("[!] Surface non-manifold, clip with plane skipped")

                fieldonsurf = surf.sample(points, tolerance=tolerance)

                if clip_interactive: # interactive plotting with a plane
                    ac1 = pl.add_mesh_clip_plane(fieldonsurf, normal=clip_normal, normal_rotation=False,
                                        scalars=field+component, opacity=field_opacity,
                                        cmap=cmap, clim=clim, log_scale=log_scale,
                                        **kwargs)

                elif clip_box: # Clip a rectangle of the domain
                    if clip_bounds is None:
                        Lx, Ly = (self.grid.xmax-self.grid.xmin), (self.grid.ymax-self.grid.ymin)
                        clip_bounds = [self.grid.xmax-Lx/2, self.grid.xmax,
                                    self.grid.ymax-Ly/2, self.grid.ymax,
                                    self.grid.zmin, self.grid.zmax]

                    ac1 = pl.add_mesh(fieldonsurf.clip_box(bounds=clip_bounds), cmap=cmap, clim=clim,
                                  scalars=field+component, opacity=field_opacity,
                                  log_scale=log_scale,
                                  **kwargs)

                else:
                    ac1 = pl.add_mesh(fieldonsurf, cmap=cmap, clim=clim,
                                  scalars=field+component, opacity=field_opacity,
                                  log_scale=log_scale,
                                  **kwargs)

            elif type(stl_with_field) is list:
                for i, key in enumerate(stl_with_field):
                    surf = self.grid.read_stl(key)
                    if clip_plane:
                        try:
                            surf = surf.clip_closed_surface(normal=clip_normal, origin=clip_origin)
                        except Exception:
                            print("Surface non-manifold, clip with plane skipped")

                    fieldonsurf = surf.sample(points, tolerance=tolerance)

                    if clip_interactive: # interactive plotting with a plane
                        ac1 = pl.add_mesh_clip_plane(fieldonsurf, normal=clip_normal, normal_rotation=False,
                                        scalars=field+component, opacity=field_opacity,
                                        cmap=cmap, clim=clim, log_scale=log_scale,
                                        **kwargs)
                    elif clip_box: # Clip a rectangle of the domain
                        if clip_bounds is None:
                            Lx, Ly = (self.grid.xmax-self.grid.xmin), (self.grid.ymax-self.grid.ymin)
                            clip_bounds = [self.grid.xmax-Lx/2, self.grid.xmax,
                                        self.grid.ymax-Ly/2, self.grid.ymax,
                                        self.grid.zmin, self.grid.zmax]
                        ac1 = pl.add_mesh(fieldonsurf.clip_box(bounds=clip_bounds), cmap=cmap, clim=clim,
                                  scalars=field+component, opacity=field_opacity,
                                  log_scale=log_scale, **kwargs)
                    else:
                        ac1 = pl.add_mesh(fieldonsurf, cmap=cmap, clim=clim,
                                      scalars=field+component, opacity=field_opacity,
                                      log_scale=log_scale,
                                      **kwargs)
        if camera_position is None:
            pl.camera_position = 'zx'
            pl.camera.azimuth += 30
            pl.camera.elevation += 30
        else:
            pl.camera_position = camera_position

        pl.set_background('mistyrose', top='white')
        self._add_logo_widget(pl)
        pl.camera.zoom(zoom)
        pl.add_axes()
        pl.enable_anti_aliasing()
        #pl.enable_3_lights()

        if n is not None:
            pl.add_title(field+component+f' field, timestep={n}', font='times', font_size=12)
            title += '_'+str(n).zfill(6)

        # Save
        if off_screen:
            pl.screenshot(title+'.png')
            pl.remove_actor(ac1)
            self.pl = pl
        else:
            pl.show(full_screen=False)

    def plot2D(self, field='E', component='z', plane='ZY', pos=0.5, norm=None,
               vmin=None, vmax=None, figsize=[8,4], cmap='jet', patch_alpha=0.1,
               patch_reverse=False, add_patch=False, title=None, off_screen=False,
               n=None, interpolation='antialiased', dpi=100, return_handles=False):
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
        plane: arr or str, default 'XZ'
            Plane where to plot the 2d field cut: array of 2 slices() and 1 int [x,y,z]
            or a str 'XY', 'ZY' or 'ZX'
        pos: float, default 0.5
            Position of the cutting plane, as a franction of the plane's normal dimension
            e.g. plane 'XZ' wil be sitting at y=pos*(ymax-ymin)
        norm: str, default None
            Plotting scale to pass to matplotlib imshow: 'linear', 'log', 'symlog'
            ** Only for matplotlib version >= 3.8
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
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        xmin, xmax = self.grid.xmin, self.grid.xmax
        ymin, ymax = self.grid.ymin, self.grid.ymax
        zmin, zmax = self.grid.zmin, self.grid.zmax
        _z = self.z

        if self.use_mpi:
            zmin, zmax = self.grid.ZMIN, self.grid.ZMAX
            Nz = self.grid.NZ
            _z = self.grid.Z

        if type(field) is str:
            if len(field) == 2: #support for e.g. field='Ex'
                component = field[1]
                field = field[0]
            elif len(field) == 4: #support for e.g. field='EAbs'
                component = field[1:]
                field = field[0]

        if title is None:
            title = field + component +'2d'

        if type(plane) is not str and len(plane) == 3:
            x, y, z = plane[0], plane[1], plane[2]

            if type(plane[2]) is int:
                cut = f'(x,y,a) a={round(self.z[z],3)}'
                xax, yax = 'y', 'x'
                extent = [self.y[y].min(), self.y[y].max(),
                          self.x[x].min(), self.x[x].max()]

            if type(plane[0]) is int:
                cut = f'(a,y,z) a={round(self.x[x],3)}'
                xax, yax = 'z', 'y'
                extent = [_z[z].min(), _z[z].max(),
                          self.y[y].min(), self.y[y].max()]

            if type(plane[1]) is int:
                cut = f'(x,a,z) a={round(self.y[y],3)}'
                xax, yax = 'z', 'x'
                extent = [_z[z].min(), _z[z].max(),
                          self.x[x].min(), self.x[x].max()]

        elif plane == 'XY':
            x, y, z = slice(0,Nx), slice(0,Ny), int(Nz*pos) #plane XY
            cut = f'(x,y,a) a={round(pos*(zmax-zmin)+zmin,3)}'
            xax, yax = 'y', 'x'
            extent = [ymin, ymax, xmin, xmax]

        elif plane == 'ZY' or plane == 'YZ':
            x, y, z = int(Nx*pos), slice(0,Ny), slice(0,Nz) #plane ZY
            cut = f'(a,y,z) a={round(pos*(xmax-xmin)+xmin,3)}'
            xax, yax = 'z', 'y'
            extent = [zmin, zmax, ymin, ymax]

        elif plane == 'ZX' or plane == 'XZ':
            x, y, z = slice(0,Nx),  int(Ny*pos), slice(0,Nz) #plane XZ
            cut = f'(x,a,z) a={round(pos*(ymax-ymin)+ymin,3)}'
            xax, yax = 'z', 'x'
            extent = [zmin, zmax, xmin, xmax]

        else:
            print("Plane needs to be an array of slices [x,y,z] or a str 'XY', 'ZY', 'ZX'")

        if self.use_mpi: # only in rank=0
            _field = self.mpi_gather(field, x=x, y=y, z=z, component=component)

            if self.rank == 0:
                fig, ax = plt.subplots(1,1, figsize=figsize, dpi=dpi)
                im = ax.imshow(_field, cmap=cmap,  norm=norm,
                    extent=extent, origin='lower', vmin=vmin, vmax=vmax,
                    interpolation=interpolation)

                fig.colorbar(im, cax=make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05))
                ax.set_title(f'Wakis {field}{component}{cut}')
                ax.set_xlabel(xax)
                ax.set_ylabel(yax)

                if n is not None:
                    fig.suptitle('$'+str(field)+'_{'+str(component)+'}$ field, timestep='+str(n))
                    title += '_'+str(n).zfill(6)

                fig.tight_layout()

                if off_screen:
                    fig.savefig(title+'.png')
                    plt.clf()
                    plt.close(fig)
                elif return_handles:
                    return fig, ax
                else:
                    plt.show(block=False)
        else:
            fig, ax = plt.subplots(1,1, figsize=figsize, dpi=dpi)
            if field == 'E':
                _field = self.E[x, y, z, component]
            if field == 'H':
                _field = self.H[x, y, z, component]
            if field == 'J':
                _field = self.J[x, y, z, component]

            im = ax.imshow(_field, cmap=cmap,  norm=norm,
                            extent=extent, origin='lower', vmin=vmin, vmax=vmax,
                            interpolation=interpolation)

            fig.colorbar(im, cax=make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05))
            ax.set_title(f'Wakis {field}{component}{cut}')
            ax.set_xlabel(xax)
            ax.set_ylabel(yax)

            # Patch stl - not supported when running MPI
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
                        if patch_reverse:
                            patch[mask] = np.nan
                        else:
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
            elif return_handles:
                return fig, ax
            else:
                plt.show(block=False)

    def plot1D(self, field='E', component='z', line='z', pos=[0.5],
               xscale='linear', yscale='linear', xlim=None, ylim=None,
               figsize=[8,4], title=None, off_screen=False, n=None,
               colors=None, **kwargs):
        '''
        Built-in 1D plotting of a field line using matplotlib

        Parameters:
        ----------
        field: str, default 'E'
            Field magnitude ('E', 'H', or 'J') to plot
            To plot a component 'Ex', 'Hy' is also accepted
        component: str, default 'z'
            Field compoonent ('x', 'y', 'z', 'Abs') to plot. It will be overriden
            if a component is defined in field
        line: str or list, default 'z'
            line of indexes to plot. E.g. line=[0, slice(10,Ny-10), 0]
        pos: float or list, default 0.5
            Float or list of floats betwwen 0-1 indicating the cut position.
            Only used if line is str.
        xlim, ylim: tupple
            limits for x and y axis (see matplotlib.ax.set_xlim for more)
        xscale, yscale: str
            scale to use in x and y axes (see matplotlib.ax.set_xscale for more)
        figsize: list, default [8,4]
            Figure size to pass to the plot initialization
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
        colors: list, optional
            List of matplotlib-compatible colors. len(colors) >= len(pos)
        **kwargs:
            Keyword arguments to be passed to the `matplotlib.plot` function.
            Default kwargs used:
                kwargs = {'color':'g', 'lw':1.2, 'ls':'-'}
        '''
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        xmin, xmax = self.grid.xmin, self.grid.xmax
        ymin, ymax = self.grid.ymin, self.grid.ymax
        zmin, zmax = self.grid.zmin, self.grid.zmax
        _z = self.z

        if self.use_mpi:
            zmin, zmax = self.grid.ZMIN, self.grid.ZMAX
            Nz = self.grid.NZ
            _z = self.grid.Z
            if self.rank == 0:
                fig, ax = plt.subplots(1,1, figsize=figsize)
        else:
            fig, ax = plt.subplots(1,1, figsize=figsize)


        plotkw = {'lw':1.2, 'ls':'-'}

        if colors is None:
            colors = ['k', 'tab:red', 'tab:blue', 'tab:green',
                    'tab:orange', 'tab:purple', 'tab:pink']
        plotkw.update(kwargs)

        if len(field) == 2: #support for e.g. field='Ex'
            component = field[1]
            field = field[0]

        if title is None:
            title = field + component +'1d'

        if type(pos) is not list: #support for a list of cut positions
            pos_arr = [pos]
        else:
            pos_arr = pos

        for i, pos in enumerate(pos_arr):

            if type(line) is not str and len(line) == 3:
                x, y, z = line[0], line[1], line[2]

                #z-axis
                if type(line[2]) is slice:
                    cut = f'(a,b,z) a={round(self.x[x],3)}, b={round(self.y[y],3)}'
                    xax = 'z'
                    xx = _z[z]
                    xlims = (_z[z].min(), _z[z].max())

                #x-axis
                elif type(line[0]) is slice:
                    cut = f'(x,a,b) a={round(self.y[y],3)}, b={round(_z[z],3)}'
                    xax = 'x'
                    xx = self.x[x]
                    xlims = (self.x[x].min(), self.x[x].max())

                #y-axis
                elif type(line[1]) is slice:
                    cut = f'(a,y,b) a={round(self.x[x],3)}, b={round(_z[z],3)}'
                    xax = 'y'
                    xx = self.y[y]
                    xlims = (self.y[y].min(), self.y[y].max())

            elif line.lower() == 'x':
                x, y, z = slice(0,Nx), int(Ny*pos), int(Nz*pos) #x-axis
                cut = f'(x,a,b) a={round(self.y[y],3)}, b={round(_z[z],3)}'
                xax = 'x'
                xx = self.x[x]
                xlims = (xmin, xmax)

            elif line.lower() == 'y':
                x, y, z = int(Nx*pos), slice(0,Ny), int(Nz*pos) #y-axis
                cut = f'(a,y,b) a={round(self.x[x],3)}, b={round(_z[z],3)}'
                xax = 'y'
                xx = self.y[y]
                xlims = (ymin, ymax)

            elif line.lower() == 'z':
                x, y, z = int(Nx*pos), int(Ny*pos), slice(0,Nz) #z-axis
                cut = f'(a,b,z) a={round(self.x[x],3)}, b={round(self.y[y],3)}'
                xax = 'z'
                xx = _z[z]
                xlims = (zmin, zmax)

            else:
                print("line needs to be an array of slices [x,y,z] or a str 'x', 'y', 'z'")

            if i == 0: # first one on top
                zorder = 10
            else:
                zorder = i

            if self.use_mpi: # only in rank=0
                _field = self.mpi_gather(field, x=x, y=y, z=z, component=component)

                if self.rank == 0:
                    ax.plot(xx, _field, color=colors[i], zorder=zorder,
                            label=f'{field}{component}{cut}', **plotkw)
            else:
                if field == 'E':
                    _field = self.E[x, y, z, component]
                if field == 'H':
                    _field = self.H[x, y, z, component]
                if field == 'J':
                    _field = self.J[x, y, z, component]

                ax.plot(xx, _field, color=colors[i], zorder=zorder,
                    label=f'{field}{component}{cut}', **plotkw)

        if not self.use_mpi:
            yax = f'{field}{component} amplitude'

            ax.set_title(f'Wakis {field}{component}'+(len(pos_arr)==1)*f'{cut}')
            ax.set_xlabel(xax)
            ax.set_ylabel(yax, color=colors[0])
            ax.set_xlim(xlims)

            ax.set_xscale(xscale)
            ax.set_yscale(yscale)

            if len(pos_arr) > 1:
                ax.legend(loc=1)

            if xlim is not None:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)

            if n is not None:
                fig.suptitle('$'+field+'_{'+component+'}$ field, timestep='+str(n))
                title += '_'+str(n).zfill(6)

            fig.tight_layout()

            if off_screen:
                fig.savefig(title+'.png')
                plt.clf()
                plt.close(fig)

            else:
                plt.show()

        elif self.use_mpi and self.rank == 0:
            yax = f'{field}{component} amplitude'

            ax.set_title(f'Wakis {field}{component}'+(len(pos_arr)==1)*f'{cut}')
            ax.set_xlabel(xax)
            ax.set_ylabel(yax, color=colors[0])
            ax.set_xlim(xlims)

            ax.set_xscale(xscale)
            ax.set_yscale(yscale)

            if len(pos_arr) > 1:
                ax.legend(loc=1)

            if xlim is not None:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)

            if n is not None:
                fig.suptitle('$'+field+'_{'+component+'}$ field, timestep='+str(n))
                title += '_'+str(n).zfill(6)

            fig.tight_layout()

            if off_screen:
                fig.savefig(title+'.png')
                plt.clf()
                plt.close(fig)

            else:
                plt.show()

    def inspect(self, wake=None, window_size=None, off_screen=False,
                opacity=1.0, inactive_opacity=0.1, add_silhouette=False,
                specular=0.5, smooth_shading=False):

        if self.use_mpi:
            print('[!] plot3D is not supported when `use_mpi=True`')
            return
        if wake is not None:
            self.wake = wake

        # Initialize plotter
        pl = pv.Plotter(window_size=window_size)
        solid_state = {}
        for key, path in self.stl_solids.items():
            surf = self.grid.read_stl(key)
            color = self.stl_colors.get(key, "lightgray")
            actor = pl.add_mesh(surf, color=color, name=key,
                                opacity=inactive_opacity, silhouette=False,
                                smooth_shading=smooth_shading, specular=specular)
            sil = None
            if add_silhouette:
                sil = pl.add_silhouette(surf, color="black", line_width=3.0)
                sil.SetVisibility(False)

            solid_state[key] = {
                "actor": actor,
                "silhouette": sil,
                "active_opacity": opacity,
                "inactive_opacity": inactive_opacity,
                "checked": False,
                "highlight": False,
                "button": None,
            }

        # UI scale and solid checkboxes positioning
        w, h = pl.window_size
        ui = h / 800.0
        box = max(16, int(20 * ui))
        font = max(8, int(12 * ui))
        pad = int(10 * ui)
        dy = box + pad
        cx = int(10 * ui)
        cy = h // 2

        # checkboxes callbacks for solids and master (all On/Off)
        color_on = 'green'
        color_off = 'white'
        def apply(state):
            if state["highlight"]:
                state["actor"].GetProperty().SetOpacity(state["active_opacity"])
                if add_silhouette:
                    state["silhouette"].SetVisibility(True)
            else:
                state["actor"].GetProperty().SetOpacity(state["inactive_opacity"])
                if add_silhouette:
                    state["silhouette"].SetVisibility(False)

        def make_cb(name):
            def _cb(v):
                s = solid_state[name]
                s["checked"] = bool(v)
                s["highlight"] = s["checked"]
                apply(s)
            return _cb

        master_on = True
        def master_cb(v):
            nonlocal master_on
            master_on = bool(v)
            for s in solid_state.values():
                s["checked"] = master_on
                s["highlight"] = master_on
                btn = s["button"]
                if btn:
                    rep = btn.GetRepresentation()
                    if hasattr(rep, "SetState"):
                        rep.SetState(1 if master_on else 0)
                apply(s)

        pl.add_checkbox_button_widget(master_cb, value=False,
                                    color_on=color_on, color_off=color_off,
                                    position=(cx, cy), size=box)
        pl.add_text("All on", position=(cx + box + pad, cy), font_size=font)

        for i, name in enumerate(solid_state):
            y = cy - (i + 2) * dy
            _color_on = self.stl_colors.get(name, color_on)
            if str(_color_on) == 'white' or list(_color_on) == [1.0, 1.0, 1.0]:
                _color_on = 'gray'
            btn = pl.add_checkbox_button_widget(make_cb(name), value=False,
                                                color_on=_color_on,
                                                color_off=color_off,
                                                position=(cx, y), size=box)
            solid_state[name]["button"] = btn
            pl.add_text(name, position=(cx + box + pad, y), font_size=font)

        # Add beam & integration path checkboxes if wake object is passed
        if self.wake is not None:
            z_center = 0.5 * (self.grid.zmin + self.grid.zmax)
            z_height = self.grid.zmax - self.grid.zmin
            radius = 0.005 * max(self.grid.xmax-self.grid.xmin, self.grid.ymax-self.grid.ymin)

            beam = pv.Cylinder(center=(self.wake.xsource, self.wake.ysource, z_center),
                              direction=(0, 0, 1), height=z_height, radius=radius*1.1)
            path = pv.Cylinder(center=(self.wake.xtest, self.wake.ytest, z_center),
                               direction=(0, 0, 1), height=z_height, radius=radius)

            beam_actor = pl.add_mesh(beam, color="orange", name="beam", opacity=1.0)
            beam_actor.SetVisibility(True)
            path_actor = pl.add_mesh(path, color="blue", name="integration_path", opacity=1.0)
            path_actor.SetVisibility(True)

            bx = int(w - box - 200 * ui)
            to = box + pad

            def beam_cb(v):
                beam_actor.SetVisibility(bool(v))
            def path_cb(v):
                path_actor.SetVisibility(bool(v))

            pl.add_checkbox_button_widget(path_cb, value=True,
                                        color_off=color_off, color_on="blue",
                                        position=(bx, cy + dy), size=box)
            pl.add_text("Integration path", position=(bx + to, cy + dy), font_size=font)

            pl.add_checkbox_button_widget(beam_cb, value=True,
                                        color_off=color_off, color_on="orange",
                                        position=(bx, cy), size=box)
            pl.add_text("Beam", position=(bx + to, cy), font_size=font)

        pl.set_background('mistyrose', top='white')
        self._add_logo_widget(pl)
        pl.add_axes()

        # Save
        if off_screen:
            pl.off_screen = True
            return pl
        else:
            pl.show()
            return None

    def _add_logo_widget(self, pl):
        """Add packaged logo via importlib.resources (Python 3.9+)."""
        try:
            from importlib import resources
            # resource inside the installed package (use current package)
            logo_res = resources.files(__package__).joinpath('static', 'img', 'wakis-logo-pink.png')
            with resources.as_file(logo_res) as logo_path:
                pl.add_logo_widget(str(logo_path))
                return
        except Exception as e:
            # fallback to the legacy relative path for dev installs
            try:
                pl.add_logo_widget('../docs/img/wakis-logo-pink.png')
            except Exception:
                if self.verbose > 2:
                    print(f'[!] Could not add logo widget: {e}')