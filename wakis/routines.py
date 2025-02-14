# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

import numpy as np
import h5py
from tqdm import tqdm
from scipy.constants import c as c_light

class RoutinesMixin():

    def emsolve(self, Nt, source=None, save=False, fields=['E'], components=['Abs'], 
            every=1, subdomain=None, plot=False, plot_every=1, use_etd=False, 
            plot3d=False, **kwargs):
        '''
        Run the simulation and save the selected field components in HDF5 files
        for every timestep. Each field will be saved in a separate HDF5 file 'Xy.h5'
        where X is the field and y the component.

        Parameters:
        ----------
        Nt: int
            Number of timesteps to run
        source: source object
            source object from `sources.py` defining the time-dependednt source. 
            It should have an update function `source.update(solver, t)`
        save: bool
            Flag to enable saving the field in HDF5 format
        fields: list, default ['E']
            3D field magnitude ('E', 'H', or 'J') to save
            'Ex', 'Hy', etc., is also accepted and will override 
            the `components` parameter.
        components: list, default ['z']
            Field compoonent ('x', 'y', 'z', 'Abs') to save. It will be overriden
            if a component is specified in the`field` parameter
        every: int, default 1
            Number of timesteps between saves
        subdomain: list, default None
            Slice [x,y,z] of the domain to be saved
        plot: bool, default False
            Flag to enable 2D plotting
        plot3d: bool, default False
            Flag to enable 3D plotting
        plot_every: int
            Number of timesteps between consecutive plots
        **kwargs:
            Keyword arguments to be passed to the Plot2D function.
            * Default kwargs used for 2D plotting: 
                {'field':'E', 'component':'z',
                'plane':'ZY', 'pos':0.5, 'title':'Ez', 
                'cmap':'rainbow', 'patch_reverse':True, 
                'off_screen': True, 'interpolation':'spline36'}
            * Default kwargs used for 3D plotting:
                {'field':'E', 'component':'z',
                'add_stl':None, 'stl_opacity':0.0, 'stl_colors':'white',
                'title':'Ez', 'cmap':'jet', 'clip_volume':False, 'clip_normal':'-y',
                'field_on_stl':True, 'field_opacity':1.0,
                'off_screen':True, 'zoom':1.0, 'nan_opacity':1.0}

        Raises:
        -------
        ImportError:
            If the hdf5 dependency cannot be imported

        Dependencies:
        -------------
        h5py
        '''
        self.Nt = Nt
        if source is not None: self.source = source
        
        if save:

            hfs = {}
            for field in fields:

                if len(field) == 1:
                    for component in components:
                        hfs[field+component] = h5py.File(field+component+'.h5', 'w')

                else:
                    hfs[field] = h5py.File(field+'.h5', 'w')

            for hf in hfs:
                hf['x'], hf['y'], hf['z'] = self.x, self.y, self.z
                hf['t'] = np.arange(0, Nt*self.dt, every*self.dt)

            if subdomain is not None:
                xx, yy, zz = subdomain
            else:
                xx, yy, zz = slice(0,self.Nx), slice(0,self.Ny), slice(0,self.Nz)

        if plot:
            plotkw = {'field':'E', 'component':'z',
                    'plane':'ZY', 'pos':0.5, 'cmap':'rainbow', 
                    'patch_reverse':True, 'title':'Ez', 
                    'off_screen': True, 'interpolation':'spline36'}
            plotkw.update(kwargs)

        if plot3d:
            plotkw = {'field':'E', 'component':'z',
                    'add_stl':None, 'stl_opacity':0.0, 'stl_colors':'white',
                    'title':'Ez', 'cmap':'jet', 'clip_volume':False, 'clip_normal':'-y',
                    'field_on_stl':True, 'field_opacity':1.0,
                    'off_screen':True, 'zoom':1.0, 'nan_opacity':1.0}
            
            plotkw.update(kwargs)

        # get update equations
        if use_etd:
            update = self.one_step_etd
        else:
            update = self.one_step

        # Time loop 
        for n in tqdm(range(Nt)):

            if source is not None: 
                source.update(self, n*self.dt)

            if save:
                for field in hfs.keys():
                    try:
                        d = getattr(self, field[0])[xx,yy,zz,field[1:]]
                    except:
                        raise(f'Component {field} not valid. Input must have a \
                              field ["E", "H", "J"] and a component ["x", "y", "z", "Abs"]')
                    
                    # Save timestep in HDF5
                    hfs[field]['#'+str(n).zfill(5)] = d

            # Advance
            update()

            # Plot
            if plot and n%plot_every == 0:
                self.plot2D(n=n, **plotkw)

            if plot3d and n%plot_every == 0:
                self.plot3D(n=n, **plotkw)

        # End
        if save:
            for hf in hfs:
                hf.close()

    def wakesolve(self, wakelength, wake=None, 
                  save_J=False, add_space=None, use_etd=False,
                  plot=False, plot_from=None, plot_every=1, plot_until=None, 
                  **kwargs):
        '''
        Run the EM simulation and compute the longitudinal (z) and transverse (x,y)
        wake potential WP(s) and impedance Z(s). 
        
        The `Ez` field is saved every timestep in a subdomain (xtest, ytest, z) around 
        the beam trajectory in HDF5 format file `Ez.h5`.

        The computed results are available as Solver class attributes: 
            - wake potential: WP (longitudinal), WPx, WPy (transverse) [V/pC]
            - impedance: Z (longitudinal), Zx, Zy (transverse) [Ohm]
            - beam charge distribution: lambdas (distance) [C/m] lambdaf (spectrum) [C]

        Parameters:
        -----------
        wakelength: float
            Desired length of the wake in [m] to be computed 
            
            Maximum simulation time in [s] can be computed from the wakelength parameter as:
            .. math::    t_{max} = t_{inj} + (wakelength + (z_{max}-z_{min}))/c 
        wake: Wake obj, default None
            `Wake()` object containing the information needed to run 
            the wake solver calculation. See Wake() docstring for more information.
            Can be passed at `Solver()` instantiation as parameter too.
        save_J: bool, default False
            Flag to enable saving the current J in a diferent HDF5 file 'Jz.h5'
        plot: bool, default False
            Flag to enable 2D plotting
        plot_every: int
            Number of timesteps between consecutive plots
        **kwargs:
            Keyword arguments to be passed to the Plot2D function.
            Default kwargs used: 
                {'plane':'ZY', 'pos':0.5, 'title':'Ez', 
                'cmap':'rainbow', 'patch_reverse':True, 
                'off_screen': True, 'interpolation':'spline36'}
        
        Raises:
        -------
        AttributeError:
            If the Wake object is not provided
        ImportError:
            If the hdf5 dependency cannot be imported

        Dependencies:
        -------------
        h5py
        '''

        if wake is not None: self.wake = wake
        if self.wake is None:
            raise('Wake solver information not passed to the solver instantiation')
        
        self.wake.wakelength = wakelength
        self.Ez_file = self.wake.Ez_file

        # beam parameters
        self.q = self.wake.q
        self.ti = self.wake.ti
        self.sigmaz = self.wake.sigmaz
        self.beta = self.wake.beta
        self.v = self.beta*c_light

        # source position
        self.xsource, self.ysource = self.wake.xsource, self.wake.ysource
        self.ixs, self.iys = np.abs(self.x-self.xsource).argmin(), np.abs(self.y-self.ysource).argmin()
        
        # integration path (test position)
        self.xtest, self.ytest = self.wake.xtest, self.wake.ytest
        self.ixt, self.iyt = np.abs(self.x-self.xtest).argmin(), np.abs(self.y-self.ytest).argmin()
        self.add_space = add_space

        # plot params defaults
        if plot:
            plotkw = {'plane':'ZY', 'pos':0.5, 'title':'Ez',
                    'cmap':'rainbow', 'patch_reverse':True,  
                    'off_screen': True, 'interpolation':'spline36'}
            plotkw.update(kwargs)

        def beam(self, t):
            '''
            Update the current J every timestep 
            to introduce a gaussian beam 
            moving in +z direction
            '''
            s0 = self.z.min() - self.v*self.ti
            s = self.z - self.v*t

            # gaussian
            profile = 1/np.sqrt(2*np.pi*self.sigmaz**2)*np.exp(-(s-s0)**2/(2*self.sigmaz**2))

            # update 
            self.J[self.ixs,self.iys,:,'z'] = self.q*self.v*profile/self.dx/self.dy
            
        tmax = (wakelength + self.ti*self.v + (self.z.max()-self.z.min()))/self.v #[s]
        Nt = int(tmax/self.dt)
        xx, yy = slice(self.ixt-1, self.ixt+2), slice(self.iyt-1, self.iyt+2)
        if add_space is not None and add_space !=0:
            zz = slice(add_space, -add_space)
        else: 
            zz = slice(0, self.Nz)

        # hdf5 
        hf = h5py.File(self.Ez_file, 'w')
        hf['x'], hf['y'], hf['z'] = self.x[xx], self.y[yy], self.z[zz]
        hf['t'] = np.arange(0, Nt*self.dt, self.dt)

        if save_J:
            hfJ = h5py.File('Jz.h5', 'w')
            hfJ['x'], hfJ['y'], hfJ['z'] = self.x[xx], self.y[yy], self.z[zz]
            hfJ['t'] = np.arange(0, Nt*self.dt, self.dt)

        # get update equations
        if use_etd:
            update = self.one_step_etd
        else:
            update = self.one_step

        if plot_until is None: plot_until = Nt
        if plot_from is None: plot_from = int(self.ti/self.dt)

        print('Running electromagnetic time-domain simulation...')
        for n in tqdm(range(Nt)):

            # Initial condition
            beam(self, n*self.dt)

            # Save
            hf['#'+str(n).zfill(5)] = self.E[xx, yy, zz, 'z'] 
            if save_J:
                hfJ['#'+str(n).zfill(5)] = self.J[xx, yy, zz, 'z'] 
            
            # Advance
            update()
            
            # Plot
            if plot:
                if n%plot_every == 0 and n<plot_until and n>plot_from:
                    self.plot2D(field='E', component='z', n=n, **plotkw)
                else:
                    pass

        hf.close()
        if save_J:
            hfJ.close()
        
        # wake computation 
        self.wake.solve()

        self.wakelength = wakelength
        self.s = self.wake.s
        self.WP = self.wake.WP
        self.WPx = self.wake.WPx
        self.WPy = self.wake.WPy
        self.Z = self.wake.Z
        self.Zx = self.wake.Zx
        self.Zy = self.wake.Zy
        self.lambdas = self.wake.lambdas 
        self.lambdaf = self.wake.lambdaf     