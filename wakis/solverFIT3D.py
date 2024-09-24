# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

from tqdm import tqdm

import numpy as np
import time

from scipy.constants import c as c_light, epsilon_0 as eps_0, mu_0 as mu_0
from scipy.sparse import csc_matrix as sparse_mat
from scipy.sparse import diags, hstack, vstack

from .field import Field
from .materials import material_lib
from .plotting import PlotMixin

try:
    from cupyx.scipy.sparse import csc_matrix as gpu_sparse_mat
    imported_cupyx = True
except ImportError:
    imported_cupyx = False

class SolverFIT3D(PlotMixin):

    def __init__(self, grid, wake=None, cfln=0.5, dt=None,
                 bc_low=['Periodic', 'Periodic', 'Periodic'],
                 bc_high=['Periodic', 'Periodic', 'Periodic'],
                 use_stl=False, use_conductors=False, use_gpu=False,
                 bg=[1.0, 1.0], verbose=1):
        '''
        Class holding the 3D time-domain electromagnetic solver 
        algorithm based on the Finite Integration Technique (FIT)

        Parameters:
        -----------
        grid: GridFIT3D object
            Instance of GridFIT3D class containing the simulation mesh and the 
            imported geometry
        wake: WakeSolver object, optional
            Instance of WakeSolver class containing the beam parameters. Needed to 
            run a wakefield simulation to compute wake potential and impedance
        cfln: float, default 0.5
            Convergence condition by Courant–Friedrichs–Lewy, used to compute the
            simulation timestep
        dt: float, optional
            Simulation timestep. If not None, it overrides the cfln-based timestep
        bc_low: list, default ['Periodic', 'Periodic', 'Periodic']
            Domain box boundary conditions for X-, Y-, Z-
        bc_high: list, default ['Periodic', 'Periodic', 'Periodic']
            Domain box boundary conditions for X+, Y+, Z+
        use_conductors: bool, default False
            If true, enables geometry import based on elements from `conductors.py`
        use_stl: bool, default False
            If true, activates all the solids and materials passed to the `grid` object
        use_gpu: bool, default False, 
            Using cupyx, enables GPU accelerated computation of every timestep
        bg: list, default [1.0, 1.0]
            Background material for the simulation box [eps_r, mu_r, sigma]. Default is vacuum.
            It supports any material from the material library in `materials.py`, of a 
            custom list of floats. If conductivity (sigma) is passed, 
            it enables flag: use_conductivity
        verbose: int or bool, default True
            Enable verbose ouput on the terminal if 1 or True

        Attributes
        ----------
        E: Field object
            Object to access the Electric field data in [V/m]. 
            E.g.: solver.E[:,:,n,'z'] gives a 2D numpy.ndarray fieldmap of Ez component, located at the n-th cell in z
        H: Field object
            Object to access the Magnetic field data in [A/m]. 
            E.g.: solver.H[i,j,k,'x'] gives a point value of Hx component, located at the i,j,k cell
        J: Field object
            Object to access the Current density field data in [A/m^2]. 
        ieps: Field object
            Object to access the ε^-1 tensor containing 1/permittivity values in the 3 dimensions. 
        imu: Field object
            Object to access the μ^-1 tensor containing 1/permeability values in the 3 dimensions.
        sigma: Field object
            Object to access the condutcity σ tensor in the 3 dimensions.
        '''

        self.verbose = verbose
        if verbose:  t0 = time.time()

        # Flags
        self.step_0 = True
        self.plotter_active = False
        self.use_conductors = use_conductors
        self.use_stl = use_stl
        self.use_gpu = use_gpu
        self.activate_abc = False        # Will turn true if abc BCs are chosen
        self.activate_pml = False        # Will turn true if pml BCs are chosen
        self.use_conductivity = False    # Will turn true if conductive material or pml is added

        if use_stl:
            self.use_conductors = False

        # Grid 
        self.grid = grid
        self.cfln = cfln
        if dt is None:
            self.dt = cfln / (c_light * np.sqrt(1 / self.grid.dx ** 2 + 1 / self.grid.dy ** 2 + 1 / self.grid.dz ** 2))
        else:
            self.dt = dt

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

        # Wake computation
        self.wake = wake

        # Fields
        self.E = Field(self.Nx, self.Ny, self.Nz, use_gpu=self.use_gpu)
        self.H = Field(self.Nx, self.Ny, self.Nz, use_gpu=self.use_gpu)
        self.J = Field(self.Nx, self.Ny, self.Nz, use_gpu=self.use_gpu)

        # Matrices
        if verbose: print('Assembling operator matrices...')
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
        if verbose: print('Appliying boundary conditions...')
        self.bc_low = bc_low
        self.bc_high = bc_high
        self.npml = 10

        self.apply_bc_to_C() 

        # Materials 
        if verbose: print('Adding material tensors...')
        if type(bg) is str:
            bg = material_lib[bg.lower()]

        if len(bg) == 3:
            self.eps_bg, self.mu_bg, self.sigma_bg = bg[0]*eps_0, bg[1]*mu_0, bg[2]
            self.use_conductivity = True
        else:
            self.eps_bg, self.mu_bg, self.sigma_bg = bg[0]*eps_0, bg[1]*mu_0, 0.0

        self.ieps = Field(self.Nx, self.Ny, self.Nz, use_ones=True)*(1./self.eps_bg) 
        self.imu = Field(self.Nx, self.Ny, self.Nz, use_ones=True)*(1./self.mu_bg) 
        self.sigma = Field(self.Nx, self.Ny, self.Nz, use_ones=True)*self.sigma_bg

        if self.use_stl:
            self.apply_stl()

        # Fill PML BCs
        if self.activate_pml:
            if verbose: print('Filling PML sigmas...')
            self.fill_pml_sigmas()

        self.iDeps = diags(self.ieps.toarray(), shape=(3*N, 3*N), dtype=float)
        self.iDmu = diags(self.imu.toarray(), shape=(3*N, 3*N), dtype=float)
        self.Dsigma = diags(self.sigma.toarray(), shape=(3*N, 3*N), dtype=float)

        # Pre-computing
        if verbose: print('Pre-computing ...') 
        self.tDsiDmuiDaC = self.tDs * self.iDmu * self.iDa * self.C 
        self.itDaiDepsDstC = self.itDa * self.iDeps * self.Ds * self.C.transpose()
        
        # Move to GPU
        if use_gpu:
            if verbose: print('Moving to GPU...') 
            if imported_cupyx:
                self.tDsiDmuiDaC = gpu_sparse_mat(self.tDsiDmuiDaC)
                self.itDaiDepsDstC = gpu_sparse_mat(self.itDaiDepsDstC)
                self.iDeps = gpu_sparse_mat(self.iDeps)
                self.Dsigma = gpu_sparse_mat(self.Dsigma)
            else:
                print('*** cupyx could not be imported, please check CUDA installation')

        if verbose:  print(f'Total initialization time: {time.time() - t0} s')

    def update_tensors(self, tensor='ieps'):
        '''Update tensor matrices after 
        Field ieps, imu or sigma have been modified 
        and pre-compute the time-stepping matrices

        Parameters:
        -----------
        tensor : str, default 'all'
            Name of the tensor to update: 'ieps', 'imu', 'sigma' 
            for permitivity, permeability and conductivity, respectively. 
            If left to default 'all', all thre tensors will be recomputed. 
        '''
        if self.verbose: print(f'Re-computing tensor "{tensor}"...') 

        if tensor == 'ieps': 
            self.iDeps = diags(self.ieps.toarray(), shape=(3*self.N, 3*self.N), dtype=float)
        elif tensor =='imu':
            self.iDmu = diags(self.imu.toarray(), shape=(3*self.N, 3*self.N), dtype=float)
        elif tensor == 'sigma':
            self.Dsigma = diags(self.sigma.toarray(), shape=(3*self.N, 3*self.N), dtype=float)
        elif tensor == 'all':
            self.iDeps = diags(self.ieps.toarray(), shape=(3*self.N, 3*self.N), dtype=float)
            self.iDmu = diags(self.imu.toarray(), shape=(3*self.N, 3*self.N), dtype=float)
            self.Dsigma = diags(self.sigma.toarray(), shape=(3*self.N, 3*self.N), dtype=float)

        if self.verbose: print('Re-Pre-computing ...') 
        self.tDsiDmuiDaC = self.tDs * self.iDmu * self.iDa * self.C 
        self.itDaiDepsDstC = self.itDa * self.iDeps * self.Ds * self.C.transpose()
        self.step_0 = False

    def one_step(self):

        if self.step_0:
            self.set_ghosts_to_0()
            self.step_0 = False
            self.attrcleanup()

        self.H.fromarray(self.H.toarray() -
                         self.dt*self.tDsiDmuiDaC*self.E.toarray()
                         )
     
        self.E.fromarray(self.E.toarray() +
                         self.dt*(self.itDaiDepsDstC * self.H.toarray() 
                                  - self.iDeps*self.J.toarray()
                                  )
                         )
        
        #include current computation                 
        if self.use_conductivity:
            self.J.fromarray(self.Dsigma*self.E.toarray())
     
        #update ABC
        if self.activate_abc:
            self.update_abc()


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

        if save:
            try:
                import h5py
            except:
                raise('Python package `h5py` is needed to save field data in HDF5 format')

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

            if source is not None: #TODO test
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
                  plot=False, plot_every=1, plot_until=None, **kwargs):
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

        try:
            import h5py
        except:
            raise('Python package `h5py` is needed to save field data in HDF5 format')
        
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
                if n%plot_every == 0 and n<plot_until and n>int(self.ti/self.dt):
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
        if any(True for x in self.bc_low if x.lower() in ('electric','pec','pml')):
    
            if self.bc_low[0].lower() in ('electric','pec', 'pml'):
                xlo = 0
            if self.bc_low[1].lower() in ('electric','pec', 'pml'):
                ylo = 0    
            if self.bc_low[2].lower() in ('electric','pec', 'pml'):
                zlo = 0   
            if self.bc_high[0].lower() in ('electric','pec', 'pml'):
                xhi = 0
            if self.bc_high[1].lower() in ('electric','pec', 'pml'):
                yhi = 0
            if self.bc_high[2].lower() in ('electric','pec', 'pml'):
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
            if self.bc_high[0].lower() == 'abc':
                self.tL[-1, :, :, 'x'] = self.L[0, :, :, 'x']
                self.itA[-1, :, :, 'y'] = self.iA[0, :, :, 'y']
                self.itA[-1, :, :, 'z'] = self.iA[0, :, :, 'z']

            if self.bc_high[1].lower() == 'abc':
                self.tL[:, -1, :, 'y'] = self.L[:, 0, :, 'y']
                self.itA[:, -1, :, 'x'] = self.iA[:, 0, :, 'x']
                self.itA[:, -1, :, 'z'] = self.iA[:, 0, :, 'z']

            if self.bc_high[2].lower() == 'abc':
                self.tL[:, :, -1, 'z'] = self.L[:, :, 0, 'z']
                self.itA[:, :, -1, 'x'] = self.iA[:, :, 0, 'x']
                self.itA[:, :, -1, 'y'] = self.iA[:, :, 0, 'y']

            self.tDs = diags(self.tL.toarray(), shape=(3*self.N, 3*self.N), dtype=float)
            self.itDa = diags(self.itA.toarray(), shape=(3*self.N, 3*self.N), dtype=float)
            self.activate_abc = True

        # Perfect Matching Layers (PML)
        if any(True for x in self.bc_low if x.lower() == 'pml'):
            self.activate_pml = True
            self.use_conductivity = True

    def fill_pml_sigmas(self):
        '''
        Routine to calculate pml sigmas and apply them 
        to the conductivity tensor sigma

        [IN-PROGRESS]
        '''

        # Initialize
        sx, sy, sz = np.zeros(self.Nx), np.zeros(self.Ny), np.zeros(self.Nz)
        pml_exp = 2

        # Fill
        if self.bc_low[0].lower() == 'pml':
            sx[0:self.npml] = eps_0/(2*self.dt)*((self.x[self.npml] - self.x[:self.npml])/(self.npml*self.dx))**pml_exp
            for d in ['x', 'y', 'z']:
                for i in range(self.npml):
                    self.sigma[i, :, :, d] = sx[i]
                    if sx[i] > 0 : self.ieps[i, :, :, d] = 1/(eps_0+sx[i]*(2*self.dt)) 

        if self.bc_low[1].lower() == 'pml':
            sy[0:self.npml] = 1/(2*self.dt)*((self.y[self.npml] - self.y[:self.npml])/(self.npml*self.dy))**pml_exp
            for d in ['x', 'y', 'z']:
                for j in range(self.npml):
                    self.sigma[:, j, :, d] = sy[j]
                    if sy[j] > 0 : self.ieps[:, j, :, d] = 1/(eps_0+sy[j]*(2*self.dt)) 

        if self.bc_low[2].lower() == 'pml':
            #sz[0:self.npml] = eps_0/(2*self.dt)*((self.z[self.npml] - self.z[:self.npml])/(self.npml*self.dz))**pml_exp
            sz[0:self.npml] = 2*((self.z[self.npml] - self.z[:self.npml])/(self.npml*self.dz))**pml_exp
            for d in ['x', 'y', 'z']:
                for k in range(self.npml):
                    self.sigma[:, :, k, d] = sz[k]
                    if sz[k] > 0. : self.ieps[:, :, k, d] = 1/(np.mean(sz[:self.npml])*eps_0) 

        if self.bc_high[0].lower() == 'pml':
            sx[-self.npml:] = 1/(2*self.dt)*((self.x[-self.npml:] - self.x[-self.npml])/(self.npml*self.dx))**pml_exp
            for d in ['x', 'y', 'z']:
                for i in range(self.npml):
                    i +=1
                    self.sigma[-i, :, :, 'x'] = sx[-i]
                    if sx[-i] > 0 : self.ieps[-i, :, :, d] = 1/(eps_0+sx[-i]*(2*self.dt)) 

        if self.bc_high[1].lower() == 'pml':
            sy[-self.npml:] = 1/(2*self.dt)*((self.y[-self.npml:] - self.y[-self.npml])/(self.npml*self.dy))**pml_exp
            for d in ['x', 'y', 'z']:
                for j in range(self.npml):
                    j +=1
                    self.sigma[:, -j, :, d] = sy[-j]
                    if sy[-j] > 0 : self.ieps[:, -j, :, d] = 1/(eps_0+sy[-j]*(2*self.dt)) 

        if self.bc_high[2].lower() == 'pml':
            #sz[-self.npml:] = eps_0/(2*self.dt)*((self.z[-self.npml:] - self.z[-self.npml])/(self.npml*self.dz))**pml_exp
            sz[-self.npml:] = 2*((self.z[-self.npml:] - self.z[-self.npml])/(self.npml*self.dz))**pml_exp
            for d in ['x', 'y', 'z']:
                for k in range(self.npml):
                    k +=1
                    self.sigma[:, :, -k, d] = sz[-k]
                    self.ieps[:, :, -k, d] = 1/(np.mean(sz[-self.npml:])*eps_0)


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
                self.E[-1, :, :, d] = self.E[-1, :, :, d]
                self.H[-1, :, :, d] = self.H[-1, :, :, d] 

        if self.bc_high[1].lower() == 'abc':
            for d in ['x', 'y', 'z']:
                self.E[:, -1, :, d] = self.E[:, -1, :, d]
                self.H[:, -1, :, d] = self.H[:, -1, :, d] 

        if self.bc_high[2].lower() == 'abc':
            for d in ['x', 'y', 'z']:
                self.E[:, :, -1, d] = self.E[:, :, -1, d]
                self.H[:, :, -1, d] = self.H[:, :, -1, d] 

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

                # Conductivity
                if len(material_lib[mat_key]) == 3:
                    sigma = material_lib[mat_key][2]
                    self.sigma += self.sigma * (-1.0*mask)
                    self.sigma += mask * sigma
                    self.use_conductivity = True
                
                elif self.sigma_bg > 0.0: # assumed sigma = 0
                    self.sigma += self.sigma * (-1.0*mask)

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

                # Conductivity
                if len(self.stl_materials[key]) == 3:
                    sigma = self.stl_materials[key][2]
                    self.sigma += self.sigma * (-1.0*mask)
                    self.sigma += mask * sigma
                    self.use_conductivity = True

                elif self.sigma_bg > 0.0: # assumed sigma = 0
                    self.sigma += self.sigma * (-1.0*mask)

    def attrcleanup(self):
        # Fields
        del self.L, self.tL, self.iA, self.itA
        if hasattr(self, 'BC'):
            del self.BC
            del self.Dbc

        # Matrices
        del self.Px, self.Py, self.Pz
        del self.Ds, self.iDa, self.tDs, self.itDa
        del self.C