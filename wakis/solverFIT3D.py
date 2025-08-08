# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

from tqdm import tqdm

import numpy as np
import time
import h5py

from scipy.constants import c as c_light, epsilon_0 as eps_0, mu_0 as mu_0
from scipy.sparse import csc_matrix as sparse_mat
from scipy.sparse import diags, hstack, vstack

from .field import Field
from .materials import material_lib
from .plotting import PlotMixin
from .routines import RoutinesMixin

try:
    from cupyx.scipy.sparse import csc_matrix as gpu_sparse_mat
    imported_cupyx = True
except ImportError:
    imported_cupyx = False

try:
    from sparse_dot_mkl import dot_product_mkl
    imported_mkl = True
except ImportError:
    imported_mkl = False


class SolverFIT3D(PlotMixin, RoutinesMixin):

    def __init__(self, grid, wake=None, cfln=0.5, dt=None,
                 bc_low=['Periodic', 'Periodic', 'Periodic'],
                 bc_high=['Periodic', 'Periodic', 'Periodic'],
                 use_stl=False, use_conductors=False, 
                 use_gpu=False, use_mpi=False, dtype=np.float64,
                 n_pml=10, bg=[1.0, 1.0], verbose=1):
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
        self.nstep = int(0)
        self.plotter_active = False
        self.use_conductors = use_conductors
        self.use_stl = use_stl
        self.use_gpu = use_gpu
        self.use_mpi = use_mpi
        self.activate_abc = False        # Will turn true if abc BCs are chosen
        self.activate_pml = False        # Will turn true if pml BCs are chosen
        self.use_conductivity = False    # Will turn true if conductive material or pml is added
        self.imported_mkl = imported_mkl # Use MKL backend when available

        if use_stl:
            self.use_conductors = False

        # Grid 
        self.grid = grid

        self.Nx = self.grid.Nx
        self.Ny = self.grid.Ny
        self.Nz = self.grid.Nz
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
        self.dtype = dtype
        self.E = Field(self.Nx, self.Ny, self.Nz, use_gpu=self.use_gpu, dtype=self.dtype)
        self.H = Field(self.Nx, self.Ny, self.Nz, use_gpu=self.use_gpu, dtype=self.dtype)
        self.J = Field(self.Nx, self.Ny, self.Nz, use_gpu=self.use_gpu, dtype=self.dtype)

        # MPI init
        if self.use_mpi:
            if self.grid.use_mpi: 
                self.mpi_initialize()
            else:
                print('*** Grid not subdivided for MPI, set `use_mpi`=True also in `GridFIT3D` to enable MPI')
            
        # Matrices
        if verbose: print('Assembling operator matrices...')
        N = self.N
        self.Px = diags([-1, 1], [0, 1], shape=(N, N), dtype=np.int8)
        self.Py = diags([-1, 1], [0, self.Nx], shape=(N, N), dtype=np.int8)
        self.Pz = diags([-1, 1], [0, self.Nx*self.Ny], shape=(N, N), dtype=np.int8)

        # original grid
        self.Ds = diags(self.L.toarray(), shape=(3*N, 3*N), dtype=self.dtype)
        self.iDa = diags(self.iA.toarray(), shape=(3*N, 3*N), dtype=self.dtype)

        # tilde grid
        self.tDs = diags(self.tL.toarray(), shape=(3*N, 3*N), dtype=self.dtype)
        self.itDa = diags(self.itA.toarray(), shape=(3*N, 3*N), dtype=self.dtype)

        # Curl matrix
        self.C = vstack([
                            hstack([sparse_mat((N,N)), -self.Pz, self.Py]),
                            hstack([self.Pz, sparse_mat((N,N)), -self.Px]),
                            hstack([-self.Py, self.Px, sparse_mat((N,N))])
                        ], dtype=np.int8)
                
        # Boundaries
        if verbose: print('Applying boundary conditions...')
        self.bc_low = bc_low
        self.bc_high = bc_high
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

        self.ieps = Field(self.Nx, self.Ny, self.Nz, use_ones=True, dtype=self.dtype)*(1./self.eps_bg) 
        self.imu = Field(self.Nx, self.Ny, self.Nz, use_ones=True, dtype=self.dtype)*(1./self.mu_bg) 
        self.sigma = Field(self.Nx, self.Ny, self.Nz, use_ones=True, dtype=self.dtype)*self.sigma_bg

        if self.use_stl:
            self.apply_stl()

        # Fill PML BCs
        if self.activate_pml:
            if verbose: print('Filling PML sigmas...')
            self.n_pml = n_pml
            self.pml_lo = 5e-3
            self.pml_hi = 1.e-1
            self.pml_func = np.geomspace
            self.fill_pml_sigmas()

        # Timestep calculation 
        if verbose: print('Calculating maximal stable timestep...') 
        self.cfln = cfln
        if dt is None:
            self.dt = cfln / (c_light * np.sqrt(1 / self.grid.dx ** 2 + 1 / self.grid.dy ** 2 + 1 / self.grid.dz ** 2))
        else:
            self.dt = dt
        
        if self.use_conductivity: # relaxation time criterion tau

            mask = np.logical_and(self.sigma.toarray()!=0, #for non-conductive
                                self.ieps.toarray()!=0)  #for PEC eps=inf
            
            self.tau = (1/self.ieps.toarray()[mask]) / \
                        self.sigma.toarray()[mask] 
            
            if self.dt > self.tau.min():
                self.dt = self.tau.min()

        # Pre-computing
        if verbose: print('Pre-computing...') 
        self.iDeps = diags(self.ieps.toarray(), shape=(3*N, 3*N), dtype=self.dtype)
        self.iDmu = diags(self.imu.toarray(), shape=(3*N, 3*N), dtype=self.dtype)
        self.Dsigma = diags(self.sigma.toarray(), shape=(3*N, 3*N), dtype=self.dtype)

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
                raise ImportError('*** cupyx could not be imported, please check CUDA installation')

        if verbose:  print(f'Total initialization time: {time.time() - t0} s')

    def update_tensors(self, tensor='all'):
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
            self.iDeps = diags(self.ieps.toarray(), shape=(3*self.N, 3*self.N), dtype=self.dtype)
        elif tensor =='imu':
            self.iDmu = diags(self.imu.toarray(), shape=(3*self.N, 3*self.N), dtype=self.dtype)
        elif tensor == 'sigma':
            self.Dsigma = diags(self.sigma.toarray(), shape=(3*self.N, 3*self.N), dtype=self.dtype)
        elif tensor == 'all':
            self.iDeps = diags(self.ieps.toarray(), shape=(3*self.N, 3*self.N), dtype=self.dtype)
            self.iDmu = diags(self.imu.toarray(), shape=(3*self.N, 3*self.N), dtype=self.dtype)
            self.Dsigma = diags(self.sigma.toarray(), shape=(3*self.N, 3*self.N), dtype=self.dtype)

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

    def one_step_mkl(self):
        if self.step_0:
            self.set_ghosts_to_0()
            self.step_0 = False
            self.attrcleanup()

        self.H.fromarray(self.H.toarray() -
                         self.dt*dot_product_mkl(self.tDsiDmuiDaC,self.E.toarray())
                         )
     
        self.E.fromarray(self.E.toarray() +
                         self.dt*(dot_product_mkl(self.itDaiDepsDstC,self.H.toarray()) 
                                  - dot_product_mkl(self.iDeps,self.J.toarray())
                                  )
                         )
        
        #include current computation                 
        if self.use_conductivity:
            self.J.fromarray(dot_product_mkl(self.Dsigma,self.E.toarray()))

    def mpi_initialize(self):
        self.comm = self.grid.comm
        self.rank = self.grid.rank
        self.size = self.grid.size

        self.NZ = self.grid.NZ
        self.ZMIN = self.grid.ZMIN
        self.ZMAX = self.grid.ZMAX
        self.Z = self.grid.Z

    def mpi_one_step(self):
        if self.step_0:
            self.set_ghosts_to_0()
            self.step_0 = False
            self.attrcleanup()

        self.H.fromarray(self.H.toarray() -
                         self.dt*self.tDsiDmuiDaC*self.E.toarray()
                         )
        
        self.mpi_communicate(self.H)
        self.mpi_communicate(self.J)
        self.E.fromarray(self.E.toarray() +
                         self.dt*(self.itDaiDepsDstC * self.H.toarray() 
                                  - self.iDeps*self.J.toarray()
                                  )
                         )

        self.mpi_communicate(self.E)
        # include current computation                 
        if self.use_conductivity:
            self.J.fromarray(self.Dsigma*self.E.toarray())

    def mpi_communicate(self, field):
        if self.use_gpu:
            field.from_gpu()

        # ghosts lo
        if self.rank > 0:
            for d in ['x','y','z']:
                self.comm.Sendrecv(field[:, :, 1, d], 
                            recvbuf=field[:, :, 0, d],
                            dest=self.rank-1, sendtag=0,
                            source=self.rank-1, recvtag=1)
        # ghosts hi
        if self.rank < self.size - 1:
            for d in ['x','y','z']:
                self.comm.Sendrecv(field[:, :, -2, d], 
                            recvbuf=field[:, :, -1, d], 
                            dest=self.rank+1, sendtag=1,
                            source=self.rank+1, recvtag=0)
        
        if self.use_gpu:
            field.to_gpu()

    def mpi_gather(self, field, x=None, y=None, z=None, component=None):
        '''
        Gather a specific component or slice of a distributed field from all MPI ranks.

        This function collects a selected component of a field (E, H, J, or custom) 
        from all MPI processes along the z-axis and reconstructs the global field data 
        on the root rank (rank 0). The user can specify slices or single indices 
        along x, y, and z to control the subset of data gathered.

        Parameters
        ----------
        field : str or Field obj
            The field to gather. If a string, it must begin with one of:
            - `'E'`, `'H'`, or `'J'` followed optionally by a component label 
            (e.g., `'Ex'`, `'Hz'`, `'JAbs'`). 
            If no component is specified, defaults to `'z'`.

        x : int or slice, optional
            Range of x-indices to gather. If None, defaults to the full x-range.

        y : int or slice, optional
            Range of y-indices to gather. If None, defaults to the full y-range.

        z : int or slice, optional
            Range of z-indices to gather. If None, defaults to the full z-range.

        component : str or slice, optional
            Component of the field to gather ('x', 'y', 'z', or a slice). 
            If None and not inferred from `field`, defaults to `'z'`.

        Returns
        -------
        numpy.ndarray or None
            The gathered field values assembled on rank 0 with shape depending on
            the selected slices along (x, y, z). Returns `None` on non-root ranks.

        Notes
        -----
        - Assumes field data is distributed along the z-dimension.
        - Automatically handles removal of ghost cells in reconstruction.
        - Field components are inferred from the input `field` string or the 
        `component` argument.
        - This method supports full 3D subvolume extraction or 1D/2D slices 
        for performance diagnostics and visualization.

        Examples
        --------
        >>> # Gather Ex component on full domain on rank 0
        >>> global_Ex = solver.mpi_gather('Ex')
        
        >>> # Gather a 2D yz-slice at x=10 of the J field
        >>> yz_J = solver.mpi_gather('J', x=10)
        '''

        if x is None:
            x = slice(0, self.Nx)
        if y is None:
            y = slice(0, self.Ny)
        if z is None:
            z = slice(0, self.NZ)

        if type(field) is str:
            if len(field) == 2: #support for e.g. field='Ex'
                component = field[1]
                field = field[0]
            elif len(field) == 4: #support for Abs
                component = field[1:]
                field = field[0]
            elif component is None: 
                component = 'z'
                print("[!] `component` not specified, using default component='z'")

            if field == 'E':
                local = self.E[x, y, :, component].ravel()
            elif field == 'H':
                local = self.H[x, y, :, component].ravel()
            elif field == 'J':
                local = self.J[x, y, :, component].ravel()
        else:
            if component is None: 
                component = 'z'
                print("[!] `component` not specified, using default component='z'")
            local = field[x, y, :, component].ravel() 

        buffer = self.comm.gather(local, root=0)
        _field = None

        if self.rank == 0:
            if type(x) is int and type(y) is int:  # 1d array at x=a, y=b
                nz = self.NZ//self.size
                _field = np.zeros((self.NZ))
                for r in range(self.size):
                    zz = np.s_[r*nz:(r+1)*nz]
                    if r == 0:
                        _field[zz] = np.reshape(buffer[r], (nz+self.grid.n_ghosts))[:-1] 
                    elif r == (self.size-1):
                        _field[zz] = np.reshape(buffer[r], (nz+self.grid.n_ghosts))[1:]  
                    else:
                        _field[zz] = np.reshape(buffer[r], (nz+2*self.grid.n_ghosts))[1:-1]
                _field = _field[z]

            elif type(x) is int:    # 2d slice at x=a
                ny = y.stop-y.start
                nz = self.NZ//self.size
                _field = np.zeros((ny, self.NZ))
                for r in range(self.size):
                    zz = np.s_[r*nz:(r+1)*nz]
                    if r == 0:
                        _field[:, zz] = np.reshape(buffer[r], (ny, nz+self.grid.n_ghosts))[:, :-1] 
                    elif r == (self.size-1):
                        _field[:, zz] = np.reshape(buffer[r], (ny, nz+self.grid.n_ghosts))[:, 1:]  
                    else:
                        _field[:, zz] = np.reshape(buffer[r], (ny, nz+2*self.grid.n_ghosts))[:, 1:-1] 
                _field = _field[:, z]

            elif type(y) is int:  # 2d slice at y=a
                nx = x.stop-x.start
                nz = self.NZ//self.size
                _field = np.zeros((nx, self.NZ))
                for r in range(self.size):
                    zz = np.s_[r*nz:(r+1)*nz]
                    if r == 0:
                        _field[:, zz] = np.reshape(buffer[r], (nx, nz+self.grid.n_ghosts))[:, :-1] 
                    elif r == (self.size-1):
                        _field[:, zz] = np.reshape(buffer[r], (nx, nz+self.grid.n_ghosts))[:, 1:]  
                    else:
                        _field[:, zz] = np.reshape(buffer[r], (nx, nz+2*self.grid.n_ghosts))[:, 1:-1] 
                _field = _field[:, z]
                        
            else: # both type slice -> 3d array
                nx = x.stop-x.start
                ny = y.stop-y.start
                nz = self.NZ//self.size
                _field = np.zeros((nx, ny, self.NZ))
                for r in range(self.size):
                    zz = np.s_[r*nz:(r+1)*nz]
                    if r == 0:
                        _field[:, :, zz] = np.reshape(buffer[r], (nx, ny, nz+self.grid.n_ghosts))[:, :, :-1] 
                    elif r == (self.size-1):
                        _field[:, :, zz] = np.reshape(buffer[r], (nx, ny, nz+self.grid.n_ghosts))[:, :, 1:]  
                    else:
                        _field[:, :, zz] = np.reshape(buffer[r], (nx, ny, nz+2*self.grid.n_ghosts))[:, :, 1:-1] 
                _field = _field[:, :, z]

        return _field 

    def mpi_gather_asField(self, field):
        '''
        Gather distributed field data from all MPI ranks and return a global Field object.

        This method collects the specified electromagnetic field data (E, H, or J) 
        from all MPI processes and assembles it into a single global `Field` object 
        on the root rank (rank 0). The field data can be specified as a string 
        ('E', 'H', or 'J') or as a `wakis.Field` object.

        Parameters
        ----------
        field : str or Field obj
            The field to gather. If a string, it must be one of:
            - `'E'` for the electric field
            - `'H'` for the magnetic field
            - `'J'` for the current density

            Passing a `wakis.Field` is also supported (e.g. ieps, imu, sigma)

        Returns
        -------
        Field
            A `wakis.Field` object containing the gathered global field data 
            with shape (Nx, Ny, NZ, 3). Only returned on rank 0. On other 
            ranks, the returned value is undefined and should not be used.

        Notes
        -----
        - This method assumes the field is distributed along the `z`-axis.
        - Ghost cells are removed appropriately when reassembling the global field.
        '''

        _field = Field(self.Nx, self.Ny, self.NZ) 

        for d in ['x','y','z']:
            if type(field) is str:
                if field == 'E':
                    local = self.E[:, :, :,d].ravel()
                elif field == 'H':
                    local = self.H[:, :, :,d].ravel()
                elif field == 'J':
                    local = self.J[:, :, :,d].ravel()
            else:
                local = field[:, :, :, d].ravel()

            buffer = self.comm.gather(local, root=0)
            if self.rank == 0:
                nz = self.NZ//self.size
                for r in range(self.size):
                    zz = np.s_[r*nz:(r+1)*nz]
                    if r == 0:
                        _field[:, :, zz, d] = np.reshape(buffer[r], (self.Nx, self.Ny, nz+self.grid.n_ghosts))[:, :, :-1] 
                    elif r == (self.size-1):
                        _field[:, :, zz, d] = np.reshape(buffer[r], (self.Nx, self.Ny, nz+self.grid.n_ghosts))[:, :, 1:]  
                    else:
                        _field[:, :, zz, d] = np.reshape(buffer[r], (self.Nx, self.Ny, nz+2*self.grid.n_ghosts))[:, :, 1:-1] 
        
        return _field 

    def apply_bc_to_C(self):
        '''
        Modifies rows or columns of C and tDs and itDa matrices
        according to bc_low and bc_high
        '''
        xlo, ylo, zlo = 1., 1., 1.
        xhi, yhi, zhi = 1., 1., 1.

        # Check BCs for internal MPI subdomains 
        if self.use_mpi and self.grid.use_mpi:
            if self.rank > 0:
                self.bc_low=['pec', 'pec', 'mpi'] 

            if self.rank < self.size - 1:
                self.bc_high=['pec', 'pec', 'mpi']

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
        if any(True for x in self.bc_low if x.lower() in ('electric','pec','pml')) \
           or any(True for x in self.bc_high if x.lower() in ('electric','pec','pml')):
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
        if any(True for x in self.bc_low if x.lower() in ('magnetic','pmc')) \
           or any(True for x in self.bc_high if x.lower() in ('magnetic','pmc')):
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
        if any(True for x in self.bc_low if x.lower() == 'abc') \
           or any(True for x in self.bc_high if x.lower() == 'abc'):
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
        if any(True for x in self.bc_low if x.lower() == 'pml') \
           or any(True for x in self.bc_high if x.lower() == 'pml'):
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
            #sx[0:self.n_pml] = eps_0/(2*self.dt)*((self.x[self.n_pml] - self.x[:self.n_pml])/(self.n_pml*self.dx))**pml_exp
            sx[0:self.n_pml] = np.linspace( self.pml_hi, self.pml_lo, self.n_pml)
            for d in ['x', 'y', 'z']:
                for i in range(self.n_pml):
                    self.ieps[i, :, :, d] = 1./eps_0 
                    self.sigma[i, :, :, d] = sx[i]
                    #if sx[i] > 0 : self.ieps[i, :, :, d] = 1/(eps_0+sx[i]*(2*self.dt)) 

        if self.bc_low[1].lower() == 'pml':
            #sy[0:self.n_pml] = 1/(2*self.dt)*((self.y[self.n_pml] - self.y[:self.n_pml])/(self.n_pml*self.dy))**pml_exp
            sy[0:self.n_pml] = self.pml_func( self.pml_hi, self.pml_lo, self.n_pml)
            for d in ['x', 'y', 'z']:
                for j in range(self.n_pml):
                    self.ieps[:, j, :, d] = 1./eps_0 
                    self.sigma[:, j, :, d] = sy[j]
                    #if sy[j] > 0 : self.ieps[:, j, :, d] = 1/(eps_0+sy[j]*(2*self.dt)) 

        if self.bc_low[2].lower() == 'pml':
            #sz[0:self.n_pml] = eps_0/(2*self.dt)*((self.z[self.n_pml] - self.z[:self.n_pml])/(self.n_pml*self.dz))**pml_exp
            sz[0:self.n_pml] = self.pml_func( self.pml_hi, self.pml_lo, self.n_pml)
            for d in ['x', 'y', 'z']:
                for k in range(self.n_pml):
                    self.ieps[:, :, k, d] = 1./eps_0 
                    self.sigma[:, :, k, d] = sz[k]
                    #if sz[k] > 0. : self.ieps[:, :, k, d] = 1/(np.mean(sz[:self.n_pml])*eps_0) 

        if self.bc_high[0].lower() == 'pml':
            #sx[-self.n_pml:] = 1/(2*self.dt)*((self.x[-self.n_pml:] - self.x[-self.n_pml])/(self.n_pml*self.dx))**pml_exp
            sx[-self.n_pml:] = self.pml_func( self.pml_lo, self.pml_hi, self.n_pml)
            for d in ['x', 'y', 'z']:
                for i in range(self.n_pml):
                    i +=1
                    self.ieps[-i, :, :, d] = 1./eps_0 
                    self.sigma[-i, :, :, d] = sx[-i]
                    #if sx[-i] > 0 : self.ieps[-i, :, :, d] = 1/(eps_0+sx[-i]*(2*self.dt)) 

        if self.bc_high[1].lower() == 'pml':
            #sy[-self.n_pml:] = 1/(2*self.dt)*((self.y[-self.n_pml:] - self.y[-self.n_pml])/(self.n_pml*self.dy))**pml_exp
            sy[-self.n_pml:] = self.pml_func( self.pml_lo, self.pml_hi, self.n_pml)
            for d in ['x', 'y', 'z']:
                for j in range(self.n_pml):
                    j +=1
                    self.ieps[:, -j, :, d] = 1./eps_0 
                    self.sigma[:, -j, :, d] = sy[-j]
                    #if sy[-j] > 0 : self.ieps[:, -j, :, d] = 1/(eps_0+sy[-j]*(2*self.dt)) 

        if self.bc_high[2].lower() == 'pml':
            #sz[-self.n_pml:] = eps_0/(2*self.dt)*((self.z[-self.n_pml:] - self.z[-self.n_pml])/(self.n_pml*self.dz))**pml_exp
            sz[-self.n_pml:] = self.pml_func( self.pml_lo, self.pml_hi, self.n_pml)
            for d in ['x', 'y', 'z']:
                for k in range(self.n_pml):
                    k +=1
                    self.ieps[:, :, -k, d] = 1./eps_0 
                    self.sigma[:, :, -k, d] = sz[-k]
                    #self.ieps[:, :, -k, d] = 1/(np.mean(sz[-self.n_pml:])*eps_0)

    def get_abc(self):
        '''
        Save the n-2 timestep to apply ABC 
        '''
        E_abc, H_abc = {}, {}

        if self.bc_low[0].lower() == 'abc':
            E_abc[0] = {}
            H_abc[0] = {}
            for d in ['x', 'y', 'z']:
                E_abc[0][d+'lo'] = self.E[1, :, :, d]
                H_abc[0][d+'lo'] = self.H[1, :, :, d]  

        if self.bc_low[1].lower() == 'abc':
            E_abc[1] = {}
            H_abc[1] = {}
            for d in ['x', 'y', 'z']:
                E_abc[1][d+'lo'] = self.E[:, 1, :, d]
                H_abc[1][d+'lo'] = self.H[:, 1, :, d]  
                   
        if self.bc_low[2].lower() == 'abc':
            E_abc[2] = {}
            H_abc[2] = {}
            for d in ['x', 'y', 'z']:
                E_abc[2][d+'lo'] = self.E[:, :, 1, d]
                H_abc[2][d+'lo'] = self.H[:, :, 1, d]  

        if self.bc_high[0].lower() == 'abc':
            E_abc[0] = {}
            H_abc[0] = {}
            for d in ['x', 'y', 'z']:
                E_abc[0][d+'hi'] = self.E[-1, :, :, d]
                H_abc[0][d+'hi'] = self.H[-1, :, :, d]  

        if self.bc_high[1].lower() == 'abc':
            E_abc[1] = {}
            H_abc[1] = {}
            for d in ['x', 'y', 'z']:
                E_abc[1][d+'hi'] = self.E[:, -1, :, d]
                H_abc[1][d+'hi'] = self.H[:, -1, :, d]  
                   
        if self.bc_high[2].lower() == 'abc':
            E_abc[2] = {}
            H_abc[2] = {}
            for d in ['x', 'y', 'z']:
                E_abc[2][d+'hi'] = self.E[:, :, -1, d]
                H_abc[2][d+'hi'] = self.H[:, :, -1, d]  

        return E_abc, H_abc

    def update_abc(self, E_abc, H_abc):
        '''
        Apply ABC algo to the selected BC, 
        to be applied after each timestep
        '''

        if self.bc_low[0].lower() == 'abc':
            for d in ['x', 'y', 'z']:
                self.E[0, :, :, d] = E_abc[0][d+'lo']
                self.H[0, :, :, d] = H_abc[0][d+'lo']  

        if self.bc_low[1].lower() == 'abc':
            for d in ['x', 'y', 'z']:
                self.E[:, 0, :, d] = E_abc[1][d+'lo']
                self.H[:, 0, :, d] = H_abc[1][d+'lo'] 
                   
        if self.bc_low[2].lower() == 'abc':
            for d in ['x', 'y', 'z']:
                self.E[:, :, 0, d] = E_abc[2][d+'lo']
                self.H[:, :, 0, d] = H_abc[2][d+'lo']   

        if self.bc_high[0].lower() == 'abc':
            for d in ['x', 'y', 'z']:
                self.E[-1, :, :, d] = E_abc[0][d+'hi']
                self.H[-1, :, :, d] = H_abc[0][d+'hi']

        if self.bc_high[1].lower() == 'abc':
            for d in ['x', 'y', 'z']:
                self.E[:, -1, :, d] = E_abc[1][d+'hi']
                self.H[:, -1, :, d] = H_abc[1][d+'hi']

        if self.bc_high[2].lower() == 'abc':
            for d in ['x', 'y', 'z']:
                self.E[:, :, -1, d] = E_abc[2][d+'hi']
                self.H[:, :, -1, d] = H_abc[2][d+'hi'] 

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

    def save_state(self, filename="solver_state.h5", close=True):
        """Save the solver state to an HDF5 file.
        
        This function saves only the key state variables (`H`, `E`, `J`) that are updated
        in `one_step()`, storing them as datasets in an HDF5 file.
        
        Parameters:
        -----------
        filename : str, optional
            The name of the HDF5 file where the state will be stored. Default is "solver_state.h5".
        close : bool, optional (default=True)
            - If True, the HDF5 file is closed after saving, and the function returns nothing.
            - If False, the function returns an open HDF5 file object, which must be closed manually.

        Returns:
        --------
        h5py.File or None
            - If `close=True`, nothing is returned.
            - If `close=False`, returns an open `h5py.File` object for further manipulation.
        """
        
        if self.use_mpi: # MPI savestate
            E = self.mpi_gather_asField('E')
            H = self.mpi_gather_asField('H')
            J = self.mpi_gather_asField('J')

            if self.rank == 0:
                    state = h5py.File(filename, "w") 
                    state.create_dataset("H", data=self.H.toarray())
                    state.create_dataset("E", data=self.E.toarray())
                    state.create_dataset("J", data=self.J.toarray())
            # TODO: check for MPI-GPU

        elif self.use_gpu: # GPU savestate
            state = h5py.File(filename, "w") 
            state.create_dataset("H", data=self.H.toarray().get())
            state.create_dataset("E", data=self.E.toarray().get())
            state.create_dataset("J", data=self.J.toarray().get())

        else: # CPU savestate
            state = h5py.File(filename, "w") 
            state.create_dataset("H", data=self.H.toarray())
            state.create_dataset("E", data=self.E.toarray())
            state.create_dataset("J", data=self.J.toarray())

        if close:
            state.close()
        else:
            return state  # Caller must close this manually

    def load_state(self, filename="solver_state.h5"):
        """Load the solver state from an HDF5 file.
        
        Reads the saved state variables (`H`, `E`, `J`) from the specified HDF5 file
        and restores them to the solver.

        Parameters:
        -----------
        filename : str, optional
            The name of the HDF5 file to load the solver state from. Default is "solver_state.h5".

        Returns:
        --------
        None
        """
        state = h5py.File(filename, "r")  
        
        self.E.fromarray(state["E"][:])
        self.H.fromarray(state["H"][:])
        self.J.fromarray(state["J"][:])

        # TODO: support MPI loadstate

        state.close()

    def read_state(self, filename="solver_state.h5"):
        """Open an HDF5 file for reading without loading its contents.

        This function returns an open `h5py.File` object, allowing the caller
        to manually inspect or extract data as needed. The file must be closed
        by the caller after use.

        Parameters:
        -----------
        filename : str, optional
            The name of the HDF5 file to open. Default is "solver_state.h5".

        Returns:
        --------
        h5py.File
            An open HDF5 file object in read mode.
        """
        return h5py.File(filename, "r")  
    
    def reset_fields(self):
        """
        Resets the electromagnetic field components to zero.

        This function clears the electric field (E), magnetic field (H), and 
        current density (J) by setting all their components to zero in the 
        simulation domain. It ensures a clean restart for a new simulation.

        Notes
        -----
        - This method is useful when reusing an existing simulation object 
        without reinitializing all attributes.
        """
        for d in ['x', 'y', 'z']:
            self.E[:, :, :, d] = 0.0
            self.H[:, :, :, d] = 0.0
            self.J[:, :, :, d] = 0.0