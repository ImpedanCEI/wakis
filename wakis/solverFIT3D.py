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

class SolverFIT3D(PlotMixin, RoutinesMixin):

    def __init__(self, grid, wake=None, cfln=0.5, dt=None,
                 bc_low=['Periodic', 'Periodic', 'Periodic'],
                 bc_high=['Periodic', 'Periodic', 'Periodic'],
                 use_stl=False, use_conductors=False, use_gpu=False,
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

        self.ieps = Field(self.Nx, self.Ny, self.Nz, use_ones=True)*(1./self.eps_bg) 
        self.imu = Field(self.Nx, self.Ny, self.Nz, use_ones=True)*(1./self.mu_bg) 
        self.sigma = Field(self.Nx, self.Ny, self.Nz, use_ones=True)*self.sigma_bg

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
        self.iDeps = diags(self.ieps.toarray(), shape=(3*N, 3*N), dtype=float)
        self.iDmu = diags(self.imu.toarray(), shape=(3*N, 3*N), dtype=float)
        self.Dsigma = diags(self.sigma.toarray(), shape=(3*N, 3*N), dtype=float)

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

    def save_state(self, filename="solver_state.h5",
                   close=True):
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
        state = h5py.File(filename, "w") 
        if imported_cupyx:
            state.create_dataset("H", data=self.H.toarray().get())
            state.create_dataset("E", data=self.E.toarray().get())
            state.create_dataset("J", data=self.J.toarray().get())

        else:
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
