import numpy as np
from scipy.constants import c as c_light, epsilon_0 as eps_0, mu_0 as mu_0
from scipy.sparse import csc_matrix as sparse_mat
from scipy.sparse import diags, block_diag, hstack, vstack
from scipy.sparse.linalg import inv
from field import Field

class SolverFIT3D:

    def __init__(self, grid, tgrid=None, sol_type='FIT', cfln=1.0,
                 bc_low=['Periodic', 'Periodic', 'Periodic'],
                 bc_high=['Periodic', 'Periodic', 'Periodic'],
                 i_s=0, j_s=0, k_s=0, N_pml_low=None, N_pml_high=None):

        # Grid 
        self.sol_type = sol_type
        self.grid = grid
        self.cfln = cfln
        self.dt = cfln / (c_light * np.sqrt(1 / self.grid.dx ** 2 + 1 / self.grid.dy ** 2 +
                                            1 / self.grid.dz ** 2))
        self.Nx = self.grid.nx
        self.Ny = self.grid.ny
        self.Nz = self.grid.nz
        self.N = self.Nx*self.Ny*self.Nz

        self.dx = self.grid.dx
        self.dy = self.grid.dy
        self.dz = self.grid.dz
        self.Sx = self.grid.dy*self.grid.dz
        self.Sy = self.grid.dx*self.grid.dz
        self.Sz = self.grid.dx*self.grid.dy

        if tgrid is None:
            self.tgrid = grid
        else: 
            self.tgrid = tgrid

        self.L = self.grid.L
        self.iA = self.grid.iA
        self.tL = self.grid.tL
        self.itA = self.grid.itA

        # Lengths and Areas using grid flags
        '''
        self.L = Field(self.Nx, self.Ny, self.Nz)
        self.tL = Field(self.Nx, self.Ny, self.Nz)
        #self.A = Field(self.Nx, self.Ny, self.Nz)
        self.iA = Field(self.Nx, self.Ny, self.Nz)
        self.itA = Field(self.Nx, self.Ny, self.Nz)

        self.L.field_x = self.grid.flag_int_cell_yz[:-1,:, :]*self.dx
        self.L.field_y = self.grid.flag_int_cell_zx[:,:-1, :]*self.dy
        self.L.field_z = self.grid.flag_int_cell_xy[:,:, :-1]*self.dz

        self.iA.field_x = self.grid.flag_int_cell_yz[:-1,:, :]/self.dy/self.dz
        self.iA.field_y = self.grid.flag_int_cell_zx[:,:-1, :]/self.dz/self.dx
        self.iA.field_z = self.grid.flag_int_cell_xy[:,:, :-1]/self.dx/self.dy

        self.itA.field_x = self.tgrid.flag_int_cell_yz[:-1,:, :]/self.dy/self.dz
        self.itA.field_y = self.tgrid.flag_int_cell_zx[:,:-1, :]/self.dz/self.dx
        self.itA.field_z = self.tgrid.flag_int_cell_xy[:,:, :-1]/self.dx/self.dy

        self.tL.field_x = self.tgrid.flag_int_cell_yz[:-1,:, :]*self.dx
        self.tL.field_y = self.tgrid.flag_int_cell_zx[:,:-1, :]*self.dy
        self.tL.field_z = self.tgrid.flag_int_cell_xy[:,:, :-1]*self.dz
        '''

        # Fields
        self.E = Field(self.Nx, self.Ny, self.Nz)
        self.H = Field(self.Nx, self.Ny, self.Nz)
        self.J = Field(self.Nx, self.Ny, self.Nz)

        # Matrices
        Nx = self.Nx
        Ny = self.Ny
        Nz = self.Nz
        N = self.N

        self.Px = diags([-1, 1], [0, 1], shape=(N, N), dtype=np.int8)
        self.Py = diags([-1, 1], [0, Nx], shape=(N, N), dtype=np.int8)
        self.Pz = diags([-1, 1], [0, Nx*Ny], shape=(N, N), dtype=np.int8)

        '''
        # Construct diagonal matrices from grid data [OLD]

        self.Ds = block_diag((
                             diags([self.dx], shape=(N, N), dtype=float),
                             diags([self.dy], shape=(N, N), dtype=float),
                             diags([self.dz], shape=(N, N), dtype=float)
                             ))

        self.Da = block_diag((
                             diags([self.Sx], shape=(N, N), dtype=float),
                             diags([self.Sy], shape=(N, N), dtype=float),
                             diags([self.Sz], shape=(N, N), dtype=float)
                             ))

        self.iDs = block_diag((
                             diags([1/self.dx], shape=(N, N), dtype=float),
                             diags([1/self.dy], shape=(N, N), dtype=float),
                             diags([1/self.dz], shape=(N, N), dtype=float)
                             ))

        self.iDa = block_diag((
                             diags([1/self.Sx], shape=(N, N), dtype=float),
                             diags([1/self.Sy], shape=(N, N), dtype=float),
                             diags([1/self.Sz], shape=(N, N), dtype=float)
                             ))

        self.tDs = self.Ds
        self.tDa = self.Da

        self.itDs = self.iDs
        self.itDa = self.iDa
        '''

        # original grid
        self.Ds = diags(self.L.toarray(), shape=(3*N, 3*N), dtype=float)
        self.iDa = diags(self.iA.toarray(), shape=(3*N, 3*N), dtype=float)

        # tilde grid
        self.tDs = diags(self.tL.toarray(), shape=(3*N, 3*N), dtype=float)
        self.itDa = diags(self.itA.toarray(), shape=(3*N, 3*N), dtype=float)

        # Materials [TODO]
        self.epsx = eps_0 
        self.epsy = eps_0
        self.epsz = eps_0

        self.mux = mu_0
        self.muy = mu_0
        self.muz = mu_0

        self.iMeps = block_diag((
                               diags([1/self.epsx], shape=(N, N), dtype=float),
                               diags([1/self.epsy], shape=(N, N), dtype=float),
                               diags([1/self.epsz], shape=(N, N), dtype=float)
                               ))

        self.iMmu = block_diag((
                             diags([1/self.mux], shape=(N, N), dtype=float),
                             diags([1/self.muy], shape=(N, N), dtype=float),
                             diags([1/self.muz], shape=(N, N), dtype=float)
                             ))

        # Curl matrix
        self.C = vstack([
                            hstack([sparse_mat((N,N)), -self.Pz, self.Py]),
                            hstack([self.Pz, sparse_mat((N,N)), -self.Px]),
                            hstack([-self.Py, self.Px, sparse_mat((N,N))])
                        ])
        

        # Boundaries
        self.bc_low = bc_low
        self.bc_high = bc_high

        self.apply_bc()

        # Pre-computing
        self.iMuiDaCDs = self.iMmu * self.iDa * self.C * self.Ds
        self.iMepsitDaCttDs = self.iMeps * self.itDa * self.C.transpose() * self.tDs

    def one_step(self):

        self.H.fromarray(self.H.toarray() -
                         self.dt*self.iMuiDaCDs*self.E.toarray()
                         )

        self.E.fromarray(self.E.toarray() +
                         self.dt*(self.iMepsitDaCttDs * self.H.toarray() - self.iMeps*self.J.toarray())
                         )


    def apply_bc(self):

        # Perodic: by default enforeced
        '''
        if self.bc_low[0].lower() == 'periodic':
            xlo = 1
        if self.bc_low[1].lower() == 'periodic':
            ylo = 1    
        if self.bc_low[2].lower() == 'periodic':
            zlo = 1   
        if self.bc_high[0].lower() == 'periodic':
            xhi = 1
        if self.bc_high[1].lower() == 'periodic':
            yhi = 1
        if self.bc_high[2].lower() == 'periodic':
            zhi = 1
        '''

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

            '''
            for d in ['x', 'y', 'z']:
                self.BC[0, :, :, d] = xlo
                self.BC[:, 0, :, d] = ylo
                self.BC[:, :, 0, d] = zlo
                self.BC[-1, :, :, d] = xhi
                self.BC[:, -1, :, d] = yhi
                self.BC[:, :, -1, d] = zhi
            '''
            for d in ['x', 'y', 'z']: #tangential to zero
                if d is not 'x':
                    self.BC[0, :, :, d] = xlo
                    self.BC[-1, :, :, d] = xhi
                if d is not 'y':
                    self.BC[:, 0, :, d] = ylo
                    self.BC[:, -1, :, d] = yhi
                if d is not 'z':
                    self.BC[:, :, 0, d] = zlo
                    self.BC[:, :, -1, d] = zhi
            
            self.Dbc = diags(self.BC.toarray(),
                            shape=(3*self.N, 3*self.N), 
                            dtype=np.int8
                            )

            # Update iMeps
            self.iMeps = self.Dbc*self.iMeps*self.Dbc


        # Dirichlet PMC: tangential H field = 0 at boundary
        if any(True for x in self.bc_low if x.lower() == 'magnetic' or x.lower() == 'pmc'):

            if self.bc_low[0].lower() == 'magnetic' or self.bc_low[1] == 'pmc':
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
            '''
            for d in ['x', 'y', 'z']: #all components to 0
                self.BC[0, :, :, d] = xlo
                self.BC[:, 0, :, d] = ylo
                self.BC[:, :, 0, d] = zlo
                self.BC[-1, :, :, d] = xhi
                self.BC[:, -1, :, d] = yhi
                self.BC[:, :, -1, d] = zhi
            '''
            for d in ['x', 'y', 'z']: #tangential to zero
                if d is not 'x':
                    self.BC[0, :, :, d] = xlo
                    self.BC[-1, :, :, d] = xhi
                if d is not 'y':
                    self.BC[:, 0, :, d] = ylo
                    self.BC[:, -1, :, d] = yhi
                if d is not 'z':
                    self.BC[:, :, 0, d] = zlo
                    self.BC[:, :, -1, d] = zhi
            
            self.Dbc = diags(self.BC.toarray(),
                            shape=(3*self.N, 3*self.N), 
                            dtype=np.int8
                            )

            # Update iMmu
            self.iMmu = self.Dbc*self.iMmu*self.Dbc

        # Open; pml [TODO]
        '''
        if self.bc_low[0] == 'pml':
            self.N_pml_low[0] = 10 if N_pml_low is None else N_pml_low[0]
        if self.bc_low[1] == 'pml':
            self.N_pml_low[1] = 10 if N_pml_low is None else N_pml_low[1]
        if self.bc_low[2] == 'pml':
            self.N_pml_low[2] = 10 if N_pml_low is None else N_pml_low[2]
        if self.bc_high[0] == 'pml':
            self.N_pml_high[0] = 10 if N_pml_high is None else N_pml_high[0]
        if self.bc_high[1] == 'pml':
            self.N_pml_high[1] = 10 if N_pml_high is None else N_pml_high[1]
        if self.bc_high[2] == 'pml':
            self.N_pml_high[2] = 10 if N_pml_high is None else N_pml_high[2]
        '''

        # Update C
        #self.C = self.Dbc*self.C*self.Dbc
