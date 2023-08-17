import numpy as np
from scipy.constants import c as c_light, epsilon_0 as eps_0, mu_0 as mu_0
from scipy.sparse import csc_matrix as sparse_mat
from scipy.sparse import diags, block_diag, hstack, vstack
from scipy.sparse.linalg import inv
from field import Field

class SolverFIT3D:

    def __init__(self, grid, sol_type, cfln,
                 bc_low=['Dirichlet', 'Dirichlet', 'Dirichlet'],
                 bc_high=['Dirichlet', 'Dirichlet', 'Dirichlet'],
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

        self.L = Field(self.Nx, self.Ny, self.Nz)
        self.A = Field(self.Nx, self.Ny, self.Nz)
        self.iA = Field(self.Nx, self.Ny, self.Nz)

        # Lengths and Areas
        self.L.field_x = self.grid.l_x[:,:-1, :-1]
        self.L.field_y = self.grid.l_y[:-1,:, :-1]
        self.L.field_z = self.grid.l_z[:-1,:-1, :]

        self.A.field_x = self.grid.Syz[:-1,:, :].astype(int)*self.dy*self.dz
        self.A.field_y = self.grid.Szx[:,:-1, :].astype(int)*self.dz*self.dx
        self.A.field_z = self.grid.Sxy[:,:, :-1].astype(int)*self.dx*self.dy

        self.iA.field_x = self.grid.Syz[:-1,:, :].astype(int)/self.dy/self.dz
        self.iA.field_y = self.grid.Szx[:,:-1, :].astype(int)/self.dz/self.dx
        self.iA.field_z = self.grid.Sxy[:,:, :-1].astype(int)/self.dx/self.dy

        # Materials [TODO]
        self.epsx = eps_0 
        self.epsy = eps_0
        self.epsz = eps_0

        self.mux = mu_0
        self.muy = mu_0
        self.muz = mu_0


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
        self.Ds = diags(self.L.toarray(), shape=(3*N, 3*N), dtype=float)
        self.Da = diags(self.A.toarray(), shape=(3*N, 3*N), dtype=float)

        self.iDs = diags(
                        np.divide(1, self.L.toarray(), 
                                  out=np.zeros_like(self.L.toarray()), 
                                  where=self.L.toarray()!=0 ), 
                        shape=(3*N, 3*N), dtype=float
                        )

        # idea: construct iA from grid3d's Sxy
        self.iDa = diags(self.iA.toarray(), shape=(3*N, 3*N), dtype=float)
        '''
        self.iDa = diags(
                        np.divide(1, self.A.toarray(), 
                                  out=np.zeros_like(self.A.toarray()), 
                                  where=self.A.toarray()!=0 ), 
                        shape=(3*N, 3*N), dtype=float
                        )
        '''

        self.tDs = self.Ds
        self.tDa = self.Da

        self.itDs = self.iDs
        self.itDa = self.iDa
        

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

        self.C = vstack([
                            hstack([sparse_mat((N,N)), -self.Pz, self.Py]),
                            hstack([self.Pz, sparse_mat((N,N)), -self.Px]),
                            hstack([-self.Py, self.Px, sparse_mat((N,N))])
                        ])
        

        # Boundaries
        self.bc_low = bc_low
        self.bc_high = bc_high

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


