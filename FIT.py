''' 3D FIT solver

Notation:
tX == tilde X
iX == (X)^-1
itX == (tilde(X))^-1

linear numbering:
n = 1 + (i-1) + (j-1)*Nx + (k-1)*Nx*Ny
len(n) = Nx*Ny*Nz
'''


import numpy as np
from scipy.constants import c as c_light, epsilon_0 as eps_0, mu_0 as mu_0

def allocate_matrices():
    pass 

def compute_Ds(self, dx, dy, dz, Nx, Ny, Nz):
    '''
    Main grid mesh step diagonal matrix
    '''
    Nx = self.grid.nx
    Ny = self.grid.ny
    Nz = self.grid.nz

    dx = self.grid.dx
    dy = self.grid.dy
    dz = self.grid.dz

    Ds = np.diag( np.concatenate((dx*np.ones(N), dy*np.ones(N), dz*np.ones(N))) )

def compute_Da(self, dx, dy, dz, Nx, Ny, Nz):
    '''
    Main grid face areas diagonal matrix
    '''
    Nx = self.grid.nx
    Ny = self.grid.ny
    Nz = self.grid.nz

    dx = self.grid.dx
    dy = self.grid.dy
    dz = self.grid.dz

    Da = np.diag( np.concatenate((dy*dz*np.ones(N), dx*dz*np.ones(N), dx*dy*np.ones(N))) )

def compute_iDs(self, dx, dy, dz, Nx, Ny, Nz):
    '''
    Main grid mesh step diagonal matrix
    '''
    Nx = self.grid.nx
    Ny = self.grid.ny
    Nz = self.grid.nz

    dx = self.grid.dx
    dy = self.grid.dy
    dz = self.grid.dz

    Ds = np.diag( np.concatenate((1/dx*np.ones(N), 1/dy*np.ones(N), 1/dz*np.ones(N))) )

def compute_tDs(self, dx, dy, dz, x, y, z):
    '''
    Tilde or Dual grid mesh step diagonal matrix

    Main grid: 
    x = xmin, xmin + dx, ..., xmax
    len(x) = Nx + 1

    Dual grid
    u = xmin + dx/2, xmin + 3dx/2, ..., 0
      = (x[i+1] + x[i])/2 with u[-1] = 0
    len(u) = Nx + 1 

    mesh step Dual == mesh step Main
    '''
    x = self.grid.x
    y = self.grid.y
    z = self.grid.z

    u = x + dx/2
    v = y + dy/2
    w = z + dz/2

    u[-1] = 0.
    v[-1] = 0.
    w[-1] = 0.

    tDs = np.diag( np.concatenate((
                   dx*np.ones(Nx)
                   dy*np.ones(Ny)
                   dz*np.ones(Nz) )) 
                 )


def compute_tDa(self, dx, dy, dz, Nx, Ny, Nz):
    '''
    Dual grid face areas diagonal matrix
    '''
    Nx = self.grid.nx
    Ny = self.grid.ny
    Nz = self.grid.nz

    dx = self.grid.dx
    dy = self.grid.dy
    dz = self.grid.dz

    tDa = np.diag( np.concatenate((
                  dy*dz*np.ones(Nx), 
                  dx*dz*np.ones(Ny), 
                  dx*dy*np.ones(Nz) )) 
                )


def compute_itDa(self, dx, dy, dz, Nx, Ny, Nz):
    '''
    Dual grid face areas diagonal matrix
    '''
    Nx = self.grid.nx
    Ny = self.grid.ny
    Nz = self.grid.nz

    dx = self.grid.dx
    dy = self.grid.dy
    dz = self.grid.dz

    itDa = np.diag( np.concatenate((
                  1/(dy*dz)*np.ones(Nx), 
                  1/(dx*dz)*np.ones(Ny), 
                  1/(dx*dy)*np.ones(Nz) )) 
                )

def update_b_FIT(dt, Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz, Nx, Ny, Nz, Ds, iDa, C):
    '''
    tX == tilde X
    iX == (X)^-1
    '''

    for ii in range(Nx):
        for jj in range(Ny):
            for kk in range(Nz):
                
                n = 1 + (i-1) + (j-1)*Nx + (k-1)*Nx*Ny

                Bx[n] = Bx[n] - dt*iDa[n]*C[n]*Ds[n]*Ex[n]

                By[n] = By[n] - dt*iDa[n]*C[n]*Ds[n]*Ey[n]

                Bz[n] = Bz[n] - dt*iDa[n]*C[n]*Ds[n]*Ez[n]

def update_e_FIT(dt, Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz, Nx, Ny, Nz, Dk, tDs, iDeps, iDmu, itDa, tC):
    '''
    tX == tilde X
    iX == (X)^-1
    '''

    for ii in range(Nx):
        for jj in range(Ny):
            for kk in range(Nz):

                    n = 1 + (i-1) + (j-1)*Nx + (k-1)*Nx*Ny

                    Dalpha[n] = exp(-iDeps[n]*Dk[n]*dt)

                    Ex[n] = (Dalpha[n]*Ex[n] + \
                                      (1-Dalpha[n])*iDk[n]*itDa[n] \
                                      *C[n]*iDmu[n]*Bx[n] - \
                                      (1-Dalpha[n])*iDk[n]*Jx[n])

                    Ey[n] = (Dalpha[n]*Ey[n] + \
                                      (1-Dalpha[n])*iDk[n]*itDa[n] \
                                      *C[n]*iDmu[n]*By[n] - \
                                      (1-Dalpha[n])*iDk[n]*Jy[n])

                    Ez[n] = (Dalpha[n]*Ez[n] + \
                                      (1-Dalpha[n])*iDk[n]*itDa[n] \
                                      *C[n]*iDmu[n]*Bz[n] - \
                                      (1-Dalpha[n])*iDk[n]*Jz[n])

''' 
def update_b_FIT(dt, Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz, Nx, Ny, Nz, Ds, iDa, C):

    for ii in range(1,Nx):
        for jj in range(1, Ny):
            for kk in range(1, Nz):
                
                Bx[ii, jj, kk] = Bx[ii, jj, kk] - \
                                 dt*iDa[ii, jj, kk]*C[ii, jj, kk]*Ds[ii, jj, kk]*Ex[ii, jj, kk]

                By[ii, jj, kk] = By[ii, jj, kk] - \
                                 dt*iDa[ii, jj, kk]*C[ii, jj, kk]*Ds[ii, jj, kk]*Ey[ii, jj, kk]

                Bz[ii, jj, kk] = Bz[ii, jj, kk] - \
                                 dt*iDa[ii, jj, kk]*C[ii, jj, kk]*Ds[ii, jj, kk]*Ez[ii, jj, kk]


def update_e_FIT(dt, Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz, Nx, Ny, Nz, Dk, tDs, iDeps, iDmu, itDa, tC):

    for ii in range(1,Nx):
        for jj in range(1, Ny):
            for kk in range(1, Nz):

                    Dalpha[ii, jj, kk] = exp(-iDeps[ii, jj, kk]*Dk[ii, jj, kk]*dt

                    Ex[ii, jj, kk] = (Dalpha[ii, jj, kk]*Ex[ii, jj, kk] + \
                                      (1-Dalpha[ii, jj, kk])*iDk[ii, jj, kk]*itDa[ii, jj, kk] \
                                      *C[ii, jj, kk]*iDmu[ii, jj, kk]*Bx[ii, jj, kk]) - \
                                      (1-Dalpha[ii, jj, kk])*iDk[ii, jj, kk]*Jx[ii, jj, kk]

                    Ey[ii, jj, kk] = (Dalpha[ii, jj, kk]*Ey[ii, jj, kk] + \
                                      (1-Dalpha[ii, jj, kk])*iDk[ii, jj, kk]*itDa[ii, jj, kk] \
                                      *C[ii, jj, kk]*iDmu[ii, jj, kk]*By[ii, jj, kk]) - \
                                      (1-Dalpha[ii, jj, kk])*iDk[ii, jj, kk]*Jy[ii, jj, kk]

                    Ez[ii, jj, kk] = (Dalpha[ii, jj, kk]*Ez[ii, jj, kk] + \
                                      (1-Dalpha[ii, jj, kk])*iDk[ii, jj, kk]*itDa[ii, jj, kk] \
                                      *C[ii, jj, kk]*iDmu[ii, jj, kk]*Bz[ii, jj, kk]) - \
                                      (1-Dalpha[ii, jj, kk])*iDk[ii, jj, kk]*Jz[ii, jj, kk]
'''