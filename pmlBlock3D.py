import numpy as np
from scipy.constants import c as c_light, epsilon_0 as eps_0, mu_0 as mu_0


class PmlBlock3D:
    def __init__(self, Nx, Ny, Nz, dt, dx, dy, dz, lx_block=None, ly_block=None, lz_block=None, rx_block=None,
                 ry_block=None, rz_block=None):
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.dt = dt
        self.dx = dx
        self.dy = dy
        self.dz = dz

        self.Ex = np.zeros((Nx, Ny + 1, Nz + 1))
        self.Ey = np.zeros((Nx + 1, Ny, Nz + 1))
        self.Ez = np.zeros((Nx + 1, Ny + 1, Nz))
        self.Exy = np.zeros((Nx, Ny + 1, Nz + 1))
        self.Exz = np.zeros((Nx, Ny + 1, Nz + 1))
        self.Eyx = np.zeros((Nx + 1, Ny, Nz + 1))
        self.Eyz = np.zeros((Nx + 1, Ny, Nz + 1))
        self.Ezx = np.zeros((Nx + 1, Ny + 1, Nz))
        self.Ezy = np.zeros((Nx + 1, Ny + 1, Nz))
        self.Hx = np.zeros((Nx, Ny, Nz))
        self.Hy = np.zeros((Nx, Ny, Nz))
        self.Hz = np.zeros((Nx, Ny, Nz))
        self.Hzx = np.zeros((Nx, Ny, Nz))
        self.Hzy = np.zeros((Nx, Ny, Nz))
        self.Hxy = np.zeros((Nx, Ny, Nz))
        self.Hxz = np.zeros((Nx, Ny, Nz))
        self.Hyx = np.zeros((Nx, Ny, Nz))
        self.Hyz = np.zeros((Nx, Ny, Nz))
        self.Jx = np.zeros((self.Nx, self.Ny + 1, self.Nz + 1))
        self.Jy = np.zeros((self.Nx + 1, self.Ny, self.Nz + 1))
        self.Jz = np.zeros((self.Nx + 1, self.Ny + 1, self.Nz))

        # the sigmas must be assembled by the solver
        self.sigma_x = np.zeros_like(self.Ex)
        self.sigma_y = np.zeros_like(self.Ey)
        self.sigma_z = np.zeros_like(self.Ez)
        self.sigma_star_x = np.zeros_like(self.Hx)
        self.sigma_star_y = np.zeros_like(self.Hy)
        self.sigma_star_z = np.zeros_like(self.Hz)

        # we assemble these after the sigmas
        self.Ax = np.zeros_like(self.sigma_x)
        self.Ay = np.zeros_like(self.sigma_y)
        self.Az = np.zeros_like(self.sigma_z)
        self.Bx = np.zeros_like(self.sigma_x)
        self.By = np.zeros_like(self.sigma_y)
        self.Bz = np.zeros_like(self.sigma_z)
        self.Cx = np.zeros_like(self.sigma_star_x)
        self.Cy = np.zeros_like(self.sigma_star_y)
        self.Cz = np.zeros_like(self.sigma_star_z)
        self.Dx = np.zeros_like(self.sigma_star_x)
        self.Dy = np.zeros_like(self.sigma_star_y)
        self.Dz = np.zeros_like(self.sigma_star_z)

        self.lx_block = lx_block
        self.ly_block = ly_block
        self.lz_block = lz_block
        self.rx_block = rx_block
        self.ry_block = ry_block
        self.rz_block = rz_block

        self.C1 = self.dt / (self.dx * mu_0)
        self.C2 = self.dt / (self.dy * mu_0)
        self.C4 = self.dt / (self.dy * eps_0)
        self.C5 = self.dt / (self.dx * eps_0)
        self.C3 = self.dt / eps_0
        self.C6 = self.dt / eps_0

    def assemble_coeffs(self):
        self.Ax = (2 * eps_0 - self.dt * self.sigma_x) / (2 * eps_0 + self.dt * self.sigma_x)
        self.Ay = (2 * eps_0 - self.dt * self.sigma_y) / (2 * eps_0 + self.dt * self.sigma_y)
        self.Az = (2 * eps_0 - self.dt * self.sigma_z) / (2 * eps_0 + self.dt * self.sigma_z)
        self.Bx = 2 * self.dt / (2 * eps_0 + self.dt * self.sigma_x)
        self.By = 2 * self.dt / (2 * eps_0 + self.dt * self.sigma_y)
        self.Bz = 2 * self.dt / (2 * eps_0 + self.dt * self.sigma_z)
        self.Cx = (2 * mu_0 - self.dt * self.sigma_star_x) / (2 * mu_0 + self.dt * self.sigma_star_x)
        self.Cy = (2 * mu_0 - self.dt * self.sigma_star_y) / (2 * mu_0 + self.dt * self.sigma_star_y)
        self.Cz = (2 * mu_0 - self.dt * self.sigma_star_z) / (2 * mu_0 + self.dt * self.sigma_star_z)
        self.Dx = 2 * self.dt / (2 * mu_0 + self.dt * self.sigma_star_x)
        self.Dy = 2 * self.dt / (2 * mu_0 + self.dt * self.sigma_star_y)
        self.Dz = 2 * self.dt / (2 * mu_0 + self.dt * self.sigma_star_z)

    def advance_h_fdtd(self):
        Hz = self.Hz
        Ex = self.Ex
        Ey = self.Ey

        for ii in range(self.Nx):
            for jj in range(self.Ny):
                for kk in range(self.Nz):
                    self.Hzx[ii, jj, kk] = self.Cx[ii, jj, kk] * self.Hzx[ii, jj, kk] - self.Dx[ii, jj, kk] / self.dx * (self.Ey[ii + 1, jj, kk]
                                                                                                        - self.Ey[ii, jj, kk])
                    self.Hzy[ii, jj, kk] = self.Cy[ii, jj, kk] * self.Hzy[ii, jj, kk] + self.Dy[ii, jj, kk] / self.dy * (self.Ex[ii, jj + 1, kk]
                                                                                                        - self.Ex[ii, jj, kk])

        for ii in range(self.Nx):
            for jj in range(self.Ny):
                for kk in range(self.Nz):
                    self.Hxy[ii, jj, kk] = self.Cy[ii, jj, kk] * self.Hxy[ii, jj, kk] - self.Dy[ii, jj, kk] / self.dy * (self.Ez[ii, jj + 1, kk]
                                                                                                        - self.Ez[ii, jj, kk])
                    self.Hxz[ii, jj, kk] = self.Cz[ii, jj, kk] * self.Hxz[ii, jj, kk] + self.Dz[ii, jj, kk] / self.dz * (self.Ey[ii, jj, kk + 1]
                                                                                                        - self.Ey[ii, jj, kk])

        for ii in range(self.Nx):
            for jj in range(self.Ny):
                for kk in range(self.Nz):
                    self.Hyx[ii, jj, kk] = self.Cx[ii, jj, kk] * self.Hyx[ii, jj, kk] + self.Dx[ii, jj, kk] / self.dx * (self.Ez[ii + 1, jj, kk]
                                                                                                        - self.Ez[ii, jj, kk])
                    self.Hyz[ii, jj, kk] = self.Cz[ii, jj, kk] * self.Hyz[ii, jj, kk] - self.Dz[ii, jj, kk] / self.dz * (self.Ex[ii, jj, kk + 1]
                                                                                                        - self.Ex[ii, jj, kk])

        self.Hx = self.Hxy + self.Hxz
        self.Hy = self.Hyx + self.Hyx
        self.Hz = self.Hzx + self.Hzy

    def advance_e_fdtd(self):
        Hz = self.Hz
        Ex = self.Ex
        Ey = self.Ey

        for ii in range(self.Nx):
            for jj in range(1, self.Ny):
                for kk in range(self.Nz):
                    self.Exy[ii, jj, kk] = self.Ay[ii, jj, kk] * self.Exy[ii, jj, kk] + self.By[ii, jj, kk] / self.dy * (
                                           self.Hz[ii, jj, kk] - self.Hz[ii, jj - 1, kk])

        for ii in range(self.Nx):
            for jj in range(self.Ny):
                for kk in range(1, self.Nz):
                    self.Exz[ii, jj, kk] = self.Az[ii, jj, kk] * self.Exz[ii, jj, kk] - self.Bz[ii, jj, kk] / self.dz * (
                                           self.Hy[ii, jj, kk] - self.Hy[ii, jj, kk - 1])

        for ii in range(1, self.Nx):
            for jj in range(self.Ny):
                for kk in range(self.Nz):
                    self.Eyx[ii, jj, kk] = self.Ax[ii, jj, kk] * self.Eyx[ii, jj, kk] - self.Bx[ii, jj, kk] / self.dx * (
                                           self.Hz[ii, jj, kk] - self.Hz[ii - 1, jj, kk])

        for ii in range(self.Nx):
            for jj in range(self.Ny):
                for kk in range(1, self.Nz):
                    self.Eyz[ii, jj, kk] = self.Az[ii, jj, kk] * self.Eyz[ii, jj, kk] + self.Bz[ii, jj, kk] / self.dz * (
                                           self.Hx[ii, jj, kk] - self.Hx[ii, jj, kk - 1])

        for ii in range(1, self.Nx):
            for jj in range(self.Ny):
                for kk in range(self.Nz):
                    self.Ezx[ii, jj, kk] = self.Ax[ii, jj, kk]*self.Ezx[ii, jj, kk] + self.Bx[ii, jj, kk] / self.dx * (
                                           self.Hy[ii, jj, kk] - self.Hy[ii - 1, jj, kk])

        for ii in range(self.Nx):
            for jj in range(self.Ny):
                for kk in range(1, self.Nz):
                    self.Ezy[ii, jj, kk] = self.Ay[ii, jj, kk]*self.Ezy[ii, jj, kk] - self.By[ii, jj, kk] / self.dy * (
                                           self.Hx[ii, jj, kk] - self.Hx[ii, jj - 1, kk])

        self.Ex = self.Exy + self.Exz
        self.Ey = self.Eyx + self.Eyz
        self.Ez = self.Ezx + self.Ezy

    def update_e_boundary(self):

        if self.lz_block is not None:
            for ii in range(self.Nx):
                for jj in range(self.Ny):
                    self.Eyz[ii, jj, 0] = self.Az[ii, jj, 0] * self.Eyz[ii, jj, 0] + self.Bz[ii, jj, 0] / self.dz * (
                                           self.Hx[ii, jj, 0] - self.lz_block.Hx[ii, jj, - 1])

        if self.rz_block is not None:
            for ii in range(self.Nx):
                for jj in range(self.Ny):
                    self.Eyz[ii, jj, -1] = self.Az[ii, jj, -1] * self.Eyz[ii, jj, -1] + self.Bz[ii, jj, -1] / self.dz * (
                                           self.rz_block.Hx[ii, jj, -1] - self.Hx[ii, jj, -1])

        if self.lx_block is not None:
            for jj in range(self.Ny):
                for kk in range(self.Nz):
                    self.Eyx[0, jj, kk] = self.Ax[0, jj] * self.Eyx[0, jj, kk] - self.Bx[0, jj, kk] / self.dx * (
                                    self.Hz[0, jj, kk] - self.lx_block.Hz[-1, jj, kk])

        if self.rx_block is not None:
            for jj in range(self.Ny):
                for kk in range(self.Nz):
                    self.Eyx[-1, jj, kk] = self.Ax[-1, jj, kk] * self.Eyx[-1, jj, kk] - self.Bx[-1, jj, kk] / self.dx * (
                                        self.rx_block.Hz[0, jj, kk] - self.Hz[-1, jj, kk])

        if self.ly_block is not None:
            for ii in range(self.Nx):
                for kk in range(self.Nz):
                    self.Exy[ii, 0, kk] = self.Ay[ii, 0, kk] * self.Exy[ii, 0, kk] + self.By[ii, 0, kk] / self.dy * (
                                         self.Hz[ii, 0, kk] - self.ly_block.Hz[ii, -1, kk])

        if self.ry_block is not None:
            for ii in range(self.Nx):
                for kk in range(self.Nz):
                    self.Exy[ii, -1, kk] = self.Ay[ii, -1, kk] * self.Exy[ii, -1, kk] + self.By[ii, -1, kk] / self.dy * (
                                        self.ry_block.Hz[ii, 0, kk] - self.Hz[ii, - 1])

        if self.lz_block is not None:
            for ii in range(self.Nx):
                for jj in range(self.Ny):
                    self.Exz[ii, jj, 0] = self.Az[ii, jj, 0] * self.Exz[ii, jj, 0] - self.Bz[ii, jj, 0] / self.dz * (
                                           self.Hy[ii, jj, 0] - self.lz_block.Hy[ii, jj, - 1])

        if self.rz_block is not None:
            for ii in range(self.Nx):
                for jj in range(self.Ny):
                    self.Exz[ii, jj, -1] = self.Az[ii, jj, -1] * self.Exz[ii, jj, -1] - self.Bz[ii, jj, -1] / self.dz * (
                                           self.rz_block.Hy[ii, jj, 0] - self.Hy[ii, jj, - 1])

        if self.lx_block is not None:
            for jj in range(self.Ny):
                for kk in range(self.Nz):
                    self.Ezx[0, jj, kk] = self.Ax[0, jj, kk]*self.Ezx[0, jj, kk] + self.Bx[0, jj, kk] / self.dx * (
                                           self.Hy[0, jj, kk] - self.lx_block.Hy[- 1, jj, kk])

        if self.rx_block is not None:
            for jj in range(self.Ny):
                for kk in range(self.Nz):
                    self.Ezx[-1, jj, kk] = self.Ax[-1, jj, kk]*self.Ezx[-1, jj, kk] + self.Bx[-1, jj, kk] / self.dx * (
                                           self.rx_block.Hy[0, jj, kk] - self.Hy[- 1, jj, kk])

        if self.ly_block is not None:
            for ii in range(self.Nx):
                for kk in range(self.Ny):
                    self.Ezy[ii, 0, kk] = self.Ay[ii, 0, kk]*self.Ezy[ii, 0, kk] - self.By[ii, 0, kk] / self.dy * (
                                           self.Hx[ii, 0, kk] - self.ly_block.Hx[ii, -1, kk])

        if self.ry_block is not None:
            for ii in range(self.Nx):
                for kk in range(self.Ny):
                    self.Ezy[ii, -1, kk] = self.Ay[ii, -1, kk]*self.Ezy[ii, -1, kk] - self.By[ii, -1, kk] / self.dy * (
                                           self.ry_block.Hx[ii, -1, kk] - self.Hx[ii, 0, kk])

        self.Ex = self.Exy + self.Exz
        self.Ey = self.Eyx + self.Eyz
        self.Ez = self.Ezx + self.Ezy