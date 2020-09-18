import numpy as np
from scipy.constants import c as c_light, epsilon_0 as eps_0, mu_0 as mu_0


def eq(a, b, tol=1e-8):
    return abs(a - b) < tol


def neq(a, b, tol=1e-8):
    return not eq(a, b, tol)


class EMSolver2D:
    def __init__(self, grid, sol_type, cfln, i_s, j_s, bc_low, bc_high,
                 N_pml_low=None, N_pml_high=None):
        self.grid = grid
        self.type = type
        self.cfln = cfln

        self.dt = cfln / (c_light * np.sqrt(1 / self.grid.dx ** 2 + 1 / self.grid.dy ** 2))
        self.dx = self.grid.dx
        self.dy = self.grid.dy
        self.Nx = self.grid.nx
        self.Ny = self.grid.ny
        self.sol_type = sol_type

        self.N_pml_low = np.zeros(2, dtype=int)
        self.N_pml_high = np.zeros(2, dtype=int)

        if bc_low[0] == 'pml':
            self.N_pml_low[0] = 10 if N_pml_low is None else N_pml_low[0]
        if bc_low[1] == 'pml':
            self.N_pml_low[1] = 10 if N_pml_low is None else N_pml_low[1]
        if bc_high[0] == 'pml':
            self.N_pml_high[0] = 10 if N_pml_high is None else N_pml_high[0]
        if bc_high[1] == 'pml':
            self.N_pml_high[1] = 10 if N_pml_high is None else N_pml_high[1]

        self.N_tot_x = self.Nx + self.N_pml_low[0] + self.N_pml_high[0]
        self.N_tot_y = self.Ny + self.N_pml_low[0] + self.N_pml_high[0]
        self.sol_type = sol_type

        self.Ex = np.zeros((self.N_tot_x + 1, self.N_tot_y + 1))
        self.Ey = np.zeros((self.N_tot_x + 1, self.N_tot_y + 1))
        self.Hz = np.zeros((self.N_tot_x + 1, self.N_tot_y + 1))
        self.Exy = np.zeros((self.N_tot_x + 1, self.N_tot_y + 1))
        self.Exz = np.zeros((self.N_tot_x + 1, self.N_tot_y + 1))
        self.Eyx = np.zeros((self.N_tot_x + 1, self.N_tot_y + 1))
        self.Eyz = np.zeros((self.N_tot_x + 1, self.N_tot_y + 1))
        self.Hzx = np.zeros((self.N_tot_x + 1, self.N_tot_y + 1))
        self.Hzy = np.zeros((self.N_tot_x + 1, self.N_tot_y + 1))

        self.Jx = np.zeros((self.N_tot_x + 1, self.N_tot_y + 1))
        self.Jy = np.zeros((self.N_tot_x + 1, self.N_tot_y + 1))
        self.rho = np.zeros((self.Nx, self.Ny))

        self.sigma_x = np.zeros_like(self.Ex)
        self.sigma_y = np.zeros_like(self.Ex)
        self.sigma_z = np.zeros_like(self.Ex)
        self.sigma_star_x = np.zeros_like(self.Ex)
        self.sigma_star_y = np.zeros_like(self.Ex)
        self.sigma_star_z = np.zeros_like(self.Ex)

        self.alpha_pml = 3
        self.R0_pml = 0.001

        self.assemble_conductivities()

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


        if (sol_type is not 'FDTD') and (sol_type is not 'DM') and (sol_type is not 'ECT'):
            raise ValueError("sol_type must be:\n" +
                             "\t'FDTD' for standard staircased FDTD\n" +
                             "\t'DM' for Dey-Mittra conformal FDTD\n" +
                             "\t'ECT' for Enlarged Cell Technique conformal FDTD")

        if sol_type is 'DM' or sol_type is 'ECT':
            self.Vxy = np.zeros((self.Nx + 1, self.Ny + 1))
        if sol_type is 'ECT':
            self.V_enl = np.zeros((self.Nx + 1, self.Ny + 1))

        if sol_type is 'ECT' or sol_type is 'DM':
            self.C1 = self.dt / mu_0
            self.C4 = self.dt / (eps_0 * self.dy)
            self.C5 = self.dt / (eps_0 * self.dx)
            self.C3 = self.dt / eps_0
            self.C6 = self.dt / eps_0

        if sol_type is 'FDTD':
            Z_0 = np.sqrt(mu_0 / eps_0)

            self.C1 = self.dt / (self.dx * mu_0)
            self.C2 = self.dt / (self.dy * mu_0)
            self.C4 = self.dt / (self.dy * eps_0)
            self.C5 = self.dt / (self.dx * eps_0)
            self.C3 = self.dt / eps_0
            self.C6 = self.dt / eps_0

        # indices for the source
        self.i_s = i_s
        self.j_s = j_s

        self.time = 0

    def assemble_conductivities(self):

        sigma_m_low_x = -(self.alpha_pml + 1) * eps_0 * c_light / (2 * self.N_pml_low[0]) * np.log(self.R0_pml)
        sigma_m_low_y = -(self.alpha_pml + 1) * eps_0 * c_light / (2 * self.N_pml_low[1]) * np.log(self.R0_pml)
        sigma_m_high_x = -(self.alpha_pml + 1) * eps_0 * c_light / (2 * self.N_pml_high[0]) * np.log(self.R0_pml)
        sigma_m_high_y = -(self.alpha_pml + 1) * eps_0 * c_light / (2 * self.N_pml_high[1]) * np.log(self.R0_pml)

        for n in range(self.N_pml_low[0]):
            self.sigma_x[n, :] = sigma_m_low_x * ((self.N_pml_low[0] - n) / self.N_pml_low[0]) ** self.alpha_pml
            self.sigma_x[self.N_tot_x - n, :] = sigma_m_high_x * (
                        (self.N_pml_high[0] - n) / self.N_pml_high[0]) ** self.alpha_pml
            self.sigma_y[:, n] = sigma_m_low_y * ((self.N_pml_low[1] - n) / self.N_pml_low[1]) ** self.alpha_pml
            self.sigma_y[:, self.N_tot_y - n] = sigma_m_high_y * (
                        (self.N_pml_high[1] - n) / self.N_pml_high[1]) ** self.alpha_pml

        self.sigma_star_x = self.sigma_x * mu_0 / eps_0
        self.sigma_star_y = self.sigma_y * mu_0 / eps_0

    def gauss(self, t):
        tau = 20 * self.dt
        if t < 6 * tau:
            return 100 * np.exp(-(t - 3 * tau) ** 2 / tau ** 2)
        else:
            return 0.

    def one_step(self):
        if self.sol_type == 'ECT':
            self.compute_v_and_rho()
            self.one_step_ect(Nx=self.Nx, Ny=self.Ny, V_enl=self.V_enl,
                              rho=self.rho, Hz=self.Hz, C1=self.C1,
                              flag_int_cell=self.grid.flag_int_cell,
                              flag_unst_cell=self.grid.flag_unst_cell, S=self.grid.S,
                              borrowing=self.grid.borrowing, S_enl=self.grid.S_enl,
                              lending=self.grid.lending, S_red=self.grid.S_red)

            self.advance_e_dm()
            self.time += self.dt
        if self.sol_type == 'FDTD':
            self.one_step_fdtd()
        if self.sol_type == 'DM':
            self.one_step_dm()

    def one_step_fdtd(self):
        Z_0 = np.sqrt(mu_0 / eps_0)
        Ex = self.Ex
        Ey = self.Ey
        Hz = self.Hz
        # Compute cell voltages
        self.advance_h_fdtd()
        self.advance_h_pml_fdtd()
        self.advance_e_fdtd()
        self.advance_e_pml_fdtd()

        self.time += self.dt

    def advance_h_fdtd(self):
        Ex = self.Ex
        Ey = self.Ey
        Hz = self.Hz
        for ii in range(self.Nx):
            for jj in range(self.Ny):
                if self.grid.flag_int_cell[ii-self.N_pml_low[0], jj-self.N_pml_low[1]]:
                    Hz[ii, jj] = (Hz[ii, jj] - self.C1 * (Ey[ii + 1, jj] - Ey[ii, jj]) + self.C2 * (
                            Ex[ii, jj + 1]
                            - Ex[ii, jj]))

    def advance_h_pml_fdtd(self):

        # 00
        for ii in range(self.N_pml_low[0]):
            for jj in range(self.N_pml_low[1]):
                self.Hzx[ii, jj] = self.Cx[ii, jj] * self.Hzx[ii, jj] + self.Dx[ii, jj] / self.dx * (self.Ey[ii + 1, jj]
                                                                                                     - self.Ey[ii, jj])
                self.Hzy[ii, jj] = self.Cy[ii, jj] * self.Hzy[ii, jj] - self.Dy[ii, jj] / self.dy * (self.Ex[ii, jj + 1]
                                                                                                     - self.Ex[ii, jj])
        # 10
        for ii in range(self.N_pml_low[0], self.N_pml_low[0] + self.Nx):
            for jj in range(self.N_pml_low[1]):
                self.Hzx[ii, jj] = self.Cx[ii, jj] * self.Hzx[ii, jj] + self.Dx[ii, jj] / self.dx * (self.Ey[ii + 1, jj]
                                                                                                     - self.Ey[ii, jj])
                self.Hzy[ii, jj] = self.Cy[ii, jj] * self.Hzy[ii, jj] - self.Dy[ii, jj] / self.dy * (self.Ex[ii, jj + 1]
                                                                                                     - self.Ex[ii, jj])

        # 01
        for ii in range(self.N_pml_low[0]):
            for jj in range(self.N_pml_low[1], self.N_pml_low[1] + self.Ny):
                self.Hzx[ii, jj] = self.Cx[ii, jj] * self.Hzx[ii, jj] + self.Dx[ii, jj] / self.dx * (self.Ey[ii + 1, jj]
                                                                                                     - self.Ey[ii, jj])
                self.Hzy[ii, jj] = self.Cy[ii, jj] * self.Hzy[ii, jj] - self.Dy[ii, jj] / self.dy * (self.Ex[ii, jj + 1]
                                                                                                     - self.Ex[ii, jj])

        # 20
        for ii in range(self.N_pml_low[0] + self.Nx, self.N_tot_x):
            for jj in range(self.N_pml_low[1]):
                self.Hzx[ii, jj] = self.Cx[ii, jj] * self.Hzx[ii, jj] + self.Dx[ii, jj] / self.dx * (self.Ey[ii + 1, jj]
                                                                                                     - self.Ey[ii, jj])
                self.Hzy[ii, jj] = self.Cy[ii, jj] * self.Hzy[ii, jj] - self.Dy[ii, jj] / self.dy * (self.Ex[ii, jj + 1]
                                                                                                     - self.Ex[ii, jj])

        # 02
        for ii in range(self.N_pml_low[0]):
            for jj in range(self.N_pml_low[1] + self.Ny, self.N_tot_y):
                self.Hzx[ii, jj] = self.Cx[ii, jj] * self.Hzx[ii, jj] + self.Dx[ii, jj] / self.dx * (self.Ey[ii + 1, jj]
                                                                                                     - self.Ey[ii, jj])
                self.Hzy[ii, jj] = self.Cy[ii, jj] * self.Hzy[ii, jj] - self.Dy[ii, jj] / self.dy * (self.Ex[ii, jj + 1]
                                                                                                     - self.Ex[ii, jj])

        # 21
        for ii in range(self.N_pml_low[0] + self.Nx, self.N_tot_x):
            for jj in range(self.N_pml_low[1], self.N_pml_low[1] + self.Ny):
                self.Hzx[ii, jj] = self.Cx[ii, jj] * self.Hzx[ii, jj] + self.Dx[ii, jj] / self.dx * (self.Ey[ii + 1, jj]
                                                                                                     - self.Ey[ii, jj])
                self.Hzy[ii, jj] = self.Cy[ii, jj] * self.Hzy[ii, jj] - self.Dy[ii, jj] / self.dy * (self.Ex[ii, jj + 1]
                                                                                                     - self.Ex[ii, jj])

        # 21
        for ii in range(self.N_pml_low[0], self.N_pml_low[0] + self.Nx):
            for jj in range(self.N_pml_low[1] + self.Ny, self.N_tot_y):
                self.Hzx[ii, jj] = self.Cx[ii, jj] * self.Hzx[ii, jj] + self.Dx[ii, jj] / self.dx * (self.Ey[ii + 1, jj]
                                                                                                     - self.Ey[ii, jj])
                self.Hzy[ii, jj] = self.Cy[ii, jj] * self.Hzy[ii, jj] - self.Dy[ii, jj] / self.dy * (self.Ex[ii, jj + 1]
                                                                                                     - self.Ex[ii, jj])

        # 22
        for ii in range(self.N_pml_low[0] + self.Nx, self.N_tot_x):
            for jj in range(self.N_pml_low[1] + self.Ny, self.N_tot_y):
                self.Hzx[ii, jj] = self.Cx[ii, jj] * self.Hzx[ii, jj] + self.Dx[ii, jj] / self.dx * (self.Ey[ii + 1, jj]
                                                                                                     - self.Ey[ii, jj])
                self.Hzy[ii, jj] = self.Cy[ii, jj] * self.Hzy[ii, jj] - self.Dy[ii, jj] / self.dy * (self.Ex[ii, jj + 1]
                                                                                                     - self.Ex[ii, jj])

        self.Hz = self.Hz + self.Hzy + self.Hzx

    def advance_e_pml_fdtd(self):

        # 00
        for ii in range(self.N_pml_low[0]):
            for jj in range(self.N_pml_low[1]):
                self.Exy[ii, jj] = self.Ay[ii, jj] * self.Exy[ii, jj] - self.By[ii, jj] / self.dy * (self.Hz[ii, jj]
                                                                                                     - self.Hz[
                                                                                                         ii, jj - 1])
                self.Exz[ii, jj] = self.Az[ii, jj] * self.Exz[ii, jj]
                self.Eyz[ii, jj] = self.Az[ii, jj] * self.Eyz[ii, jj]
                self.Eyx[ii, jj] = self.Ax[ii, jj] * self.Eyx[ii, jj] + self.Bx[ii, jj] / self.dx * (self.Hz[ii, jj]
                                                                                                     - self.Hz[
                                                                                                         ii - 1, jj])

        # 10
        for ii in range(self.N_pml_low[0], self.N_pml_low[0] + self.Nx):
            for jj in range(self.N_pml_low[1]):
                self.Exy[ii, jj] = self.Ay[ii, jj] * self.Exy[ii, jj] - self.By[ii, jj] / self.dy * (self.Hz[ii, jj]
                                                                                                     - self.Hz[
                                                                                                         ii, jj - 1])
                self.Exz[ii, jj] = self.Az[ii, jj] * self.Exz[ii, jj]
                self.Eyz[ii, jj] = self.Az[ii, jj] * self.Eyz[ii, jj]
                self.Eyx[ii, jj] = self.Ax[ii, jj] * self.Eyx[ii, jj] + self.Bx[ii, jj] / self.dx * (self.Hz[ii, jj]
                                                                                                     - self.Hz[
                                                                                                         ii - 1, jj])

        # 01
        for ii in range(self.N_pml_low[0]):
            for jj in range(self.N_pml_low[1], self.N_pml_low[1] + self.Ny):
                self.Exy[ii, jj] = self.Ay[ii, jj] * self.Exy[ii, jj] - self.By[ii, jj] / self.dy * (self.Hz[ii, jj]
                                                                                                     - self.Hz[
                                                                                                         ii, jj - 1])
                self.Exz[ii, jj] = self.Az[ii, jj] * self.Exz[ii, jj]
                self.Eyz[ii, jj] = self.Az[ii, jj] * self.Eyz[ii, jj]
                self.Eyx[ii, jj] = self.Ax[ii, jj] * self.Eyx[ii, jj] + self.Bx[ii, jj] / self.dx * (self.Hz[ii, jj]
                                                                                                     - self.Hz[
                                                                                                         ii - 1, jj])

        # 20
        for ii in range(self.N_pml_low[0] + self.Nx, self.N_tot_x):
            for jj in range(self.N_pml_low[1]):
                self.Exy[ii, jj] = self.Ay[ii, jj] * self.Exy[ii, jj] - self.By[ii, jj] / self.dy * (self.Hz[ii, jj]
                                                                                                     - self.Hz[
                                                                                                         ii, jj - 1])
                self.Exz[ii, jj] = self.Az[ii, jj] * self.Exz[ii, jj]
                self.Eyz[ii, jj] = self.Az[ii, jj] * self.Eyz[ii, jj]
                self.Eyx[ii, jj] = self.Ax[ii, jj] * self.Eyx[ii, jj] + self.Bx[ii, jj] / self.dx * (self.Hz[ii, jj]
                                                                                                     - self.Hz[
                                                                                                         ii - 1, jj])

        # 02
        for ii in range(self.N_pml_low[0]):
            for jj in range(self.N_pml_low[1] + self.Ny, self.N_tot_y):
                self.Exy[ii, jj] = self.Ay[ii, jj] * self.Exy[ii, jj] - self.By[ii, jj] / self.dy * (self.Hz[ii, jj]
                                                                                                     - self.Hz[
                                                                                                         ii, jj - 1])
                self.Exz[ii, jj] = self.Az[ii, jj] * self.Exz[ii, jj]
                self.Eyz[ii, jj] = self.Az[ii, jj] * self.Eyz[ii, jj]
                self.Eyx[ii, jj] = self.Ax[ii, jj] * self.Eyx[ii, jj] + self.Bx[ii, jj] / self.dx * (self.Hz[ii, jj]
                                                                                                     - self.Hz[
                                                                                                         ii - 1, jj])

        # 21
        for ii in range(self.N_pml_low[0] + self.Nx, self.N_tot_x):
            for jj in range(self.N_pml_low[1], self.N_pml_low[1] + self.Ny):
                self.Exy[ii, jj] = self.Ay[ii, jj] * self.Exy[ii, jj] - self.By[ii, jj] / self.dy * (self.Hz[ii, jj]
                                                                                                     - self.Hz[
                                                                                                         ii, jj - 1])
                self.Exz[ii, jj] = self.Az[ii, jj] * self.Exz[ii, jj]
                self.Eyz[ii, jj] = self.Az[ii, jj] * self.Eyz[ii, jj]
                self.Eyx[ii, jj] = self.Ax[ii, jj] * self.Eyx[ii, jj] + self.Bx[ii, jj] / self.dx * (self.Hz[ii, jj]
                                                                                                     - self.Hz[
                                                                                                         ii - 1, jj])

        # 21
        for ii in range(self.N_pml_low[0], self.N_pml_low[0] + self.Nx):
            for jj in range(self.N_pml_low[1] + self.Ny, self.N_tot_y):
                self.Exy[ii, jj] = self.Ay[ii, jj] * self.Exy[ii, jj] - self.By[ii, jj] / self.dy * (self.Hz[ii, jj]
                                                                                                     - self.Hz[
                                                                                                         ii, jj - 1])
                self.Exz[ii, jj] = self.Az[ii, jj] * self.Exz[ii, jj]
                self.Eyz[ii, jj] = self.Az[ii, jj] * self.Eyz[ii, jj]
                self.Eyx[ii, jj] = self.Ax[ii, jj] * self.Eyx[ii, jj] + self.Bx[ii, jj] / self.dx * (self.Hz[ii, jj]
                                                                                                     - self.Hz[
                                                                                                         ii - 1, jj])

        # 22
        for ii in range(self.N_pml_low[0] + self.Nx, self.N_tot_x):
            for jj in range(self.N_pml_low[1] + self.Ny, self.N_tot_y):
                self.Exy[ii, jj] = self.Ay[ii, jj] * self.Exy[ii, jj] - self.By[ii, jj] / self.dy * (self.Hz[ii, jj]
                                                                                                     - self.Hz[
                                                                                                         ii, jj - 1])
                self.Exz[ii, jj] = self.Az[ii, jj] * self.Exz[ii, jj]
                self.Eyz[ii, jj] = self.Az[ii, jj] * self.Eyz[ii, jj]
                self.Eyx[ii, jj] = self.Ax[ii, jj] * self.Eyx[ii, jj] + self.Bx[ii, jj] / self.dx * (self.Hz[ii, jj]
                                                                                                     - self.Hz[
                                                                                                         ii - 1, jj])

        self.Ex = self.Ex + self.Exy + self.Exz
        self.Ey = self.Ey + self.Eyx + self.Eyz

    def advance_e_fdtd(self):
        Z_0 = np.sqrt(mu_0 / eps_0)
        Ex = self.Ex
        Ey = self.Ey
        Hz = self.Hz
        # Compute cell voltages
        for ii in range(self.N_pml_low[0], self.N_pml_low[0] + self.Nx):
            for jj in range(self.N_pml_low[1], self.N_pml_low[1] + self.Ny):
                if self.grid.flag_int_cell[ii - self.N_pml_low[0], jj - self.N_pml_low[1]]:
                    if self.grid.l_x[ii - self.N_pml_low[0], jj - self.N_pml_low[1]] > 0:
                        Ex[ii, jj] = Ex[ii, jj] - self.C3 * self.Jx[ii, jj] + self.C4 * (
                                Hz[ii, jj] - Hz[ii, jj - 1])
                    if self.grid.l_y[ii - self.N_pml_low[0], jj - self.N_pml_low[1]] > 0:
                        Ey[ii, jj] = Ey[ii, jj] - self.C3 * self.Jy[ii, jj] - self.C5 * (
                                Hz[ii, jj] - Hz[ii - 1, jj])

    def one_step_dm(self):
        self.compute_v_and_rho()

        for i in range(self.Nx):
            for j in range(self.Ny):
                if self.grid.flag_int_cell[i, j]:
                    self.Hz[i, j] = self.Hz[i, j] - self.dt / (mu_0 * self.grid.S[i, j]) * self.Vxy[
                        i, j]

        self.advance_e_dm()

        self.time += self.dt

    @staticmethod
    def one_step_ect(Nx=None, Ny=None, V_enl=None, rho=None, Hz=None, C1=None, flag_int_cell=None,
                     flag_unst_cell=None, S=None, borrowing=None, S_enl=None, lending=None,
                     S_red=None):
        # Compute cell voltages

        # take care of unstable cells
        for ii in range(Nx):
            for jj in range(Ny):
                if flag_int_cell[ii, jj] and flag_unst_cell[ii, jj]:
                    V_enl[ii, jj] = rho[ii, jj] * S[ii, jj]
                    if len(borrowing[ii, jj]) == 0:
                        print('error in one_step_ect')
                    for (ip, jp, patch, _) in borrowing[ii, jj]:
                        V_enl[ii, jj] += rho[ip, jp] * patch
                    rho_enl = V_enl[ii, jj] / S_enl[ii, jj]
                    # communicate to the intruded cell the intruding rho
                    for (ip, jp, patch, _) in borrowing[ii, jj]:
                        for num, (iii, jjj, _, _) in enumerate(lending[ip, jp]):
                            if iii == ii and jjj == jj:
                                lending[ip, jp][num][3] = rho_enl

                    Hz[ii, jj] = Hz[ii, jj] - C1 * rho_enl

        # take care of stable cells
        for ii in range(Nx):
            for jj in range(Ny):
                if flag_int_cell[ii, jj] and not flag_unst_cell[ii, jj]:
                    # stable cell which hasn't been intruded
                    if len(lending[ii, jj]) == 0:
                        Hz[ii, jj] = Hz[ii, jj] - C1 * rho[ii, jj]
                    # stable cell which has been intruded
                    elif len(lending[ii, jj]) != 0:
                        Vnew = 0
                        red_area = S[ii, jj]
                        for (ip, jp, patch, rho_enl) in lending[ii, jj]:
                            if rho_enl is None:
                                print('big mistake')

                            Vnew += rho_enl * patch

                        Vnew += rho[ii, jj] * S_red[ii, jj]
                        Hz[ii, jj] = Hz[ii, jj] - C1 * Vnew / S[ii, jj]

    def compute_v_and_rho(self):
        l_y = self.grid.l_y
        l_x = self.grid.l_x
        for ii in range(self.Nx):
            for jj in range(self.Ny):
                if self.grid.flag_int_cell[ii, jj]:
                    self.Vxy[ii, jj] = (
                            self.Ey[ii + 1, jj] * l_y[ii + 1, jj] - self.Ey[ii, jj] * l_y[
                        ii, jj]
                            - self.Ex[ii, jj + 1] * l_x[ii, jj + 1] + self.Ex[ii, jj] * l_x[
                                ii, jj])
                    if self.sol_type != 'DM':
                        self.rho[ii, jj] = self.Vxy[ii, jj] / self.grid.S[ii, jj]

    def advance_e_dm(self):
        for ii in range(self.Nx):
            for jj in range(self.Ny):
                if self.grid.l_x[ii, jj] > 0:
                    self.Ex[ii, jj] = self.Ex[ii, jj] + self.dt / (eps_0 * self.dy) * (
                            self.Hz[ii, jj] - self.Hz[ii, jj - 1]) - self.C3 * self.Jx[ii, jj]
                if self.grid.l_y[ii, jj] > 0:
                    self.Ey[ii, jj] = self.Ey[ii, jj] - self.dt / (eps_0 * self.dx) * (
                            self.Hz[ii, jj] - self.Hz[ii - 1, jj]) - self.C3 * self.Jy[ii, jj]
