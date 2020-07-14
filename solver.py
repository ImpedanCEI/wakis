import numpy as np
from scipy.constants import c as c_light, epsilon_0 as eps_0, mu_0 as mu_0


def eq(a, b, tol=1e-8):
    return abs(a - b) < tol


def neq(a, b, tol=1e-8):
    return not eq(a, b, tol)


class EMSolver2D:
    def __init__(self, grid, sol_type, cfln, i_s, j_s):
        self.grid = grid
        self.type = type
        self.cfln = cfln

        self.dt = cfln / (c_light * np.sqrt(1 / self.grid.dx ** 2 + 1 / self.grid.dy ** 2))
        self.dx = self.grid.dx
        self.dy = self.grid.dy
        self.Nx = self.grid.nx
        self.Ny = self.grid.ny
        self.sol_type = sol_type

        self.Ex = np.zeros((self.Nx + 1, self.Ny + 1))
        self.Ey = np.zeros((self.Nx + 1, self.Ny + 1))
        self.Hz = np.zeros((self.Nx + 1, self.Ny + 1))
        self.Jx = np.zeros((self.Nx + 1, self.Ny + 1))
        self.Jy = np.zeros((self.Nx + 1, self.Ny + 1))
        self.rho = np.zeros((self.Nx, self.Ny))
        if sol_type == 'DM' or sol_type == 'ECT' or sol_type == 'ECT_one_cell' or sol_type == 'ECT_one_cell_rho' or sol_type == 'ECT_rho':
            self.Vxy = np.zeros((self.Nx + 1, self.Ny + 1))
        if sol_type == 'ECT' or sol_type == 'ECT_one_cell' or sol_type == 'ECT_one_cell_rho' or sol_type == 'ECT_rho':
            self.V_enl = np.zeros((self.Nx + 1, self.Ny + 1))

        if sol_type == 'ECT' or sol_type == 'ECT_one_cell' or sol_type == 'ECT_one_cell_rho' or sol_type == 'ECT_rho' or sol_type == 'DM':
            self.C1 = self.dt / mu_0
            self.C4 = self.dt / (eps_0 * self.dy)
            self.C5 = self.dt / (eps_0 * self.dx)
            self.C3 = self.dt / eps_0
            self.C6 = self.dt / eps_0

        if sol_type == 'FDTD':
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

    def gauss(self, t):
        tau = 20 * self.dt
        if t < 6 * tau:
            return 100 * np.exp(-(t - 3 * tau) ** 2 / tau ** 2)
        else:
            return 0.

    def one_step(self):
        if self.sol_type == 'ECT':
            self.one_step_ect()
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
        for ii in range(self.Nx):
            for jj in range(self.Ny):
                if self.grid.flag_int_cell[ii, jj]:
                    Hz[ii, jj] = (Hz[ii, jj] - self.C1 * (Ey[ii + 1, jj] - Ey[ii, jj]) + self.C2 * (Ex[ii, jj + 1]
                                                                                                    - Ex[ii, jj]))

                    if self.grid.l_x[ii, jj] > 0:
                        Ex[ii, jj] = Ex[ii, jj] - self.C3 * self.Jx[ii, jj] + self.C4 * (Hz[ii, jj] - Hz[ii, jj - 1])
                    if self.grid.l_y[ii, jj] > 0:
                        Ey[ii, jj] = Ey[ii, jj] - self.C3 * self.Jy[ii, jj] - self.C5 * (Hz[ii, jj] - Hz[ii - 1, jj])

        self.time += self.dt

    def one_step_dm(self):
        self.compute_v_and_rho()

        for i in range(self.Nx):
            for j in range(self.Ny):
                if self.grid.flag_int_cell[i, j]:
                    self.Hz[i, j] = self.Hz[i, j] - self.dt / (mu_0 * self.grid.S[i, j]) * self.Vxy[i, j]

        self.advance_e_dm()

        self.time += self.dt

    def one_step_ect(self):
        # Compute cell voltages
        self.compute_v_and_rho()

        # take care of unstable cells
        for ii in range(self.Nx):
            for jj in range(self.Ny):
                if self.grid.flag_int_cell[ii, jj] and self.grid.flag_unst_cell[ii, jj]:
                    for (ip, jp, patch, _) in self.grid.borrowing[ii, jj]:
                        self.V_enl[ii, jj] = self.rho[ii, jj] * self.grid.S[ii, jj] + self.rho[ip, jp] * patch
                        rho_enl = self.V_enl[ii, jj] / self.grid.S_enl[ii, jj]
                        for num, (iii, jjj, _, _) in enumerate(self.grid.lending[ip, jp]):
                            if iii == ii and jjj == jj:
                                self.grid.lending[ip, jp][num][3] = rho_enl
                        # V_new = rho_enl* self.grid.S[ii, jj]
                        self.Hz[ii, jj] = self.Hz[ii, jj] - self.C1 * rho_enl

        # take care of regular cells
        for ii in range(self.Nx):
            for jj in range(self.Ny):
                if self.grid.flag_int_cell[ii, jj] and not self.grid.flag_unst_cell[ii, jj]:
                    # regular cell which hasn't been intruded
                    if len(self.grid.lending[ii, jj]) == 0:
                        self.Hz[ii, jj] = self.Hz[ii, jj] - self.C1 * self.rho[ii, jj]
                    # regular cell which has been intruded
                    elif len(self.grid.lending[ii, jj]) != 0:
                        Vnew = 0
                        red_area = self.grid.S[ii, jj]
                        for (ip, jp, patch, rho_enl) in self.grid.lending[ii, jj]:
                            if rho_enl is None:
                                print('big mistake')

                            red_area -= patch
                            Vnew += rho_enl * patch

                        Vnew += self.rho[ii, jj] * red_area
                        self.Hz[ii, jj] = self.Hz[ii, jj] - self.C1 * Vnew / self.grid.S[ii, jj]

        self.advance_e_dm()

        self.time += self.dt

    def compute_v_and_rho(self):
        l_y = self.grid.l_y
        l_x = self.grid.l_x
        for ii in range(self.Nx):
            for jj in range(self.Ny):
                if self.grid.flag_int_cell[ii, jj]:
                    self.Vxy[ii, jj] = (self.Ey[ii + 1, jj] * l_y[ii + 1, jj] - self.Ey[ii, jj] * l_y[ii, jj]
                                        - self.Ex[ii, jj + 1] * l_x[ii, jj + 1] + self.Ex[ii, jj] * l_x[ii, jj])
                    if self.sol_type != 'DM':
                        self.rho[ii, jj] = self.Vxy[ii, jj] / self.grid.S[ii, jj]

    def advance_e_dm(self):
        for ii in range(self.Nx):
            for jj in range(self.Ny):
                if self.grid.flag_int_cell[ii, jj]:
                    if self.grid.l_x[ii, jj] > 0:
                        self.Ex[ii, jj] = self.Ex[ii, jj] + self.dt / (eps_0 * self.dy) * (
                                self.Hz[ii, jj] - self.Hz[ii, jj - 1]) - self.C3*self.Jx[ii, jj]
                    if self.grid.l_y[ii, jj] > 0:
                        self.Ey[ii, jj] = self.Ey[ii, jj] - self.dt / (eps_0 * self.dx) * (
                                self.Hz[ii, jj] - self.Hz[ii - 1, jj]) - self.C3*self.Jy[ii, jj]
