import numpy as np
from scipy.constants import c as c_light, epsilon_0 as eps_0, mu_0 as mu_0

class PmlBlock2D:
    def __init__(self, Nx, Ny, dt, dx, dy):
        self.Nx = Nx
        self.Ny = Ny
        self.dt = dt
        self.dx = dx
        self.dy = dy
        
        self.Ex = np.zeros((Nx + 1, Ny + 1))
        self.Ey = np.zeros((Nx + 1, Ny + 1))
        self.Exy = np.zeros((Nx + 1, Ny + 1))
        self.Exz = np.zeros((Nx + 1, Ny + 1))
        self.Eyx = np.zeros((Nx + 1, Ny + 1))
        self.Eyz = np.zeros((Nx + 1, Ny + 1))
        self.Hz = np.zeros((Nx, Ny))
        self.Hzx = np.zeros((Nx, Ny))
        self.Hzy = np.zeros((Nx, Ny))

        #the sigmas must be assembled by the solver
        self.sigma_x = np.zeros_like(self.Ex)
        self.sigma_y = np.zeros_like(self.Ex)
        self.sigma_z = np.zeros_like(self.Ex)
        self.sigma_star_x = np.zeros_like(self.Hz)
        self.sigma_star_y = np.zeros_like(self.Hz)
        self.sigma_star_z = np.zeros_like(self.Hz)

        self.alpha_pml = 3
        self.R0_pml = 0.001

        #we assemble these after the sigmas
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
