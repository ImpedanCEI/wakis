import numpy as np
from grid2D import Grid2D


def seg_length(x_1, y_1, z_1, x_2, y_2, z_2):
    return np.linalg.norm(np.array([x_1 - x_2, y_1 - y_2, z_1 - z_2]))


def eq(a, b, tol=1e-8):
    return abs(a - b) < tol


def neq(a, b, tol=1e-8):
    return not eq(a, b, tol)


class Grid3D:
    """
  Class holding the grid info and the routines for cell extensions.
    Constructor arguments:
        - xmin, xmax, ymin, ymax, zmin, zmax: extent of the domain.
        - nx, ny, nz: number of cells per direction
        - conductors: conductor object
        - sol_type: type of solver. 'FDTD' for staircased FDTD, 'DM' for Conformal Dey-Mittra FDTD,
                    'ECT' for Enlarged Cell Technique conformal FDTD
    """

    def __init__(self, xmin, xmax, ymin, ymax, zmin, zmax, nx, ny, nz, conductors, sol_type):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.zmin = zmin
        self.zmax = zmax
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dx = (xmax - xmin) / nx
        self.dy = (ymax - ymin) / ny
        self.dz = (ymax - ymin) / nz
        self.conductors = conductors

        self.l_x = np.zeros((nx, ny + 1, nz + 1))
        self.l_y = np.zeros((nx + 1, ny, nz + 1))
        self.l_z = np.zeros((nx + 1, ny + 1, nz))
        self.Sxy = np.zeros((nx, ny, nz + 1))
        self.Syz = np.zeros((nx + 1, ny, nz))
        self.Szx = np.zeros((nx, ny + 1, nz))
        self.Sxy_stab = np.zeros_like(self.Sxy)
        self.Sxy_enl = np.zeros_like(self.Sxy)
        self.Sxy_red = np.zeros_like(self.Sxy)
        self.Syz_stab = np.zeros_like(self.Syz)
        self.Syz_enl = np.zeros_like(self.Syz)
        self.Syz_red = np.zeros_like(self.Syz)
        self.Szx_stab = np.zeros_like(self.Szx)
        self.Szx_enl = np.zeros_like(self.Szx)
        self.Szx_red = np.zeros_like(self.Szx)
        self.flag_unst_cell_xy = np.zeros_like(self.Sxy, dtype=bool)
        self.flag_intr_cell_xy = np.zeros_like(self.Sxy, dtype=bool)
        self.flag_int_cell_xy = np.zeros_like(self.Sxy, dtype=bool)
        self.flag_bound_cell_xy = np.zeros_like(self.Sxy, dtype=bool)
        self.flag_avail_cell_xy = np.zeros_like(self.Sxy, dtype=bool)
        self.flag_ext_cell_xy = np.zeros_like(self.Sxy, dtype=bool)
        self.flag_unst_cell_yz = np.zeros_like(self.Syz, dtype=bool)
        self.flag_intr_cell_yz = np.zeros_like(self.Syz, dtype=bool)
        self.flag_int_cell_yz = np.zeros_like(self.Syz, dtype=bool)
        self.flag_bound_cell_yz = np.zeros_like(self.Syz, dtype=bool)
        self.flag_avail_cell_yz = np.zeros_like(self.Syz, dtype=bool)
        self.flag_ext_cell_yz = np.zeros_like(self.Syz, dtype=bool)
        self.flag_unst_cell_zx = np.zeros_like(self.Szx, dtype=bool)
        self.flag_intr_cell_zx = np.zeros_like(self.Szx, dtype=bool)
        self.flag_int_cell_zx = np.zeros_like(self.Szx, dtype=bool)
        self.flag_bound_cell_zx = np.zeros_like(self.Szx, dtype=bool)
        self.flag_avail_cell_zx = np.zeros_like(self.Szx, dtype=bool)
        self.flag_ext_cell_zx = np.zeros_like(self.Szx, dtype=bool)
        self.broken_xy = np.zeros_like(self.Sxy, dtype=bool)
        self.broken_yz = np.zeros_like(self.Syz, dtype=bool)
        self.broken_zx = np.zeros_like(self.Szx, dtype=bool)

        if (sol_type is not 'FDTD') and (sol_type is not 'DM') and (sol_type is not 'ECT'):
            raise ValueError("sol_type must be:\n" +
                             "\t'FDTD' for standard staircased FDTD\n" +
                             "\t'DM' for Dey-Mittra conformal FDTD\n" +
                             "\t'ECT' for Enlarged Cell Technique conformal FDTD")

        self.compute_edges()
        if sol_type is 'DM' or sol_type is 'FDTD':
            self.compute_areas()
            self.mark_cells()
        elif sol_type is 'ECT':
            self.compute_areas()
            self.mark_cells()
            # info about intruded cells (i,j,[(i_borrowing,j_borrowing,area_borrowing, )])
            self.borrowing_xy = np.empty((nx, ny, nz + 1), dtype=object)
            self.borrowing_yz = np.empty((nx + 1, ny, nz), dtype=object)
            self.borrowing_zx = np.empty((nx, ny + 1, nz), dtype=object)

            for ii in range(nx):
                for jj in range(ny):
                    for kk in range(nz + 1):
                        self.borrowing_xy[ii, jj, kk] = []
            for ii in range(nx + 1):
                for jj in range(ny):
                    for kk in range(nz):
                        self.borrowing_yz[ii, jj, kk] = []
            for ii in range(nx):
                for jj in range(ny + 1):
                    for kk in range(nz):
                        self.borrowing_zx[ii, jj, kk] = []

            self.flag_ext_cell_xy = self.flag_unst_cell_xy.copy()
            self.flag_ext_cell_yz = self.flag_unst_cell_yz.copy()
            self.flag_ext_cell_zx = self.flag_unst_cell_zx.copy()
            self.mark_cells()
            self.compute_extensions()

    """
  Function to compute the length of the edges of the conformal grid.
    Inputs:
        - tol: an edge shorter than tol will be considered as of zero length
    """

    def compute_edges(self, tol=1e-8):
        # edges xy
        for kk in range(self.nz + 1):
            for ii in range(self.nx):
                for jj in range(self.ny + 1):
                    x_1 = ii * self.dx + self.xmin
                    y_1 = jj * self.dy + self.ymin
                    x_2 = (ii + 1) * self.dx + self.xmin
                    y_2 = jj * self.dy + self.ymin
                    z = kk * self.dz + self.zmin
                    # if point 1 is in conductor
                    if self.conductors.in_conductor(x_1, y_1, z):
                        # if point 2 is in conductor, length of the l_x[i, j] is zero
                        if self.conductors.in_conductor(x_2, y_2, z):
                            self.l_x[ii, jj, kk] = 0
                        # if point 2 is not in conductor, length of l_x[i, j]
                        # is the fractional length
                        else:
                            self.l_x[ii, jj, kk] = seg_length(self.conductors.intersec_x(x_2,
                                                                                         y_2, z),
                                                              y_2, z, x_2, y_2, z)
                    # if point 1 is not in conductor
                    else:
                        # if point 2 is in conductor, length of l_x[i, j] is the fractional length
                        if self.conductors.in_conductor(x_2, y_2, z):
                            self.l_x[ii, jj, kk] = seg_length(x_1, y_1, z,
                                                              self.conductors.intersec_x(x_1, y_1,
                                                                                         z),
                                                              y_1, z)
                        # if point 2 is not in conductor, length of l_x[i, j] is dx
                        else:
                            self.l_x[ii, jj, kk] = self.dx

        for kk in range(self.nz + 1):
            for ii in range(self.nx + 1):
                for jj in range(self.ny):
                    x_1 = ii * self.dx + self.xmin
                    y_1 = jj * self.dy + self.ymin
                    x_3 = ii * self.dx + self.xmin
                    y_3 = (jj + 1) * self.dy + self.ymin
                    z = kk * self.dz + self.zmin
                    # if point 1 is in conductor
                    if self.conductors.in_conductor(x_1, y_1, z):
                        # if point 3 to the right is in conductor, length of the l_y[i, j] is zero
                        if self.conductors.in_conductor(x_3, y_3, z):
                            self.l_y[ii, jj, kk] = 0
                        # if point 3 is not in conductor, length of l_y[i, j]
                        # is the fractional length
                        else:
                            self.l_y[ii, jj, kk] = seg_length(x_3,
                                                              self.conductors.intersec_y(x_3, y_3,
                                                                                         z),
                                                              z, x_3, y_3, z)
                    # if point 1 is not in conductor
                    else:
                        # if point 3 is in conductor, length of the l_y[i, j]
                        # is the fractional length
                        if self.conductors.in_conductor(x_3, y_3, z):
                            self.l_y[ii, jj, kk] = seg_length(x_1, y_1, z, x_1,
                                                              self.conductors.intersec_y(x_1, y_1,
                                                                                         z),
                                                              z)
                        # if point 3 is not in conductor, length of l_y[i, j] is dy
                        else:
                            self.l_y[ii, jj, kk] = self.dy

        for jj in range(self.ny + 1):
            for ii in range(self.nx + 1):
                for kk in range(self.nz):
                    y = jj*self.dy + self.ymin
                    x_1 = ii * self.dx + self.xmin
                    z_1 = kk * self.dz + self.zmin
                    x_3 = ii * self.dx + self.xmin
                    z_3 = (kk + 1) * self.dz + self.zmin
                    # if point 1 is in conductor
                    if self.conductors.in_conductor(x_1, y, z_1):
                        # if point 3 to the right is in conductor, length of the l_y[i, j] is zero
                        if self.conductors.in_conductor(x_3, y, z_3):
                            self.l_z[ii, jj, kk] = 0
                        # if point 3 is not in conductor, length of l_y[i, j]
                        # is the fractional length
                        else:
                            self.l_z[ii, jj, kk] = seg_length(x_3, y,
                                                              self.conductors.intersec_z(x_3,
                                                                                         y, z_3),
                                                              x_3, y, z_3)
                    # if point 1 is not in conductor
                    else:
                        # if point 3 is in conductor, length of the l_y[i, j]
                        # is the fractional length
                        if self.conductors.in_conductor(x_3, y, z_3):
                            self.l_z[ii, jj, kk] = seg_length(x_1, y, z_1, x_1, y,
                                                              self.conductors.intersec_z(x_1, y,
                                                                                         z_1))
                        # if point 3 is not in conductor, length of l_y[i, j] is dy
                        else:
                            self.l_z[ii, jj, kk] = self.dz

        # set to zero the length of very small cells
        if tol > 0.:
            self.l_x /= self.dx
            self.l_y /= self.dy
            self.l_z /= self.dz

            low_values_flags = abs(self.l_x) < tol
            high_values_flags = abs(self.l_x - 1.) < tol
            self.l_x[low_values_flags] = 0
            self.l_x[high_values_flags] = 1.

            low_values_flags = abs(self.l_y) < tol
            high_values_flags = abs(self.l_y - 1.) < tol
            self.l_y[low_values_flags] = 0
            self.l_y[high_values_flags] = 1.

            low_values_flags = abs(self.l_z) < tol
            high_values_flags = abs(self.l_z - 1.) < tol
            self.l_z[low_values_flags] = 0
            self.l_z[high_values_flags] = 1.

            self.l_x *= self.dx
            self.l_y *= self.dy
            self.l_z *= self.dz

    """
  Function to compute the area of the cells of the conformal grid.
    """

    def compute_areas(self):
        for kk in range(self.nz + 1):
            Grid2D.compute_areas(l_x=self.l_x[:, :, kk], l_y=self.l_y[:, :, kk],
                                 S=self.Sxy[:, :, kk], S_red=self.Sxy_red[:, :, kk], nx=self.nx,
                                 ny=self.ny, dx=self.dx, dy=self.dy)

        for ii in range(self.nx + 1):
            Grid2D.compute_areas(l_x=self.l_y[ii, :, :], l_y=self.l_z[ii, :, :],
                                 S=self.Syz[ii, :, :], S_red=self.Syz_red[ii, :, :], nx=self.ny,
                                 ny=self.nz, dx=self.dy, dy=self.dz)

        for jj in range(self.ny + 1):
            Grid2D.compute_areas(l_x=self.l_x[:, jj, :], l_y=self.l_z[:, jj, :],
                                 S=self.Szx[:, jj, :], S_red=self.Szx_red[:, jj, :], nx=self.nx,
                                 ny=self.nz, dx=self.dx, dy=self.dz)

    """
  Function to mark wich cells are interior (int), require extension (unst), 
  are on the boundary(bound), are available for intrusion (avail)
    """

    def mark_cells(self):
        for kk in range(self.nz + 1):
            Grid2D.mark_cells(l_x=self.l_x[:, :, kk], l_y=self.l_y[:, :, kk], nx=self.nx,
                              ny=self.ny, dx=self.dx, dy=self.dy, S=self.Sxy[:, :, kk],
                              flag_int_cell=self.flag_int_cell_xy[:, :, kk],
                              S_stab=self.Sxy_stab[:, :, kk],
                              flag_unst_cell=self.flag_unst_cell_xy[:, :, kk],
                              flag_bound_cell=self.flag_bound_cell_xy[:, :, kk],
                              flag_avail_cell=self.flag_avail_cell_xy[:, :, kk])

        for ii in range(self.nx + 1):
            Grid2D.mark_cells(l_x=self.l_y[ii, :, :], l_y=self.l_z[ii, :, :], nx=self.ny,
                              ny=self.nz, dx=self.dy, dy=self.dz, S=self.Syz[ii, :, :],
                              flag_int_cell=self.flag_int_cell_yz[ii, :, :],
                              S_stab=self.Syz_stab[ii, :, :],
                              flag_unst_cell=self.flag_unst_cell_yz[ii, :, :],
                              flag_bound_cell=self.flag_bound_cell_yz[ii, :, :],
                              flag_avail_cell=self.flag_avail_cell_yz[ii, :, :])

        for jj in range(self.ny + 1):
            Grid2D.mark_cells(l_x=self.l_x[:, jj, :], l_y=self.l_z[:, jj, :], nx=self.nx,
                              ny=self.nz, dx=self.dx, dy=self.dz, S=self.Szx[:, jj, :],
                              flag_int_cell=self.flag_int_cell_zx[:, jj, :],
                              S_stab=self.Szx_stab[:, jj, :],
                              flag_unst_cell=self.flag_unst_cell_zx[:, jj, :],
                              flag_bound_cell=self.flag_bound_cell_zx[:, jj, :],
                              flag_avail_cell=self.flag_avail_cell_zx[:, jj, :])

    """
  Function to compute the extension of the unstable cells
    """

    def compute_extensions(self):
        #breakpoint()
        for kk in range(self.nz + 1):
            #breakpoint()
            Grid2D.compute_extensions(nx=self.nx, ny=self.ny, S=self.Sxy[:, :, kk],
                                      flag_int_cell=self.flag_int_cell_xy[:, :, kk],
                                      S_stab=self.Sxy_stab[:, :, kk], S_enl=self.Sxy_enl[:, :, kk],
                                      S_red=self.Sxy_red[:, :, kk],
                                      flag_unst_cell=self.flag_unst_cell_xy[:, :, kk],
                                      flag_avail_cell=self.flag_avail_cell_xy[:, :, kk],
                                      flag_ext_cell=self.flag_ext_cell_xy[:, :, kk],
                                      flag_intr_cell=self.flag_intr_cell_xy[:,:,kk],
                                      borrowing=self.borrowing_xy[:, :, kk],
                                      kk=kk, l_verbose=False)

        for ii in range(self.nx + 1):
            Grid2D.compute_extensions(nx=self.ny, ny=self.nz, S=self.Syz[ii, :, :],
                                      flag_int_cell=self.flag_int_cell_yz[ii, :, :],
                                      S_stab=self.Syz_stab[ii, :, :], S_enl=self.Syz_enl[ii, :, :],
                                      S_red=self.Syz_red[ii, :, :],
                                      flag_unst_cell=self.flag_unst_cell_yz[ii, :, :],
                                      flag_avail_cell=self.flag_avail_cell_yz[ii, :, :],
                                      flag_ext_cell=self.flag_ext_cell_yz[ii, :, :],
                                      borrowing=self.borrowing_yz[ii, :, :],
                                      flag_intr_cell=self.flag_intr_cell_yz[ii,:,:],
                                      kk = ii, l_verbose=False)
            
        for jj in range(self.ny + 1):
            Grid2D.compute_extensions(nx=self.nx, ny=self.nz, S=self.Szx[:, jj, :],
                                      flag_int_cell=self.flag_int_cell_zx[:, jj, :],
                                      S_stab=self.Szx_stab[:, jj, :], S_enl=self.Szx_enl[:, jj, :],
                                      S_red=self.Szx_red[:, jj, :],
                                      flag_unst_cell=self.flag_unst_cell_zx[:, jj, :],
                                      flag_avail_cell=self.flag_avail_cell_zx[:, jj, :],
                                      flag_ext_cell=self.flag_ext_cell_zx[:, jj, :],
                                      flag_intr_cell=self.flag_intr_cell_zx[:,jj,:],
                                      borrowing=self.borrowing_zx[:, jj, :],
                                      kk = jj, l_verbose=False)
