import numpy as np
import pyvista as pv

from field import Field


class GridFIT3D:
    """
    Class holding the grid info and the routines for cell extensions.

    Constructor arguments
    ----------------------
    xmin, xmax, ymin, ymax, zmin, zmax: float
        extent of the domain.
    Nx, Ny, Nz: int
        number of cells per direction
    conductors: obj, optional
        conductor object. See `conductors3d.py`
    stl_solids: dict, optional
        stl files to import in the domain.
        {'Solid 1': stl_1, 'Solid 2': stl_2, ...}
    """

    def __init__(self, xmin, xmax, ymin, ymax, zmin, zmax, Nx, Ny, Nz, 
                stl_solids=None):
        
        # domain limits
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.zmin = zmin
        self.zmax = zmax
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.dx = (xmax - xmin) / Nx
        self.dy = (ymax - ymin) / Ny
        self.dz = (ymax - ymin) / Nz

        # primal Grid G
        self.x = np.linspace(self.xmin, self.xmax, self.Nx+1)
        self.y = np.linspace(self.ymin, self.ymax, self.Ny+1)
        self.z = np.linspace(self.zmin, self.zmax, self.Nz+1)

        Y, X, Z = np.meshgrid(self.y, self.x, self.z)

        self.L = Field(self.Nx, self.Ny, self.Nz)
        self.L.field_x = X[1:, 1:, 1:] - X[:-1, :-1, :-1]
        self.L.field_y = Y[1:, 1:, 1:] - Y[:-1, :-1, :-1]
        self.L.field_z = Z[1:, 1:, 1:] - Z[:-1, :-1, :-1]

        self.iA = Field(self.Nx, self.Ny, self.Nz)
        self.iA.field_x = np.divide(1.0, self.L.field_y * self.L.field_z)
        self.iA.field_y = np.divide(1.0, self.L.field_x * self.L.field_z)
        self.iA.field_z = np.divide(1.0, self.L.field_x * self.L.field_y)

        # tilde grid ~G

        self.tx = (self.x[1:]+self.x[:-1])/2 
        self.ty = (self.y[1:]+self.y[:-1])/2
        self.tz = (self.z[1:]+self.z[:-1])/2

        self.tx = np.append(self.tx, self.tx[-1])
        self.ty = np.append(self.ty, self.ty[-1])
        self.tz = np.append(self.tz, self.tz[-1])

        tY, tX, tZ = np.meshgrid(self.ty, self.tx, self.tz)

        self.tL = Field(self.Nx, self.Ny, self.Nz)
        self.tL.field_x = tX[1:, 1:, 1:] - tX[:-1, :-1, :-1]
        self.tL.field_y = tY[1:, 1:, 1:] - tY[:-1, :-1, :-1]
        self.tL.field_z = tZ[1:, 1:, 1:] - tZ[:-1, :-1, :-1]

        self.itA = Field(self.Nx, self.Ny, self.Nz)
        aux = self.tL.field_y * self.tL.field_z
        self.itA.field_x = np.divide(1.0, aux, out=np.zeros_like(aux), where=aux!=0)
        aux = self.tL.field_x * self.tL.field_z
        self.itA.field_y = np.divide(1.0, aux, out=np.zeros_like(aux), where=aux!=0)
        aux = self.tL.field_x * self.tL.field_y
        self.itA.field_z = np.divide(1.0, aux, out=np.zeros_like(aux), where=aux!=0)
        del aux
            
    """
  Function to compute the length of the edges of the conformal grid.
    Inputs:
        - tol: an edge shorter than tol will be considered as of zero length
    """

    def compute_edges(self, tol=1e-8):
        # edges xy
        for kk in range(self.Nz + 1):
            for ii in range(self.Nx):
                for jj in range(self.Ny + 1):
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

        for kk in range(self.Nz + 1):
            for ii in range(self.Nx + 1):
                for jj in range(self.Ny):
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

        for jj in range(self.Ny + 1):
            for ii in range(self.Nx + 1):
                for kk in range(self.Nz):
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
    @staticmethod
    @jit('(f8[:,:,:], f8[:,:,:], f8[:,:,:], f8[:,:,:], f8[:,:,:], f8[:,:,:], f8[:,:,:], f8[:,:,:], f8[:,:,:], i4, i4, i4, f8, f8, f8)', nopython=True)
    def compute_areas(l_x, l_y, l_z, Sxy, Syz, Szx, Sxy_red, Syz_red, Szx_red, Nx, Ny, Nz, dx, dy, dz):

        for kk in range(Nz + 1):
            compute_areas_2D(l_x[:, :, kk], l_y[:, :, kk], Sxy[:, :, kk], Sxy_red[:, :, kk],
                                 Nx, Ny, dx, dy)

        for ii in range(Nx + 1):
            compute_areas_2D(l_y[ii, :, :], l_z[ii, :, :], Syz[ii, :, :], Syz_red[ii, :, :],
                                 Ny, Nz, dy, dz)

        for jj in range(Ny + 1):
            compute_areas_2D(l_x[:, jj, :], l_z[:, jj, :], Szx[:, jj, :], Szx_red[:, jj, :],
                                 Nx, Nz, dx, dz)

    """
  Function to mark wich cells are interior (int), require extension (unst), 
  are on the boundary(bound), are available for intrusion (avail)
    """
    @staticmethod
    @jit('(f8[:,:,:], f8[:,:,:], f8[:,:,:], i4, i4, i4, f8, f8, f8, f8[:,:,:], f8[:,:,:], f8[:,:,:], b1[:,:,:], b1[:,:,:], b1[:,:,:], f8[:,:,:], f8[:,:,:], f8[:,:,:], b1[:,:,:], b1[:,:,:], b1[:,:,:], b1[:,:,:], b1[:,:,:], b1[:,:,:], b1[:,:,:], b1[:,:,:], b1[:,:,:])', nopython=True)
    def mark_cells(l_x, l_y, l_z, Nx, Ny, Nz, dx, dy, dz, Sxy, Syz, Szx, flag_int_cell_xy, flag_int_cell_yz, flag_int_cell_zx,
                   Sxy_stab, Syz_stab, Szx_stab, flag_unst_cell_xy, flag_unst_cell_yz, flag_unst_cell_zx, flag_bound_cell_xy,
                   flag_bound_cell_yz, flag_bound_cell_zx, flag_avail_cell_xy, flag_avail_cell_yz, flag_avail_cell_zx):

        for kk in range(Nz + 1):
            mark_cells_2D(l_x[:, :, kk], l_y[:, :, kk], Nx, Ny, dx, dy, Sxy[:, :, kk],
                          flag_int_cell_xy[:, :, kk], Sxy_stab[:, :, kk], flag_unst_cell_xy[:, :, kk],
                          flag_bound_cell_xy[:, :, kk], flag_avail_cell_xy[:, :, kk])

        for ii in range(Nx + 1):
            mark_cells_2D(l_y[ii, :, :], l_z[ii, :, :], Ny, Nz, dy, dz, Syz[ii, :, :],
                          flag_int_cell_yz[ii, :, :], Syz_stab[ii, :, :], flag_unst_cell_yz[ii, :, :],
                          flag_bound_cell_yz[ii, :, :], flag_avail_cell_yz[ii, :, :])

        for jj in range(Ny + 1):
            mark_cells_2D(l_x[:, jj, :], l_z[:, jj, :], Nx, Nz, dx, dz, Szx[:, jj, :],
                          flag_int_cell_zx[:, jj, :], Szx_stab[:, jj, :], flag_unst_cell_zx[:, jj, :],
                          flag_bound_cell_zx[:, jj, :], flag_avail_cell_zx[:, jj, :])

    """
  Function to compute the extension of the unstable cells
    """

    def compute_extensions(self):
        #breakpoint()
        for kk in range(self.Nz + 1):
            #breakpoint()
            Grid2D.compute_extensions(Nx=self.Nx, Ny=self.Ny, S=self.Sxy[:, :, kk],
                                      flag_int_cell=self.flag_int_cell_xy[:, :, kk],
                                      S_stab=self.Sxy_stab[:, :, kk], S_enl=self.Sxy_enl[:, :, kk],
                                      S_red=self.Sxy_red[:, :, kk],
                                      flag_unst_cell=self.flag_unst_cell_xy[:, :, kk],
                                      flag_avail_cell=self.flag_avail_cell_xy[:, :, kk],
                                      flag_ext_cell=self.flag_ext_cell_xy[:, :, kk],
                                      flag_intr_cell=self.flag_intr_cell_xy[:,:,kk],
                                      borrowing=self.borrowing_xy[:, :, kk],
                                      kk=kk, l_verbose=False)

        for ii in range(self.Nx + 1):
            Grid2D.compute_extensions(Nx=self.Ny, Ny=self.Nz, S=self.Syz[ii, :, :],
                                      flag_int_cell=self.flag_int_cell_yz[ii, :, :],
                                      S_stab=self.Syz_stab[ii, :, :], S_enl=self.Syz_enl[ii, :, :],
                                      S_red=self.Syz_red[ii, :, :],
                                      flag_unst_cell=self.flag_unst_cell_yz[ii, :, :],
                                      flag_avail_cell=self.flag_avail_cell_yz[ii, :, :],
                                      flag_ext_cell=self.flag_ext_cell_yz[ii, :, :],
                                      borrowing=self.borrowing_yz[ii, :, :],
                                      flag_intr_cell=self.flag_intr_cell_yz[ii,:,:],
                                      kk = ii, l_verbose=False)
            
        for jj in range(self.Ny + 1):
            Grid2D.compute_extensions(Nx=self.Nx, Ny=self.Nz, S=self.Szx[:, jj, :],
                                      flag_int_cell=self.flag_int_cell_zx[:, jj, :],
                                      S_stab=self.Szx_stab[:, jj, :], S_enl=self.Szx_enl[:, jj, :],
                                      S_red=self.Szx_red[:, jj, :],
                                      flag_unst_cell=self.flag_unst_cell_zx[:, jj, :],
                                      flag_avail_cell=self.flag_avail_cell_zx[:, jj, :],
                                      flag_ext_cell=self.flag_ext_cell_zx[:, jj, :],
                                      flag_intr_cell=self.flag_intr_cell_zx[:,jj,:],
                                      borrowing=self.borrowing_zx[:, jj, :],
                                      kk = jj, l_verbose=False)
