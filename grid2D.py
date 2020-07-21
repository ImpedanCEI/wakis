import numpy as np


def seg_length(x_1, y_1, x_2, y_2):
    return np.linalg.norm(np.array([x_1 - x_2, y_1 - y_2]))


def eq(a, b, tol=1e-8):
    return abs(a - b) < tol


def neq(a, b, tol=1e-8):
    return not eq(a, b, tol)


class Grid2D:
    """
  Class holding the grid info and the routines for cell extensions.
    Constructor arguments:
        - xmin,xmax,ymin,ymax: extent of the domain.
        - nx, ny: number of cells per direction
        - conductors: conductor object
        - sol_type: type of solver. 'FDTD' for staircased FDTD, 'DM' for Conformal Dey-Mittra FDTD,
                    'ECT' for Extended Cell Technique conformal FDTD
    """

    def __init__(self, xmin, xmax, ymin, ymax, nx, ny, conductors, sol_type):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.nx = nx
        self.ny = ny
        self.dx = (xmax - xmin) / nx
        self.dy = (ymax - ymin) / ny
        self.conductors = conductors

        self.l_x = np.zeros((nx, ny + 1))
        self.l_y = np.zeros((nx + 1, ny))
        self.S = np.zeros((nx, ny))
        self.S_stab = np.zeros((nx, ny))
        self.S_enl = np.zeros((nx, ny))
        self.S_red = np.zeros((nx, ny))
        self.flag_unst_cell = np.zeros((nx, ny), dtype=bool)
        self.flag_int_cell = np.zeros((nx, ny), dtype=bool)
        self.flag_bound_cell = np.zeros((nx, ny), dtype=bool)
        self.flag_avail_cell = np.zeros((nx, ny), dtype=bool)
        self.flag_ext_cell = np.zeros((nx, ny), dtype=bool)
        self.broken = np.zeros((self.nx, self.ny), dtype=bool)

        if (sol_type is not 'FDTD') and (sol_type is not 'DM') and (sol_type is not 'ECT'):
            raise ValueError("sol_type must be:\n" +
                             "\t'FDTD' for standard staircased FDTD\n" +
                             "\t'DM' for Dey-Mittra conformal FDTD\n" +
                             "\t'ECT' for Enlarged Cell Technique conformal FDTD")

        if sol_type is 'DM':
            self.compute_edges()
            self.compute_areas()
            self.mark_cells()
        elif sol_type is 'ECT':
            self.compute_edges()
            self.compute_areas()
            self.mark_cells()
            # info about intruded cells (i,j,[(i_borrowing,j_borrowing,area_borrowing, )])
            self.borrowing = np.empty((nx, ny), dtype=object)
            # info about intruding cells  (i, j, [(i_lending, j_lending, area_lending)])
            self.lending = np.empty((nx, ny), dtype=object)
            for i in range(nx):
                for j in range(ny):
                    self.borrowing[i, j] = []
                    self.lending[i, j] = []
            self.compute_extensions()

    """
  Function to compute the length of the edges of the conformal grid.
    Inputs:
        - tol: an edge shorter than tol will be considered as of zero length
    """

    def compute_edges(self, tol=1e-8):
        # shortcuts
        in_conductor = self.conductors.in_conductor
        intersec_x = self.conductors.intersec_x
        intersec_y = self.conductors.intersec_y
        """
         Notation:
        
               (x_3, y_3)----- l_x[i, j + 1] ---------
                   |                                  |
                   |                                  |
                  l_y[i, j]        (i, j)            l_y[i + 1, j]
                   |                                  |
                   |                                  |
               (x_1, y_1)------- l_x[i, j] -------(x_2, y_2)
         
        """
        for ii in range(self.nx):
            for jj in range(self.ny + 1):
                x_1 = ii * self.dx + self.xmin
                y_1 = jj * self.dy + self.ymin
                x_2 = (ii + 1) * self.dx + self.xmin
                y_2 = jj * self.dy + self.ymin
                # if point 1 is in conductor
                if in_conductor(x_1, y_1):
                    # if point 2 is in conductor, length of the l_x[i, j] is zero
                    if in_conductor(x_2, y_2):
                        self.l_x[ii, jj] = 0
                    # if point 2 is not in conductor, length of l_x[i, j] is the fractional length
                    else:
                        self.l_x[ii, jj] = seg_length(intersec_x(x_2, y_2), y_1, x_2, y_2)
                # if point 1 is not in conductor
                else:
                    # if point 2 is in conductor, length of l_x[i, j] is the fractional length
                    if in_conductor(x_2, y_2):
                        self.l_x[ii, jj] = seg_length(x_1, y_1, intersec_x(x_1, y_1), y_2)
                    # if point 2 is not in conductor, length of l_x[i, j] is dx
                    else:
                        self.l_x[ii, jj] = self.dx

        for ii in range(self.nx + 1):
            for jj in range(self.ny):
                x_1 = ii * self.dx + self.xmin
                y_1 = jj * self.dy + self.ymin
                x_3 = ii * self.dx + self.xmin
                y_3 = (jj + 1) * self.dy + self.ymin
                # if point 1 is in conductor
                if in_conductor(x_1, y_1):
                    # if point 3 to the right is in conductor, length of the l_y[i, j] is zero
                    if in_conductor(x_3, y_3):
                        self.l_y[ii, jj] = 0
                    # if point 3 is not in conductor, length of l_y[i, j] is the fractional length
                    else:
                        self.l_y[ii, jj] = seg_length(x_1, intersec_y(x_3, y_3), x_3, y_3)
                # if point 1 is not in conductor
                else:
                    # if point 3 is in conductor, length of the l_y[i, j] is the fractional length
                    if in_conductor(x_3, y_3):
                        self.l_y[ii, jj] = seg_length(x_1, y_1, x_3, intersec_y(x_1, y_1))
                    # if point 3 is not in conductor, length of l_y[i, j] is dy
                    else:
                        self.l_y[ii, jj] = self.dy

        # set to zero the length of very small cells
        if tol > 0.:
            self.l_x = self.l_x / self.dx
            self.l_y = self.l_y / self.dy
            dx = self.dx
            dy = self.dy
            self.dx = 1
            self.dy = 1
            low_values_flags = abs(self.l_x) < self.dx * tol
            high_values_flags = abs(self.l_x - self.dx) < self.dx * tol
            self.l_x[low_values_flags] = 0
            self.l_x[high_values_flags] = self.dx

            low_values_flags = abs(self.l_y) < self.dy * tol
            high_values_flags = abs(self.l_y - self.dy) < self.dy * tol
            self.l_y[low_values_flags] = 0
            self.l_y[high_values_flags] = self.dy

            self.dx = dx
            self.dy = dy
            self.l_x = self.l_x * self.dx
            self.l_y = self.l_y * self.dy

    """
  Function to compute the area of the cells of the conformal grid.
    """

    def compute_areas(self):
        # Normalize the grid lengths for robustness
        l_x = self.l_x / self.dx
        l_y = self.l_y / self.dy

        # Loop over the cells
        for ii in range(self.nx):
            for jj in range(self.ny):
                # If at least 3 edges have full length consider the area as full
                # (this also takes care of the case in which an edge lies exactly on the boundary)
                if np.sum([eq(l_x[ii, jj], 1.0), eq(l_y[ii, jj], 1.0),
                           eq(l_x[ii, jj + 1], 1.0), eq(l_y[ii + 1, jj], 1.0)]) >= 3:
                    self.S[ii, jj] = 1.0
                elif (eq(l_x[ii, jj], 0.) and neq(l_y[ii, jj], 0.)
                      and neq(l_x[ii, jj + 1], 0.) and neq(l_y[ii + 1, jj], 0)):
                    self.S[ii, jj] = 0.5 * (l_y[ii, jj] + l_y[ii + 1, jj]) * l_x[ii, jj + 1]
                elif (eq(l_x[ii, jj + 1], 0.) and neq(l_y[ii, jj], 0.)
                      and neq(l_x[ii, jj], 0.) and neq(l_y[ii + 1, jj], 0.)):
                    self.S[ii, jj] = 0.5 * (l_y[ii, jj] + l_y[ii + 1, jj]) * l_x[ii, jj]
                elif (eq(l_y[ii, jj], 0.) and neq(l_x[ii, jj], 0.)
                      and neq(l_x[ii, jj + 1], 0.) and neq(l_y[ii + 1, jj], 0.)):
                    self.S[ii, jj] = 0.5 * (l_x[ii, jj] + l_x[ii, jj + 1]) * l_y[ii + 1, jj]
                elif (eq(l_y[ii + 1, jj], 0.) and neq(l_x[ii, jj], 0.) and neq(l_y[ii, jj], 0.)
                      and neq(l_x[ii, jj + 1], 0.)):
                    self.S[ii, jj] = 0.5 * (l_x[ii, jj] + l_x[ii, jj + 1]) * l_y[ii, jj]
                elif (eq(l_x[ii, jj], 0.) and eq(l_y[ii, jj], 0.)
                      and neq(l_x[ii, jj + 1], 0.) and neq(l_y[ii + 1, jj], 0.)):
                    self.S[ii, jj] = 0.5 * l_x[ii, jj + 1] * l_y[ii + 1, jj]
                elif (eq(l_x[ii, jj], 0.) and eq(l_y[ii + 1, jj], 0.)
                      and neq(l_x[ii, jj + 1], 0.) and neq(l_y[ii, jj], 0.)):
                    self.S[ii, jj] = 0.5 * l_x[ii, jj + 1] * l_y[ii, jj]
                elif (eq(l_x[ii, jj + 1], 0.) and eq(l_y[ii + 1, jj], 0.)
                      and neq(l_x[ii, jj], 0.) and neq(l_y[ii, jj], 0.)):
                    self.S[ii, jj] = 0.5 * l_x[ii, jj] * l_y[ii, jj]
                elif (eq(l_x[ii, jj + 1], 0.) and eq(l_y[ii, jj], 0.)
                      and neq(l_x[ii, jj], 0.) and neq(l_y[ii + 1, jj], 0.)):
                    self.S[ii, jj] = 0.5 * l_x[ii, jj] * l_y[ii + 1, jj]
                elif (0. < l_x[ii, jj] <= 1. and 0. < l_y[ii, jj] <= 1.
                      and eq(l_x[ii, jj + 1], 1.) and eq(l_y[ii + 1, jj], 1.)):
                    self.S[ii, jj] = 1. - 0.5 * (1. - l_x[ii, jj]) * (1. - l_y[ii, jj])
                elif (0. < l_x[ii, jj] <= 1. and 0. < l_y[ii + 1, jj] <= 1.
                      and eq(l_x[ii, jj + 1], 1.) and eq(l_y[ii, jj], 1.)):
                    self.S[ii, jj] = 1. - 0.5 * (1. - l_x[ii, jj]) * (1. - l_y[ii + 1, jj])
                elif (0. < l_x[ii, jj + 1] <= 1. and 0. < l_y[ii + 1, jj] <= 1.
                      and eq(l_x[ii, jj], 1.) and eq(l_y[ii, jj], 1.)):
                    self.S[ii, jj] = 1. - 0.5 * (1. - l_x[ii, jj + 1]) * (1. - l_y[ii + 1, jj])
                elif (0. < l_x[ii, jj + 1] <= 1. and 0. < l_y[ii, jj] <= 1.
                      and eq(l_x[ii, jj], 1.) and eq(l_y[ii + 1, jj], 1.)):
                    self.S[ii, jj] = 1. - 0.5 * (1. - l_x[ii, jj + 1]) * (1. - l_y[ii, jj])

        # Undo the normalization
        self.S = self.S * self.dx * self.dy
        self.S_red = self.S.copy()

    """
  Function to mark wich cells are interior (int), require extension (unst), 
  are on the boundary(bound), are available for intrusion (avail)
    """

    def mark_cells(self):

        for ii in range(self.nx):
            for jj in range(self.ny):
                self.flag_int_cell[ii, jj] = self.S[ii, jj] > 0
                self.S_stab[ii, jj] = 0.5 * np.max(
                    [self.l_x[ii, jj] * self.dy, self.l_x[ii, jj + 1] * self.dy,
                     self.l_y[ii, jj] * self.dx,
                     self.l_y[ii + 1, jj] * self.dx])
                self.flag_unst_cell[ii, jj] = self.S[ii, jj] < self.S_stab[ii, jj]
                self.flag_bound_cell[ii, jj] = (0 < self.l_x[ii, jj] < self.dx) or (
                        0 < self.l_y[ii, jj] < self.dy) or (
                                                       0 < self.l_x[ii, jj + 1] < self.dx) or (
                                                       0 < self.l_y[ii + 1, jj] < self.dy)
                self.flag_avail_cell[ii, jj] = self.flag_int_cell[ii, jj] and (
                    not self.flag_unst_cell[ii, jj])  # and (not self.flag_bound_cell[ii, jj])

    """
  Function to compute the extension of the unstable cells 
    """

    def compute_extensions(self):
        self.flag_ext_cell = self.flag_unst_cell.copy()
        N = np.sum(self.flag_ext_cell)
        print('ext cells: %d' % N)
        # Do the simple one-cell extension
        self._compute_extensions_one_cell()
        N_one_cell = (N - np.sum(self.flag_ext_cell))
        print('one cell exts: %d' % N_one_cell)
        # If any cell could not be extended do the four-cell extension
        if np.sum(self.flag_ext_cell) > 0:
            N = np.sum(self.flag_ext_cell)
            self._compute_extensions_four_cells()
            N_four_cells = (N - np.sum(self.flag_ext_cell))
            print('four cell exts: %d' % N_four_cells)
        # If any cell could not be extended do the eight-cell extension
        if np.sum(self.flag_ext_cell) > 0:
            N = np.sum(self.flag_ext_cell)
            self._compute_extensions_eight_cells()
            N_eight_cells = (N - np.sum(self.flag_ext_cell))
            print('eight cell exts: %d' % N_eight_cells)
        # If any cell could not be extended the algorithm failed
        if np.sum(self.flag_ext_cell) > 0:
            N = (np.sum(self.flag_ext_cell))
            raise RuntimeError(str(N) + 'cells could not be extended.\n' +
                               'Please refine the mesh')

    """
  Function to compute the one-cell extension of the unstable cells 
    """

    def _compute_extensions_one_cell(self):
        for ii in range(0, self.nx):
            for jj in range(0, self.ny):
                if (self.flag_unst_cell[ii, jj] and self.flag_int_cell[ii, jj]
                        and self.flag_ext_cell[ii, jj]):
                    S_ext = self.S_stab[ii, jj] - self.S[ii, jj]
                    if self.S[ii, jj - 1] > S_ext and self.flag_avail_cell[ii, jj - 1]:
                        denom = self.S[ii, jj - 1]
                        patch = S_ext * self.S[ii, jj - 1] / denom
                        if self.S_red[ii, jj - 1] - patch > 0:
                            self.S_red[ii, jj - 1] -= patch
                            self.borrowing[ii, jj].append([ii, jj - 1, patch, None])
                            self.lending[ii, jj - 1].append([ii, jj, patch, None])
                            self.S_enl[ii, jj] = self.S[ii, jj] + patch
                            self.flag_ext_cell[ii, jj] = False
                    if (self.S[ii + 1, jj] > S_ext and self.flag_avail_cell[ii + 1, jj]
                            and self.flag_ext_cell[ii, jj]):
                        denom = self.S[ii + 1, jj]
                        patch = S_ext * self.S[ii + 1, jj] / denom
                        if self.S_red[ii + 1, jj] - patch > 0:
                            self.S_red[ii + 1, jj] -= patch
                            self.borrowing[ii, jj].append([ii + 1, jj, patch, None])
                            self.lending[ii + 1, jj].append([ii, jj, patch, None])
                            self.S_enl[ii, jj] = self.S[ii, jj] + patch
                            self.flag_ext_cell[ii, jj] = False

                    if (self.S[ii - 1, jj] > S_ext and self.flag_avail_cell[ii - 1, jj]
                            and self.flag_ext_cell[ii, jj]):
                        denom = self.S[ii - 1, jj]
                        patch = S_ext * self.S[ii - 1, jj] / denom
                        if self.S_red[ii - 1, jj] - patch > 0:
                            self.S_red[ii - 1, jj] -= patch
                            self.borrowing[ii, jj].append([ii - 1, jj, patch, None])
                            self.lending[ii - 1, jj].append([ii, jj, patch, None])
                            self.S_enl[ii, jj] = self.S[ii, jj] + patch
                            self.flag_ext_cell[ii, jj] = False
                    if (self.S[ii, jj + 1] > S_ext and self.flag_avail_cell[ii, jj + 1]
                            and self.flag_ext_cell[ii, jj]):
                        denom = self.S[ii, jj + 1]
                        patch = S_ext * self.S[ii, jj + 1] / denom
                        if self.S_red[ii, jj + 1] - patch > 0:
                            self.S_red[ii, jj + 1] -= patch
                            self.borrowing[ii, jj].append([ii, jj + 1, patch, None])
                            self.lending[ii, jj + 1].append([ii, jj, patch, None])
                            self.S_enl[ii, jj] = self.S[ii, jj] + patch
                            self.flag_ext_cell[ii, jj] = False

    """
  Function to compute the four-cell extension of the unstable cells 
    """

    def _compute_extensions_four_cells(self):
        for ii in range(0, self.nx):
            for jj in range(0, self.ny):
                local_avail = self.flag_avail_cell.copy()
                if (self.flag_unst_cell[ii, jj] and self.flag_int_cell[ii, jj]
                        and self.flag_ext_cell[ii, jj]):
                    denom = ((self.flag_avail_cell[ii - 1, jj]) * self.S[ii - 1, jj] + (
                        self.flag_avail_cell[ii + 1, jj]) * self.S[ii + 1, jj] +
                             (self.flag_avail_cell[ii, jj - 1]) * self.S[ii, jj - 1] + (
                                 self.flag_avail_cell[ii, jj + 1]) * self.S[ii, jj + 1])
                    S_ext = self.S_stab[ii, jj] - self.S[ii, jj]
                    neg_cell = True
                    # idea: if any cell would reach negative area it is locally not available.
                    #       then denom has to be recomputed from scratch

                    while denom >= S_ext and neg_cell:
                        neg_cell = False
                        if local_avail[ii - 1, jj]:
                            patch = S_ext * self.S[ii - 1, jj] / denom
                            if self.S_red[ii - 1, jj] - patch <= 0:
                                neg_cell = True
                                local_avail[ii - 1, jj] = False
                            # self.flag_avail_cell[ii - 1, jj] = False
                        if local_avail[ii + 1, jj]:
                            patch = S_ext * self.S[ii + 1, jj] / denom
                            if self.S_red[ii + 1, jj] - patch <= 0:
                                neg_cell = True
                                local_avail[ii + 1, jj] = False
                            # self.flag_avail_cell[ii + 1, jj] = False
                        if local_avail[ii, jj - 1]:
                            patch = S_ext * self.S[ii, jj - 1] / denom
                            if self.S_red[ii, jj - 1] - patch <= 0:
                                neg_cell = True
                                local_avail[ii, jj - 1] = False
                            # self.flag_avail_cell[ii, jj - 1] = False
                        if local_avail[ii, jj + 1]:
                            patch = S_ext * self.S[ii, jj + 1] / denom
                            if self.S_red[ii, jj + 1] - patch <= 0:
                                neg_cell = True
                                local_avail[ii, jj + 1] = False
                        denom = ((local_avail[ii - 1, jj]) * self.S[ii - 1, jj] +
                                 (local_avail[ii + 1, jj]) * self.S[ii + 1, jj] +
                                 (local_avail[ii, jj - 1]) * self.S[ii, jj - 1] +
                                 (local_avail[ii, jj + 1]) * self.S[ii, jj + 1])

                    # If possible, do 4-cell extension
                    if denom >= S_ext:
                        self.S_enl[ii, jj] = self.S[ii, jj]
                        if local_avail[ii - 1, jj]:
                            patch = S_ext * self.S[ii - 1, jj] / denom
                            self.borrowing[ii, jj].append([ii - 1, jj, patch, None])
                            self.lending[ii - 1, jj].append([ii, jj, patch, None])
                            self.S_enl[ii, jj] += patch
                            self.S_red[ii - 1, jj] -= patch
                        if local_avail[ii + 1, jj]:
                            patch = S_ext * self.S[ii + 1, jj] / denom
                            self.borrowing[ii, jj].append([ii + 1, jj, patch, None])
                            self.lending[ii + 1, jj].append([ii, jj, patch, None])
                            self.S_enl[ii, jj] += patch
                            self.S_red[ii + 1, jj] -= patch
                        if local_avail[ii, jj - 1]:
                            patch = S_ext * self.S[ii, jj - 1] / denom
                            self.borrowing[ii, jj].append([ii, jj - 1, patch, None])
                            self.lending[ii, jj - 1].append([ii, jj, patch, None])
                            self.S_enl[ii, jj] += patch
                            self.S_red[ii, jj - 1] -= patch
                        if local_avail[ii, jj + 1]:
                            patch = S_ext * self.S[ii, jj + 1] / denom
                            self.borrowing[ii, jj].append([ii, jj + 1, patch, None])
                            self.lending[ii, jj + 1].append([ii, jj, patch, None])
                            self.S_enl[ii, jj] += patch
                            self.S_red[ii, jj + 1] -= patch

                        self.flag_ext_cell[ii, jj] = False

    """
  Function to compute the eight-cell extension of the unstable cells 
    """

    def _compute_extensions_eight_cells(self):
        for ii in range(0, self.nx):
            for jj in range(0, self.ny):
                local_avail = self.flag_avail_cell.copy()
                if (self.flag_unst_cell[ii, jj] and self.flag_int_cell[ii, jj]
                        and self.flag_ext_cell[ii, jj]):
                    self.S_enl[ii, jj] = self.S[ii, jj]
                    self.broken[ii, jj] = True
                    S_ext = self.S_stab[ii, jj] - self.S[ii, jj]

                    denom = ((self.flag_avail_cell[ii - 1, jj]) * self.S[ii - 1, jj] +
                             (self.flag_avail_cell[ii + 1, jj]) * self.S[ii + 1, jj] +
                             (self.flag_avail_cell[ii, jj - 1]) * self.S[ii, jj - 1] +
                             (self.flag_avail_cell[ii, jj + 1]) * self.S[ii, jj + 1] +
                             (self.flag_avail_cell[ii - 1, jj - 1]) * self.S[ii - 1, jj - 1] +
                             (self.flag_avail_cell[ii + 1, jj - 1]) * self.S[ii + 1, jj - 1] +
                             (self.flag_avail_cell[ii - 1, jj + 1]) * self.S[ii - 1, jj + 1] +
                             (self.flag_avail_cell[ii + 1, jj + 1]) * self.S[ii + 1, jj + 1])

                    neg_cell = True
                    while denom >= S_ext and neg_cell:
                        neg_cell = False
                        if local_avail[ii - 1, jj]:
                            patch = S_ext * self.S[ii - 1, jj] / denom
                            if self.S_red[ii - 1, jj] - patch <= 0:
                                neg_cell = True
                                local_avail[ii - 1, jj] = False
                        if local_avail[ii + 1, jj]:
                            patch = S_ext * self.S[ii + 1, jj] / denom
                            if self.S_red[ii + 1, jj] - patch <= 0:
                                neg_cell = True
                                local_avail[ii + 1, jj] = False
                        if local_avail[ii, jj - 1]:
                            patch = S_ext * self.S[ii, jj - 1] / denom
                            if self.S_red[ii, jj - 1] - patch <= 0:
                                neg_cell = True
                                local_avail[ii, jj - 1] = False
                        if local_avail[ii, jj + 1]:
                            patch = S_ext * self.S[ii, jj + 1] / denom
                            if self.S_red[ii, jj + 1] - patch <= 0:
                                neg_cell = True
                                local_avail[ii, jj + 1] = False
                        if local_avail[ii - 1, jj - 1]:
                            patch = S_ext * self.S[ii - 1, jj - 1] / denom
                            if self.S_red[ii - 1, jj - 1] - patch <= 0:
                                neg_cell = True
                                local_avail[ii - 1, jj - 1] = False
                        if local_avail[ii + 1, jj - 1]:
                            patch = S_ext * self.S[ii + 1, jj - 1] / denom
                            if self.S_red[ii + 1, jj - 1] - patch <= 0:
                                neg_cell = True
                                local_avail[ii + 1, jj - 1] = False
                        if local_avail[ii - 1, jj + 1]:
                            patch = S_ext * self.S[ii - 1, jj + 1] / denom
                            if self.S_red[ii - 1, jj + 1] - patch <= 0:
                                neg_cell = True
                                local_avail[ii - 1, jj + 1] = False
                        if local_avail[ii + 1, jj + 1]:
                            patch = S_ext * self.S[ii + 1, jj + 1] / denom
                            if self.S_red[ii + 1, jj + 1] - patch <= 0:
                                neg_cell = True
                                local_avail[ii + 1, jj + 1] = False

                        denom = ((local_avail[ii - 1, jj]) * self.S[ii - 1, jj] +
                                 (local_avail[ii + 1, jj]) * self.S[ii + 1, jj] +
                                 (local_avail[ii, jj - 1]) * self.S[ii, jj - 1] +
                                 (local_avail[ii, jj + 1]) * self.S[ii, jj + 1] +
                                 (local_avail[ii - 1, jj - 1]) * self.S[ii - 1, jj - 1] +
                                 (local_avail[ii + 1, jj - 1]) * self.S[ii + 1, jj - 1] +
                                 (local_avail[ii - 1, jj + 1]) * self.S[ii - 1, jj + 1] +
                                 (local_avail[ii + 1, jj + 1]) * self.S[ii + 1, jj + 1])

                    if denom >= S_ext:
                        self.S_enl[ii, jj] = self.S[ii, jj]
                        if local_avail[ii - 1, jj]:
                            patch = S_ext * self.S[ii - 1, jj] / denom
                            self.borrowing[ii, jj].append([ii - 1, jj, patch, None])
                            self.lending[ii - 1, jj].append([ii, jj, patch, None])
                            self.S_enl[ii, jj] += patch
                            self.S_red[ii - 1, jj] -= patch
                        if local_avail[ii + 1, jj]:
                            patch = S_ext * self.S[ii + 1, jj] / denom
                            self.borrowing[ii, jj].append([ii + 1, jj, patch, None])
                            self.lending[ii + 1, jj].append([ii, jj, patch, None])
                            self.S_enl[ii, jj] += patch
                            self.S_red[ii + 1, jj] -= patch
                        if local_avail[ii, jj - 1]:
                            patch = S_ext * self.S[ii, jj - 1] / denom
                            self.borrowing[ii, jj].append([ii, jj - 1, patch, None])
                            self.lending[ii, jj - 1].append([ii, jj, patch, None])
                            self.S_enl[ii, jj] += patch
                            self.S_red[ii, jj - 1] -= patch
                        if local_avail[ii, jj + 1]:
                            patch = S_ext * self.S[ii, jj + 1] / denom
                            self.borrowing[ii, jj].append([ii, jj + 1, patch, None])
                            self.lending[ii, jj + 1].append([ii, jj, patch, None])
                            self.S_enl[ii, jj] += patch
                            self.S_red[ii, jj + 1] -= patch
                        if local_avail[ii - 1, jj - 1]:
                            patch = S_ext * self.S[ii - 1, jj - 1] / denom
                            self.borrowing[ii, jj].append([ii - 1, jj - 1, patch, None])
                            self.lending[ii - 1, jj - 1].append([ii, jj, patch, None])
                            self.S_enl[ii, jj] += patch
                            self.S_red[ii - 1, jj - 1] -= patch
                        if local_avail[ii + 1, jj - 1]:
                            patch = S_ext * self.S[ii + 1, jj - 1] / denom
                            self.borrowing[ii, jj].append([ii + 1, jj - 1, patch, None])
                            self.lending[ii + 1, jj - 1].append([ii, jj, patch, None])
                            self.S_enl[ii, jj] += patch
                            self.S_red[ii + 1, jj - 1] -= patch
                        if local_avail[ii - 1, jj + 1]:
                            patch = S_ext * self.S[ii - 1, jj + 1] / denom
                            self.borrowing[ii, jj].append([ii - 1, jj + 1, patch, None])
                            self.lending[ii - 1, jj + 1].append([ii, jj, patch, None])
                            self.S_enl[ii, jj] += patch
                            self.S_red[ii - 1, jj + 1] -= patch
                        if local_avail[ii + 1, jj + 1]:
                            patch = S_ext * self.S[ii + 1, jj + 1] / denom
                            self.borrowing[ii, jj].append([ii + 1, jj + 1, patch, None])
                            self.lending[ii + 1, jj + 1].append([ii, jj, patch, None])
                            self.S_enl[ii, jj] += patch
                            self.S_red[ii + 1, jj + 1] -= patch

                        self.flag_ext_cell[ii, jj] = False
