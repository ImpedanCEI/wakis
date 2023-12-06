import numpy as np
import pyvista as pv

from field import Field


class GridFIT3D:
    """
    Class holding the grid information and 
    stl importing handling using PyVista

    Parameters
    ----------
    xmin, xmax, ymin, ymax, zmin, zmax: float
        extent of the domain.
    Nx, Ny, Nz: int
        number of cells per direction
    stl_solids: dict, optional
        stl files to import in the domain.
        {'Solid 1': stl_1, 'Solid 2': stl_2, ...}
        If stl files are not in the same folder,
        add the path to the file name.
    stl_materials: dict, optional
        Material properties associated with stl
        {'Solid 1': [eps1, mu1],
         'Solid 2': [eps1, mu1], 
         ...}
    stl_rotate: list or dict, optional
        Angle of rotation to apply to the stl models: [rot_x, rot_y, rot_z]
        - if list, it will be applied to all stls in `stl_solids`
        - if dict, it must contain the same keys as `stl_solids`, 
          indicating the rotation angle per stl
    stl_scale: float or dict, optional
        Scaling value to apply to the stl model to convert to [m]
        - if float, it will be applied to all stl in `stl_solids`
        - if dict, it must contain the same keys as `stl_solids` 
          
    """

    def __init__(self, xmin, xmax, ymin, ymax, zmin, zmax, 
                Nx, Ny, Nz, stl_solids=None, stl_materials=None, 
                stl_rotate=[0., 0., 0.], stl_translate=[0., 0., 0.], stl_scale=1.0):
        
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

        # Compatibility with FDTD grid obj
        self.nx, self.ny, self.nz = Nx, Ny, Nz
        
        # stl info
        self.stl_solids = stl_solids
        self.stl_materials = stl_materials
        self.stl_rotate = stl_rotate
        self.stl_translate = stl_translate
        self.stl_scale = stl_scale

        # primal Grid G
        self.x = np.linspace(self.xmin, self.xmax, self.Nx+1)
        self.y = np.linspace(self.ymin, self.ymax, self.Ny+1)
        self.z = np.linspace(self.zmin, self.zmax, self.Nz+1)

        # grid
        X, Y, Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        self.grid = pv.StructuredGrid(X.transpose(), Y.transpose(), Z.transpose())

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

        tX, tY, tZ = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        #self.tgrid = pv.StructuredGrid(tX.transpose(), tY.transpose(), tZ.transpose())

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
        
        if stl_solids is not None:
            self.mark_cells_in_stl()

    def mark_cells_in_stl(self):

        if type(self.stl_solids) is not dict:
            if type(self.stl_solids) is str:
                self.stl_solids = {'Solid 1' : self.stl_solids}
            else:
                raise Exception('Attribute `stl_solids` must contain a string or a dictionary')

        if type(self.stl_rotate) is not dict:
            # if not a dict, the same values will be applied to all solids
            stl_rotate = {}
            for key in self.stl_solids.keys():
                stl_rotate[key] = self.stl_rotate
            self.stl_rotate = stl_rotate

        if type(self.stl_scale) is not dict:
            # if not a dict, the same values will be applied to all solids
            stl_scale = {}
            for key in self.stl_solids.keys():
                stl_scale[key] = self.stl_scale
            self.stl_scale = stl_scale

        if type(self.stl_translate) is not dict:
            # if not a dict, the same values will be applied to all solids
            stl_translate = {}
            for key in self.stl_solids.keys():
                stl_translate[key] = self.stl_translate
            self.stl_translate = stl_translate

        tol = np.min([self.dx, self.dy, self.dz])*1e-3
        for key in self.stl_solids.keys():

            surf = self.read_stl(key)

            # mark cells in stl [True == in stl, False == out stl]
            select = self.grid.select_enclosed_points(surf, tolerance=tol)
            self.grid[key] = select.point_data_to_cell_data()['SelectedPoints'] > tol

    def read_stl(self, key):

        # import stl
        surf = pv.read(self.stl_solids[key])

        # rotate
        surf = surf.rotate_x(self.stl_rotate[key][0])  
        surf = surf.rotate_y(self.stl_rotate[key][1])  
        surf = surf.rotate_z(self.stl_rotate[key][2])  

        # translate
        surf = surf.translate(self.stl_translate[key])

        # scale
        surf = surf.scale(self.stl_scale[key]) 

        return surf