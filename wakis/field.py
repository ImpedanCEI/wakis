# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #


import numpy as xp
import copy

try:
    import cupy as xp_gpu
    imported_cupy = True
except ImportError:
    imported_cupy = False

class Field:
    '''
    Class to switch from 3D to collapsed notation by
    defining the __getitem__ magic method

    linear numbering:
    n = 1 + (i-1) + (j-1)*Nx + (k-1)*Nx*Ny
    len(n) = Nx*Ny*Nz
    '''
    def __init__(self, Nx, Ny, Nz, dtype=float, 
                 use_ones=False, use_gpu=False):

        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.N = Nx*Ny*Nz
        self.dtype = dtype
        self.on_gpu = use_gpu

        if self.on_gpu:
            if imported_cupy:
                self.xp = xp_gpu
            else:
                print('*** cupy could not be imported, please CUDA check installation')
        else:
            self.xp = xp
            
        if use_ones:
            self.array = self.xp.ones(self.N*3, dtype=self.dtype, order='F')
        else:
            self.array = self.xp.zeros(self.N*3, dtype=self.dtype, order='F')

    @property
    def field_x(self):
        return self.array[0:self.N]

    @property
    def field_y(self):
        return self.array[self.N: 2*self.N]
    
    @property
    def field_z(self):
        return self.array[2*self.N:3*self.N]

    @field_x.setter
    def field_x(self, value):
        if len(value.shape) > 1:
            self.from_matrix(value, 'x')
        else:
            self.array[0:self.N] = value

    @field_y.setter
    def field_y(self, value):
        if len(value.shape) > 1:
            self.from_matrix(value, 'y')
        else:
            self.array[self.N: 2*self.N] = value

    @field_z.setter
    def field_z(self, value):
        if len(value.shape) > 1:
            self.from_matrix(value, 'z')
        else:
            self.array[2*self.N:3*self.N] = value

    def toarray(self):
        return self.array

    def fromarray(self, array):
        self.array = array

    def to_matrix(self, key):
        if key == 0 or key == 'x':
            return self.xp.reshape(self.array[0:self.N], (self.Nx, self.Ny, self.Nz), order='F')
        if key == 1 or key == 'y':
            return self.xp.reshape(self.array[self.N: 2*self.N], (self.Nx, self.Ny, self.Nz), order='F')
        if key == 2 or key == 'z':
            return self.xp.reshape(self.array[2*self.N:3*self.N], (self.Nx, self.Ny, self.Nz), order='F')

    def from_matrix(self, mat, key):
        if key == 0 or key == 'x':
            self.array[0:self.N] = self.xp.reshape(mat, self.N, order='F')
        elif key == 1 or key == 'y':
            self.array[self.N: 2*self.N] = self.xp.reshape(mat, self.N, order='F')
        elif key == 2 or key == 'z':
            self.array[2*self.N:3*self.N] = self.xp.reshape(mat, self.N, order='F')
        else:
            raise IndexError('Component id not valid')
    
    def __getitem__(self, key):

        if type(key) is tuple:
            if len(key) != 4:
                raise IndexError('Need 3 indexes and component to access the field')
            if key[3] == 0 or key[3] == 'x':
                if self.on_gpu:
                    field = self.xp.reshape(self.array[0:self.N], (self.Nx, self.Ny, self.Nz), order='F')
                    return field[key[0], key[1], key[2]].get()
                else:
                    field = self.xp.reshape(self.array[0:self.N], (self.Nx, self.Ny, self.Nz), order='F')
                    return field[key[0], key[1], key[2]]
            elif key[3] == 1 or key[3] == 'y':
                if self.on_gpu:
                    field = self.xp.reshape(self.array[self.N: 2*self.N], (self.Nx, self.Ny, self.Nz), order='F')
                    return field[key[0], key[1], key[2]].get()
                else:
                    field = self.xp.reshape(self.array[self.N: 2*self.N], (self.Nx, self.Ny, self.Nz), order='F')
                    return field[key[0], key[1], key[2]]
            elif key[3] == 2 or key[3] == 'z':
                if self.on_gpu:
                    field = self.xp.reshape(self.array[2*self.N:3*self.N], (self.Nx, self.Ny, self.Nz), order='F')
                    return field[key[0], key[1], key[2]].get()
                else:
                    field = self.xp.reshape(self.array[2*self.N:3*self.N], (self.Nx, self.Ny, self.Nz), order='F')
                    return field[key[0], key[1], key[2]]
            else:
                raise IndexError('Component id not valid')

        elif type(key) is int:
            if key <= self.N:
                if self.on_gpu:
                    return self.array[key].get()
                else:
                    return self.array[key]
            else:
                raise IndexError('Lexico-graphic index cannot be higher than product of dimensions')
            
        elif type(key) is slice:
            if self.on_gpu:
                return self.array[key].get()
            else:
                return self.array[key]

        else:
            raise ValueError('key must be a 3-tuple or an integer')

    def __setitem__(self, key, value):

        if self.on_gpu:
            value = self.xp.asarray(value)

        if type(key) is tuple:
            if len(key) != 4:
                raise IndexError('Need 3 indexes and component to access the field')
            else:
                field = self.to_matrix(key[3])
                field[key[0], key[1], key[2]] = value
                self.from_matrix(field, key[3])

        elif type(key) is int:
            if key <= self.N:
                self.array[key] = value
            else:
                raise IndexError('Lexico-graphic index cannot be higher than product of dimensions')
            
        elif type(key) is slice:
                self.array[key] = value   
        else:
            raise IndexError('key must be a 3-tuple or an integer')

    def __mul__(self, other, dtype=None):

        if dtype is None:
            dtype = self.dtype

        # other is number
        if type(other) is float or type(other) is int:
            mulField = Field(self.Nx, self.Ny, self.Nz, dtype=dtype)
            mulField.array = self.array * other

        # other is matrix 
        elif len(other.shape) > 1:
            mulField = Field(self.Nx, self.Ny, self.Nz, dtype=dtype)
            for d in ['x', 'y', 'z']:
                mulField.from_matrix(self.to_matrix(d) * other, d)

        # other is 1d array 
        else:
            mulField = Field(self.Nx, self.Ny, self.Nz, dtype=dtype)
            mulField.array = self.array * other

        return mulField

    def __div__(self, other, dtype=None):

        if dtype is None:
            dtype = self.dtype

        # other is number
        if type(other) is float or type(other) is int:
            divField = Field(self.Nx, self.Ny, self.Nz, dtype=dtype)
            divField.array = self.array / other

        # other is matrix
        if len(other.shape) > 1:
            divField = Field(self.Nx, self.Ny, self.Nz, dtype=dtype)
            for d in ['x', 'y', 'z']:
                divField.from_matrix(self.to_matrix(d) / other, d)

        # other is constant or 1d array 
        else:
            divField = Field(self.Nx, self.Ny, self.Nz, dtype=dtype)
            divField.array = self.array / other

        return divField

    def __add__(self, other, dtype=None):

        if dtype is None:
            dtype = self.dtype
        
        if type(other) is Field:
            
            addField = Field(self.Nx, self.Ny, self.Nz, dtype=dtype)
            addField.field_x = self.field_x + other.field_x
            addField.field_y = self.field_y + other.field_y
            addField.field_z = self.field_z + other.field_z  
            
        # other is matrix 
        elif len(other.shape) > 1:
            addField = Field(self.Nx, self.Ny, self.Nz, dtype=dtype)
            for d in ['x', 'y', 'z']:
                addField.from_matrix(self.to_matrix(d) + other, d)

        # other is constant or 1d array 
        else:
            addField = Field(self.Nx, self.Ny, self.Nz, dtype=dtype)
            addField.array = self.array + other

        return addField

    def __repr__(self):
        return 'x:\n' + self.field_x.__repr__() + '\n'+  \
                'y:\n' + self.field_y.__repr__() + '\n'+ \
                'z:\n' + self.field_z.__repr__()

    def __str__(self):
        return 'x:\n' + self.field_x.__str__() + '\n'+  \
                'y:\n' + self.field_y.__str__() + '\n'+ \
                'z:\n' + self.field_z.__str__()
    
    def copy(self):
        import copy
        obj = type(self).__new__(self.__class__)  # Create empty instance

        for key, value in self.__dict__.items():
            if key == "xp":
                obj.xp = self.xp  # Just copy reference, no need for deepcopy
            elif key == "array" and self.on_gpu:
                obj.array = self.xp.array(self.array)  # Ensure CuPy array is copied properly
            else:
                obj.__dict__[key] = copy.deepcopy(value)

        return obj

    def compute_ijk(self, n):
        if n > (self.N):
            raise IndexError('Lexico-graphic index cannot be higher than product of dimensions')

        k = n//(self.Nx*self.Ny)
        i = (n-k*self.Nx*self.Ny)%self.Nx
        j = (n-k*self.Nx*self.Ny)//self.Nx

        return i, j, k

    def get_abs(self, as_matrix=True):
        '''Computes the absolute or magnitude
        out of the field components
        '''
        if as_matrix:
            if self.on_gpu:
                return xp.sqrt(self.to_matrix('x')**2 + 
                            self.to_matrix('y')**2 + 
                            self.to_matrix('z')**2 ).get()
            else:
                return xp.sqrt(self.to_matrix('x')**2 + 
                            self.to_matrix('y')**2 + 
                            self.to_matrix('z')**2 )

        else: # 1d array
            self.fill_components()
            if self.on_gpu:
                return xp.sqrt(self.field_x**2 + self.field_y**2, self.field_z**2).get()
            else:
                return xp.sqrt(self.field_x**2 + self.field_y**2, self.field_z**2)

    def inspect(self, plane='YZ', cmap='bwr', dpi=100, figsize=[8,6], x=None, y=None, z=None, show=True, handles=False, **kwargs):
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        if None not in (x,y,z):
            pass
        elif plane == 'XY':
            key=[slice(0,self.Nx), slice(0,self.Ny), int(self.Nz//2)]
            x, y, z = key[0], key[1], key[2]
            extent = (0, self.Nx, 0, self.Ny)
            xax, yax = 'nx', 'ny'
            transpose = True

        elif plane == 'XZ':
            key=[slice(0,self.Nx), int(self.Ny//2), slice(0,self.Nz)]
            x, y, z = key[0], key[1], key[2]
            extent = (0, self.Nz, 0, self.Nx)
            xax, yax = 'nz', 'nx'
            transpose = False

        elif plane == 'YZ':
            key=[int(self.Nx//2), slice(0,self.Ny), slice(0,self.Nz)]
            x, y, z = key[0], key[1], key[2]
            extent = (0, self.Nz, 0, self.Ny)
            xax, yax = 'nz', 'ny'
            transpose = False

        fig, axs = plt.subplots(1, 3, tight_layout=True, figsize=figsize, dpi=dpi)
        dims = {0:'x', 1:'y', 2:'z'}

        im = self.xp.zeros_like(axs)

        for d in [0,1,2]:
            field = self.to_matrix(d)
            
            if self.on_gpu:
                field = field.get()

            if transpose:
                im[d] = axs[d].imshow(field[x,y,z].T, cmap=cmap, vmin=-field.max(), vmax=field.max(), extent=extent, origin='lower', **kwargs)

            else:
                im[d] = axs[d].imshow(field[x,y,z], cmap=cmap, vmin=-field.max(), vmax=field.max(), extent=extent, origin='lower', **kwargs)

        for i, ax in enumerate(axs):
            ax.set_title(f'Field {dims[i]}, plane {plane}')
            fig.colorbar(im[i], cax=make_axes_locatable(ax).append_axes('right', size='5%', pad=0.1))
            ax.set_xlabel(xax)
            ax.set_ylabel(yax)

        if handles:
            return fig, axs
        
        if show:
            plt.show()

    def inspect3D(self, field='all', xmax=None, ymax=None, zmax=None, cmap='bwr', dpi=100, show=True, handles=False):
        """
        Voxel representation of a 3D array with matplotlib
        [TODO] use pyvista instead
        """
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        fig = plt.figure(tight_layout=True, dpi=dpi, figsize=[12,6])

        plot_x, plot_y, plot_z = False, False, False

        if field == 'all':
            plot_x = True
            plot_y = True
            plot_z = True

        elif field.lower() == 'x': plot_x = True
        elif field.lower() == 'y': plot_y = True
        elif field.lower() == 'z': plot_z = True

        if xmax is None: xmax = self.Nx
        if ymax is None: ymax = self.Ny
        if zmax is None: zmax = self.Nz

        x,y,z = self.xp.mgrid[0:xmax+1,0:ymax+1,0:zmax+1]
        axs = []

        # field x
        if plot_x:
            arr = self.to_matrix('x')[0:int(xmax),0:int(ymax),0:int(zmax)]
            if field == 'all':
                ax = fig.add_subplot(1, 3, 1, projection='3d')
            else:
                ax = fig.add_subplot(1, 1, 1, projection='3d')
            
            vmin, vmax = -self.xp.max(self.xp.abs(arr)), +self.xp.max(self.xp.abs(arr))
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            colors = mpl.colormaps[cmap](norm(arr))
            vox = ax.voxels(x, y, z, filled=self.xp.ones_like(arr), facecolors=colors)
            
            m = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
            m.set_array([])
            #fig.colorbar(m, shrink=0.5, aspect=10)
            ax.set_title(f'Field x')
            axs.append(ax)

        # field y
        if plot_y:
            arr = self.to_matrix('y')[0:int(xmax),0:int(ymax),0:int(zmax)]
            if field == 'all':
                ax = fig.add_subplot(1, 3, 2, projection='3d')
            else:
                ax = fig.add_subplot(1, 1, 1, projection='3d')
            
            vmin, vmax = -self.xp.max(self.xp.abs(arr)), +self.xp.max(self.xp.abs(arr))
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            colors = mpl.colormaps[cmap](norm(arr))
            vox = ax.voxels(x, y, z, filled=self.xp.ones_like(arr), facecolors=colors)
            
            m = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
            m.set_array([])
            #fig.colorbar(m, shrink=0.5, aspect=10)
            ax.set_title(f'Field y')
            axs.append(ax)

        # field z
        if plot_z:
            arr = self.to_matrix('z')[0:int(xmax),0:int(ymax),0:int(zmax)]
            if field == 'all':
                ax = fig.add_subplot(1, 3, 3, projection='3d')
            else:
                ax = fig.add_subplot(1, 1, 1, projection='3d')

            vmin, vmax = -self.xp.max(self.xp.abs(arr)), +self.xp.max(self.xp.abs(arr))
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            colors = mpl.colormaps[cmap](norm(arr))
            vox = ax.voxels(x, y, z, filled=self.xp.ones_like(arr), facecolors=colors)
            
            m = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
            m.set_array([])
            #fig.colorbar(m, shrink=0.5, aspect=10)
            ax.set_title(f'Field z')
            axs.append(ax)

        dims = {0:'x', 1:'y', 2:'z'}
        for i, ax in enumerate(axs):
            ax.set_xlabel('nx')
            ax.set_ylabel('ny')
            ax.set_zlabel('nz')
            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.set_xlim(0,self.Nx)
            ax.set_ylim(0,self.Ny)
            ax.set_zlim(0,self.Nz)

        if handles:
            return fig, axs
        
        if show:
            plt.show()
