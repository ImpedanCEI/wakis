import numpy as np

class Field:
    '''
    Class to switch from 3D to collapsed notation by
    defining the __getitem__ magic method

    linear numbering:
    n = 1 + (i-1) + (j-1)*Nx + (k-1)*Nx*Ny
    len(n) = Nx*Ny*Nz
    '''
    def __init__(self, Nx, Ny, Nz, dtype=float, use_ones=False):

        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.N = Nx*Ny*Nz
        self.dtype = dtype

        if use_ones:
            self.field_x = np.ones((Nx, Ny, Nz), dtype=self.dtype)
            self.field_y = np.ones((Nx, Ny, Nz), dtype=self.dtype)
            self.field_z = np.ones((Nx, Ny, Nz), dtype=self.dtype)

        else:
            self.field_x = np.zeros((Nx, Ny, Nz), dtype=self.dtype)
            self.field_y = np.zeros((Nx, Ny, Nz), dtype=self.dtype)
            self.field_z = np.zeros((Nx, Ny, Nz), dtype=self.dtype)

    def __getitem__(self, key):

        if type(key) is tuple:
            if len(key) != 4:
                raise IndexError('Need 3 indexes and component to access the field')
            if key[3] == 0 or key[3] == 'x':
                return self.field_x[key[0], key[1], key[2]]
            elif key[3] == 1 or key[3] == 'y':
                return self.field_y[key[0], key[1], key[2]]
            elif key[3] == 2 or key[3] == 'z':
                return self.field_z[key[0], key[1], key[2]]
            else:
                raise IndexError('Component id not valid')

        elif type(key) is int:
            if key <= self.N:
                field = self.field_x
            elif self.N < key <= 2*self.N:
                field = self.field_y
                key -= self.N
            elif 2*self.N < key <= 3*self.N:
                field = self.field_z
                key -= 2*self.N
            else:
                raise IndexError('Lexico-graphic index cannot be higher than product of dimensions')

            i, j, k = self.compute_ijk(key)
            return field[i, j, k]

        else:
            raise ValueError('key must be a 3-tuple or an integer')

    def __setitem__(self, key, value):

        if type(key) is tuple:
            if len(key) != 4:
                raise IndexError('Need 3 indexes and component to access the field')
            if key[3] == 0 or key[3] == 'x':
                self.field_x[key[0], key[1], key[2]] = value
            elif key[3] == 1 or key[3] == 'y':
                self.field_y[key[0], key[1], key[2]] = value
            elif key[3] == 2 or key[3] == 'z':
                self.field_z[key[0], key[1], key[2]] = value
            else:
                raise IndexError('Component id not valid')

        elif type(key) is int:
            if key <= self.N:
                field = self.field_x
            elif self.N < key <= 2*self.N:
                field = self.field_y
                key -= self.N
            elif 2*self.N < key <= 3*self.N:
                field = self.field_z
                key -= 2*self.N
            else:
                raise IndexError('Lexico-graphic index cannot be higher than product of dimensions')

            i, j, k = self.compute_ijk(key)
            field[i, j, k] = value

        else:
            raise IndexError('key must be a 3-tuple or an integer')

    def __mul__(self, other, dtype=None):

        if dtype is None:
            dtype = self.dtype

        mulField = Field(self.Nx, self.Ny, self.Nz, dtype=dtype)
        mulField.field_x = self.field_x * other
        mulField.field_y = self.field_y * other
        mulField.field_z = self.field_z * other

        return mulField

    def __div__(self, other, dtype=None):

        if dtype is None:
            dtype = self.dtype
            
        mulField = Field(self.Nx, self.Ny, self.Nz, dtype=dtype)
        mulField.field_x = self.field_x / other
        mulField.field_y = self.field_y / other
        mulField.field_z = self.field_z / other

        return mulField

    def __add__(self, other, dtype=None):

        if dtype is None:
            dtype = self.dtype
        
        if type(other) is Field:
            mulField = Field(self.Nx, self.Ny, self.Nz, dtype=dtype)
            mulField.field_x = self.field_x + other.field_x
            mulField.field_y = self.field_y + other.field_y
            mulField.field_z = self.field_z + other.field_z  
        else:
            mulField = Field(self.Nx, self.Ny, self.Nz, dtype=dtype)
            mulField.field_x = self.field_x + other
            mulField.field_y = self.field_y + other
            mulField.field_z = self.field_z + other

        return mulField

    def __repr__(self):
        return 'x:\n' + self.field_x.__repr__() + '\n'+  \
                'y:\n' + self.field_y.__repr__() + '\n'+ \
                'z:\n' + self.field_z.__repr__()

    def __str__(self):
        return 'x:\n' + self.field_x.__str__() + '\n'+  \
                'y:\n' + self.field_y.__str__() + '\n'+ \
                'z:\n' + self.field_z.__str__()

    def toarray(self):
        return np.concatenate((
                np.reshape(self.field_x, self.N, order='F'),
                np.reshape(self.field_y, self.N, order='F'),
                np.reshape(self.field_z, self.N, order='F')
            ))

    def fromarray(self, arr):
        if len(arr.shape) > 1:
            raise ValueError('Can only assign 1d array to a Field')
        if len(arr) != 3*self.N:
            raise ValueError('Can only assign to a field an array of size 3*Nx*Ny*Nz')
        self.field_x = np.reshape(arr[0:self.N], (self.Nx, self.Ny, self.Nz), order='F')
        self.field_y = np.reshape(arr[self.N: 2*self.N], (self.Nx, self.Ny, self.Nz), order='F')
        self.field_z = np.reshape(arr[2*self.N:3*self.N], (self.Nx, self.Ny, self.Nz), order='F')

    def compute_ijk(self, n):
        if n > (self.N):
            raise IndexError('Lexico-graphic index cannot be higher than product of dimensions')

        k = n//(self.Nx*self.Ny)
        i = (n-k*self.Nx*self.Ny)%self.Nx
        j = (n-k*self.Nx*self.Ny)//self.Nx

        return i, j, k

    def get_abs(self):
        return np.sqrt(self.field_x**2 + self.field_y**2, self.field_z**2)

    def inspect(self, plane='XY', cmap='bwr', dpi=100, x=None, y=None, z=None, show=True, handles=False):
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        if None not in (x,y,z):
            pass
        elif plane == 'XY':
            key=[slice(0,self.Nx), slice(0,self.Ny), int(self.Nz//2)]
            x, y, z = key[0], key[1], key[2]
            extent = (0, self.Nx, 0, self.Ny)
            xax, yax = 'ny', 'nx'

        elif plane == 'XZ':
            key=[slice(0,self.Nx), int(self.Ny//2), slice(0,self.Nz)]
            x, y, z = key[0], key[1], key[2]
            extent = (0, self.Nx, 0, self.Nz)
            xax, yax = 'nz', 'nx'
        elif plane == 'YZ':
            key=[int(self.Nx//2), slice(0,self.Ny), slice(0,self.Nz)]
            x, y, z = key[0], key[1], key[2]
            extent = (0, self.Ny, 0, self.Nz)
            xax, yax = 'nz', 'ny'

        fig, axs = plt.subplots(1, 3, tight_layout=True, figsize=[8,6], dpi=dpi)
        dims = {0:'x', 1:'y', 2:'z'}

        im = np.zeros_like(axs)

        im[0] = axs[0].imshow(self.field_x[x,y,z], cmap=cmap, vmin=-np.max(np.max(self.field_x)), vmax=np.max(np.max(self.field_x)), extent=extent)
        im[1] = axs[1].imshow(self.field_y[x,y,z], cmap=cmap, vmin=-np.max(np.max(self.field_y)), vmax=np.max(np.max(self.field_y)), extent=extent)
        im[2] = axs[2].imshow(self.field_z[x,y,z], cmap=cmap, vmin=-np.max(np.max(self.field_z)), vmax=np.max(np.max(self.field_z)), extent=extent)

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

        x,y,z = np.mgrid[0:xmax+1,0:ymax+1,0:zmax+1]
        axs = []

        # field x
        if plot_x:
            arr = self.field_x[0:int(xmax),0:int(ymax),0:int(zmax)]
            if field == 'all':
                ax = fig.add_subplot(1, 3, 1, projection='3d')
            else:
                ax = fig.add_subplot(1, 1, 1, projection='3d')
            
            vmin, vmax = -np.max(np.abs(arr)), +np.max(np.abs(arr))
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            colors = mpl.colormaps[cmap](norm(arr))
            vox = ax.voxels(x, y, z, filled=np.ones_like(arr), facecolors=colors)
            
            m = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
            m.set_array([])
            fig.colorbar(m, shrink=0.5, aspect=10)
            ax.set_title(f'Field x')
            axs.append(ax)

        # field y
        if plot_y:
            arr = self.field_y[0:int(xmax),0:int(ymax),0:int(zmax)]
            if field == 'all':
                ax = fig.add_subplot(1, 3, 2, projection='3d')
            else:
                ax = fig.add_subplot(1, 1, 1, projection='3d')
            
            vmin, vmax = -np.max(np.abs(arr)), +np.max(np.abs(arr))
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            colors = mpl.colormaps[cmap](norm(arr))
            vox = ax.voxels(x, y, z, filled=np.ones_like(arr), facecolors=colors)
            
            m = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
            m.set_array([])
            fig.colorbar(m, shrink=0.5, aspect=10)
            ax.set_title(f'Field y')
            axs.append(ax)

        # field z
        if plot_z:
            arr = self.field_z[0:int(xmax),0:int(ymax),0:int(zmax)]
            if field == 'all':
                ax = fig.add_subplot(1, 3, 3, projection='3d')
            else:
                ax = fig.add_subplot(1, 1, 1, projection='3d')

            vmin, vmax = -np.max(np.abs(arr)), +np.max(np.abs(arr))
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            colors = mpl.colormaps[cmap](norm(arr))
            vox = ax.voxels(x, y, z, filled=np.ones_like(arr), facecolors=colors)
            
            m = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
            m.set_array([])
            fig.colorbar(m, shrink=0.5, aspect=10)
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

class Field2D:
    '''
    Class to switch from 2D to collapsed notation by
    defining the __getitem__ magic method

    linear numbering:
    n = 1 + (i-1) + (j-1)*Nx
    len(n) = Nx*Ny
    '''
    def __init__(self, Nx, Ny, dtype=float):

        self.Nx = Nx
        self.Ny = Ny
        self.N = Nx*Ny
        self.dtype = dtype

        self.field = np.zeros((Nx, Ny), dtype=self.dtype)

    def __getitem__(self, key):

        if type(key) is tuple:
            if len(key) != 2:
                raise ValueError('Need 3 indexes to access the field')
            return self.field[key[0], key[1]]

        elif type(key) is int:
            i, j = self.compute_ij(key)
            return self.field[i, j]

        else:
            raise ValueError('key must be a 3-tuple or an integer')

    def __setitem__(self, key, value):

        if type(key) is tuple:
            if len(key) != 2:
                raise ValueError('Need 3 indexes to access the field')
            self.field[key[0], key[1]] = value

        elif type(key) is int:
            i, j = self.compute_ij(key)
            self.field[i, j] = value

        else:
            raise ValueError('key must be a 3-tuple or an integer')

    def __repr__(self):
        return self.field.__repr__()

    def __str__(self):
        return self.field.__str__()

    def compute_ij(self, n):
        if n > (self.N):
            raise ValueError('Lexico-graphic index cannot be higher than product of dimensions')

        i = n%self.Nx
        j = n//self.Nx

        return i, j