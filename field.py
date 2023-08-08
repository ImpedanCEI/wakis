

class Field:
    '''
    Class to switch from 3D to collapsed notation by
    defining the __getitem__ magic method

    linear numbering:
    n = 1 + (i-1) + (j-1)*Nx + (k-1)*Nx*Ny
    len(n) = Nx*Ny*Nz
    '''
    def __init__(self, Nx, Ny, Nz, dtype=float):

        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.dtype = dtype

        self.field = np.zeros((Nx, Ny, Nz), dtype=self.dtype)

    def __getitem__(self, key):

        if type(key) is tuple:
            if len(key) != 3:
                raise ValueError('Need 3 indexes to access the field')
            return self.field[key[0], key[1], key[2]]

        elif type(key) is int:
            i, j, k = self.compute_ijk(key)
            return self.field[i, j, k]

        else:
            raise ValueError('key must be a 3-tuple or an integer')

    def __setitem__(self, key, value):

        if type(key) is tuple:
            if len(key) != 3:
                raise ValueError('Need 3 indexes to access the field')
            self.field[key[0], key[1], key[2]] = value

        elif type(key) is int:
            i, j, k = self.compute_ijk(key)
            self.field[i, j, k] = value

        else:
            raise ValueError('key must be a 3-tuple or an integer')

    def __repr__(self):
        return self.field.__repr__()

    def __str__(self):
        return self.field.__str__()

    def compute_ijk(self, n):
        if n > (self.Nx*self.Ny*self.Nz):
            raise ValueError('Lexico-graphic index cannot be higher than product of dimensions')

        k = n//(self.Nx*self.Ny)
        i = (n-k*self.Nx*self.Ny)%self.Nx
        j = (n-k*self.Nx*self.Ny)//self.Nx

        return i, j, k
