import numpy as np

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
        self.N = Nx*Ny*Nz
        self.dtype = dtype

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

    def __repr__(self):
        return ['x:\n' + self.field_x.__repr__() + '\n'+  
                'y:\n' + self.field_y.__repr__() + '\n'+
                'z:\n' + self.field_z.__repr__()]

    def __str__(self):
        return ['x:\n' + self.field_x.__str__() + '\n'+  
                'y:\n' + self.field_y.__str__() + '\n'+
                'z:\n' + self.field_z.__str__()]

    def compute_ijk(self, n):
        if n > (self.N):
            raise IndexError('Lexico-graphic index cannot be higher than product of dimensions')

        k = n//(self.Nx*self.Ny)
        i = (n-k*self.Nx*self.Ny)%self.Nx
        j = (n-k*self.Nx*self.Ny)//self.Nx

        return i, j, k

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