# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #


import numpy as np
from typing import Sequence
from wakis.field import Field

try:
    import cupy as xp_gpu
    imported_cupy = True
except ImportError:
    imported_cupy = False


class FieldMonitor:
    """
    Accumulates frequency-domain electric field components over time.

    Attributes:
        frequencies (np.ndarray): List of frequencies to monitor (Hz).
        time_index (int): Current time step index.
        dt (float): Time step size.
        Ex_acc, Ey_acc, Ez_acc: Accumulators for each field component.
        shape (tuple): Shape of the field (Nx, Ny, Nz).
        xp (module): Backend array library (NumPy or CuPy).
    """
    def __init__(self, frequencies: Sequence[float]):
        """
       Args:
           frequencies (Sequence[float]): Frequencies at which to monitor the field (in Hz).
           should be pre-computed.
       """
        self.frequencies = np.array(frequencies)
        self.time_index = 0
        self.dt = None
        self.Ex_acc = None
        self.Ey_acc = None
        self.Ez_acc = None
        self.shape = None
        self.xp = None


    def update(self, E: Field, dt: float):
        """
        Updates field accumulators with new field snapshot at the current time.

        Args:
            E (Field): Electric field object containing components.
            dt (float): Time step size.
        """
        if self.dt is None:
            self.dt = dt
            self.shape = (E.Nx, E.Ny, E.Nz)
            self.xp = E.xp

            n_freqs = len(self.frequencies)
            shape = (n_freqs, *self.shape)
            self.Ex_acc = self.xp.zeros(shape, dtype=self.xp.complex128)
            self.Ey_acc = self.xp.zeros(shape, dtype=self.xp.complex128)
            self.Ez_acc = self.xp.zeros(shape, dtype=self.xp.complex128)
            t = self.time_index * self.dt

        t = self.time_index * self.dt

        Ex = E.to_matrix('x') #E.array[0:E.N]
        Ey = E.to_matrix('y')
        Ez = E.to_matrix('z')

        for i, f in enumerate(self.frequencies):
            phase = self.xp.exp(-2j * self.xp.pi * f * t)
            self.Ex_acc[i] += Ex * phase
            self.Ey_acc[i] += Ey * phase
            self.Ez_acc[i] += Ez * phase

        self.time_index += 1

    def get_components(self):
        """
        Returns the accumulated frequency-domain field components.
        """
        return {
            'Ex': self.Ex_acc,
            'Ey': self.Ey_acc,
            'Ez': self.Ez_acc
        }
