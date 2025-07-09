import numpy as np
from typing import Sequence

class FieldMonitor:
    def __init__(self, frequencies: Sequence[float], mesh_type: str, dt: float):
        self.frequencies = np.array(frequencies)  # Frequencies to monitor
        self.mesh_type = mesh_type.lower()  # 'full' or 'sparse'
        self.dt = dt  # Time step
        self.time_index = 0
        self.E_freq = None
        self.indices = None
        self.n_points = None

    def initialize(self, E_flat_shape: int):
        self.n_points = E_flat_shape

        # Set monitored indices
        if self.mesh_type == 'full':
            self.indices = np.arange(self.n_points)
        elif self.mesh_type == 'sparse':
            step = max(self.n_points // 10, 1)
            self.indices = np.arange(0, self.n_points, step)
        else:
            raise ValueError("mesh_type must be 'full' or 'sparse'.")

        # Create empty accumulator: shape [n_freqs, n_monitored_points]
        self.E_freq = np.zeros((len(self.frequencies), len(self.indices)), dtype=np.complex128)

    def update(self, E_flat):
        """
        E_flat: 1D array (lexicographic) of E field at current timestep.
        """
        if self.E_freq is None:
            self.initialize(len(E_flat))

        t = self.time_index * self.dt
        E_monitored = E_flat[self.indices]

        for i, f in enumerate(self.frequencies):
            phase = np.exp(-2j * np.pi * f * t)
            self.E_freq[i] += E_monitored * phase

        self.time_index += 1

    def get_frequency_component(self, freq_index: int):

        return self.E_freq[freq_index]

    def get_all_frequency_components(self):

        return self.E_freq

    def get_monitored_indices(self):

        return self.indices