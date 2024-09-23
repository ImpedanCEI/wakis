import os, sys
import pytest 

def test_dependency_imports():
    import scipy 
    import numpy 
    import pyvista
    import h5py
    from tqdm import tqdm

def test_module_imports():
    sys.path.append('../wakis')

    from wakis import SolverFIT3D
    from wakis import GridFIT3D 
    from wakis import WakeSolver
    from wakis import Field

    from wakis.materials import material_lib
    from wakis.sources import Beam
    from wakis.sources import PlaneWave
    from wakis.sources import Pulse
    from wakis.sources import WavePacket


