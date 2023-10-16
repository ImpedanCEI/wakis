import pyvista as pv
import numpy as np
from scipy.constants import c as c_light
import sys
sys.path.append('../../')
from field import Field 

unit = 1e-3
# --- Read stl ----
surf = pv.read('goniometer.stl')
surf = surf.rotate_x(90)    # z axis longitudinal
surf = surf.scale(unit)            # [m]

# --- Domain definition ----

# bounds
xmin, xmax, ymin, ymax, zmin, zmax = surf.bounds

pad = 1.0 * unit

# n cells 
Nx = 30
Ny = 40
Nz = 50
N = Nx*Ny*Nz

# cell vertex
x = np.linspace(xmin - pad, xmax + pad, Nx + 1)
y = np.linspace(ymin - pad, ymax + pad, Ny + 1)
z = np.linspace(zmin - pad, zmax + pad, Nz + 1)

# grid
Z, Y, X = np.meshgrid(z, y, x, indexing='ij')
grid = pv.StructuredGrid(X, Y, Z)

# plot
pl = pv.Plotter()
pl.add_mesh(surf, color='blue', opacity=0.8)
pl.add_mesh(grid, show_edges=True, style='wireframe', opacity=0.45)
pl.show()

# ---- Cells inside surface ----
tol = unit*1.e-3
select = grid.select_enclosed_points(surf, tolerance=tol)
inside = select.threshold(0.1)

#solid1 = np.array(select['SelectedPoints'], dtype=float)
#solid1[solid1 < 0.1] = np.nan

# ---- Plot cells inside ----

#inside.plot(color='blue', opacity=0.85)
#select.plot(show_edges=True, scalars=solid1, opacity=0.55, nan_opacity=0.0)

pl = pv.Plotter()
pl.add_mesh(grid, show_edges=True, style='wireframe', opacity=0.25)
pl.add_mesh(inside, show_edges=True,  color='blue', opacity=0.85)
pl.add_mesh(surf, color='blue', opacity=0.3)

pl.show()