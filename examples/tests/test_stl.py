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
#surf = surf.subdivide(3, subfilter='linear') #if used, select.threshold() is empty

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

'''
# plot
pl = pv.Plotter()
pl.add_mesh(surf, show_edges=True, color='blue', opacity=0.6)
pl.add_mesh(grid, show_edges=True, style='wireframe', opacity=0.45)
pl.show()
'''

# ---- Cells inside surface ----
tol = unit*1.e-3
select = grid.select_enclosed_points(surf, tolerance=tol)
inside = select.threshold(0.1)
points_inside = np.where(select['SelectedPoints'] > 0.1)[0]
cells_inside = np.where(select.point_data_to_cell_data()['SelectedPoints'] > 0.1)[0]

grid['Solid1'] = select.point_data_to_cell_data()['SelectedPoints']
#solid1 = np.array(select['SelectedPoints'], dtype=float)
#solid1[solid1 < 0.1] = np.nan


# Plot
pl = pv.Plotter()
pl.add_mesh(grid, show_edges=True, style='wireframe', color='w', opacity=0.15)
#pl.add_mesh(inside, show_edges=True,  color='blue', opacity=0.85)
#pl.add_mesh(select.point_data_to_cell_data().slice(normal=[1,0,0]), scalars='SelectedPoints', cmap='Blues', opacity=0.6)
#pl.add_mesh(select.slice(normal=[1,0,0]), scalars='SelectedPoints', cmap='Blues', opacity=0.6)
#pl.add_mesh(select.threshold(0.1), scalars='SelectedPoints', cmap='Blues', opacity=0.6)
pl.add_mesh(grid.extract_cells(cells_inside), scalars='Solid1', cmap='Blues', opacity=0.6)
#pl.add_mesh(surf, color='blue', opacity=0.35)
pl.show()


'''
# ---- Voxelize and merge test ----
k = 2.0
nx, ny, nz = k*Nx, k*Ny, k*Nz
density = [(xmax-xmin)/nx, (ymax-ymin)/ny, (zmax-zmin)/nz]
voxels = pv.voxelize(surf, density=density)

ngrid = voxels.merge(grid)
ngrid = ngrid.merge()

#ngrid = grid.merge(voxels)

pl = pv.Plotter()
#pl.add_mesh(surf, color='blue', opacity=0.2)
pl.add_mesh(grid, show_edges=True, style='wireframe', color='k', opacity=0.15)
pl.add_mesh(voxels, show_edges=True, style='wireframe', color='red', opacity=0.45)
pl.show()

pl = pv.Plotter()
pl.add_mesh(ngrid.outline(), color='k')
pl.add_mesh(ngrid.slice(normal=[1,0,0]), show_edges=True )
pl.show()

select = ngrid.select_enclosed_points(surf, tolerance=tol)
inside = select.threshold(0.001)
inside.plot()
'''

