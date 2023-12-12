import pyvista as pv
import numpy as np
from scipy.constants import c as c_light
import sys
sys.path.append('../../')
from field import Field 

unit = 1e-3
# --- Read stl ----
surf = pv.read('../stl/goniometer.stl')
surf = surf.rotate_x(90)    # z axis longitudinal
surf = surf.scale(unit)            # [m]
#surf = surf.subdivide(3, subfilter='linear') #if used, select.threshold() is empty

# --- Domain definition ----

# bounds
xmin, xmax, ymin, ymax, zmin, zmax = surf.bounds
pad = 1.0 * unit

# n cells 
Nx = 30*2
Ny = 40*2
Nz = 50*2
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
points_inside = np.where(select['SelectedPoints'] > 0.1)[0]
cells_inside = np.where(select.point_data_to_cell_data()['SelectedPoints'] > 0.1)[0]

grid['Solid1'] = select.point_data_to_cell_data()['SelectedPoints']
#solid1 = np.array(select['SelectedPoints'], dtype=float)
#solid1[solid1 < 0.1] = np.nan

inside = select.threshold(0.1)
# Plot
pl = pv.Plotter()
pl.add_mesh(grid, show_edges=True, style='wireframe', color='w', opacity=0.15)
pl.add_mesh(grid.extract_cells(cells_inside), scalars='Solid1', cmap='Blues', opacity=0.6)

pl.show()

#pl.add_mesh(inside, show_edges=True,  color='blue', opacity=0.85)
#pl.add_mesh(select.point_data_to_cell_data().slice(normal=[1,0,0]), scalars='SelectedPoints', cmap='Blues', opacity=0.6)
#pl.add_mesh(select.slice(normal=[1,0,0]), scalars='SelectedPoints', cmap='Blues', opacity=0.6)
#pl.add_mesh(select.threshold(0.1), scalars='SelectedPoints', cmap='Blues', opacity=0.6)

#pl.add_mesh(surf, color='blue', opacity=0.35)

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
# ---- Cells_inside to Field slices ----
'''
# ---- Aux functions ------
m = 1
n = 1
p = 1
theta = 0 #np.pi/8

def analytic_sol_Hz(x, y, z, t):
    Rm = np.array([[np.cos(-theta), - np.sin(-theta), 0],[np.sin(-theta), np.cos(-theta), 0], [0, 0, 1]])
    [x_0, y_0, z_0] = np.dot(Rm, np.array([x, y, z]))

    return np.cos(m * np.pi / Lx * (x_0 - Lx/2)) * np.cos(n * np.pi / Ly * (y_0 - Ly/2)) * np.sin(
        p * np.pi / Lz * (z_0 - Lz/2)) * np.cos(np.sqrt(2) * np.pi / Lx * c_light * t)

def analytic_sol_Hy(x, y, z, t):
    Rm = np.array([[np.cos(-theta), - np.sin(-theta), 0],[np.sin(-theta), np.cos(-theta), 0], [0, 0, 1]])
    [x_0, y_0, z_0] = np.dot(Rm, np.array([x, y, z]))
    h_2 = (m * np.pi / Lx) ** 2 + (n * np.pi / Ly) ** 2 + (p * np.pi / Lz) ** 2

    return -2 / h_2 * (n * np.pi / Ly) * (p * np.pi / Lz) * np.cos(m * np.pi / Lx * (x_0 - Lx/2)) * np.sin(
        n * np.pi / Ly * (y_0 - Ly/2)) * np.cos(p * np.pi / Lz * (z_0 - Lz/2)) * np.cos(
        np.sqrt(2) * np.pi / Lx * c_light * t)

def analytic_sol_Hx(x, y, z, t):
    Rm = np.array([[np.cos(-theta), - np.sin(-theta), 0],[np.sin(-theta), np.cos(-theta), 0], [0, 0, 1]])
    [x_0, y_0, z_0] = np.dot(Rm, np.array([x, y, z]))
    h_2 = (m * np.pi / Lx) ** 2 + (n * np.pi / Ly) ** 2 + (p * np.pi / Lz) ** 2

    return -2 / h_2 * (m * np.pi / Lx) * (p * np.pi / Lz) * np.sin(m * np.pi / Lx * (x_0 - Lx/2)) * np.cos(
        n * np.pi / Ly * (y_0 - Ly/2)) * np.cos(p * np.pi / Lz * (z_0 - Lz/2)) * np.cos(
        np.sqrt(2) * np.pi / Lx * c_light * t)


# ---- Add Scalar data ----
analyticH = Field(Nx, Ny, Nz)
dx, dy, dz = x[1]-x[0], y[1]-y[0], z[1]-z[0]
Lx, Ly, Lz = (xmax-xmin), (ymax-ymin), (zmax-zmin),

dt = 1.0 / (c_light * np.sqrt(1 / dx ** 2 + 1 / dy ** 2 + 1 / dz ** 2))
Nt = 10

for ii in range(Nx):
    for jj in range(Ny):
        for kk in range(Nz):

            x = (ii+0.5) * dx + xmin
            y = (jj+0.5) * dy + ymin
            z = (kk+0.5) * dz + zmin
            analyticH[ii, jj, kk, 'z'] = analytic_sol_Hz(x, y, z, (Nt-0.5) * dt)

            x = (ii+0.5) * dx + xmin
            y = (jj+0.5) * dy + ymin
            z = (kk+0.5) * dz + zmin
            analyticH[ii, jj, kk, 'y'] = analytic_sol_Hy(x, y, z, (Nt-0.5) * dt)

            x = (ii+0.5) * dx + xmin
            y = (jj+0.5) * dy + ymin
            z = (kk+0.5) * dz + zmin
            analyticH[ii, jj, kk, 'x'] = analytic_sol_Hx(x, y, z, (Nt-0.5) * dt)

grid.cell_data['Hz'] = np.reshape(analyticH[:, :, :, 'z'], N)
grid.cell_data['Hy'] = np.reshape(analyticH[:, :, :, 'y'], N)
grid.cell_data['Hx'] = np.reshape(analyticH[:, :, :, 'x'], N)
#grid.plot(smooth_shading=True, show_edges=True, scalars='Hz', cmap='rainbow')

# ----- Clip stl cells field to 0.0 -------

n = select.point_data_to_cell_data()['SelectedPoints'] > 0.1
mask = np.reshape(np.logical_not(n), (Nx, Ny, Nz)).astype(int)
#mask = np.reshape(n, (Nx, Ny, Nz)).astype(int)*100

# clipping field obj
aux = analyticH[:,:,:, 'z']*mask
grid.cell_data['Hz'] = np.reshape(aux, N)

pl = pv.Plotter()
pl.add_mesh(grid, show_edges=True, style='wireframe', color='w', opacity=0.15)
pl.add_mesh(grid.slice(normal=[1,0,0]), show_edges=True, scalars='Hz', cmap='rainbow')
pl.show()
'''

'''
# clipping grid obj
grid.cell_data['Hz'][cells_inside] = 0.0

pl = pv.Plotter()
pl.add_mesh(grid, show_edges=True, style='wireframe', color='w', opacity=0.15)
pl.add_mesh(grid.slice(normal=[1,0,0]), scalars='Hz', cmap='rainbow')
pl.show()
'''