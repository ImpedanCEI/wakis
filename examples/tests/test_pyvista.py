import pyvista as pv
import numpy as np
from scipy.constants import c as c_light
import sys
sys.path.append('../../')
from field import Field 

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


# --- Domain definition ----

L = 1. # Domain length

Nx = 5
Ny = 8
Nz = 10
N = Nx*Ny*Nz

Lx = L
Ly = L
Lz = L
dx = L / Nx
dy = L / Ny
dz = L / Nz

xmin = -Lx/2 #+ dx / 2
xmax = Lx/2 #+ dx / 2
ymin = - Ly/2 #+ dy / 2
ymax = Ly/2 #+ dy / 2
zmin = - Lz/2 #+ dz / 2
zmax = Lz/2 #+ dz / 2

x = np.linspace(xmin, xmax, Nx + 1)
y = np.linspace(ymin, ymax, Ny + 1)
z = np.linspace(zmin, zmax, Nz + 1)

# ----- Structured Grid -----

#Z, Y, X = np.meshgrid(z,y,x, indexing='ij')
X, Y, Z = np.meshgrid(x,y,z, indexing='ij')
#grid = pv.StructuredGrid(X,Y,Z)
grid = pv.StructuredGrid(X.transpose(),Y.transpose(),Z.transpose())

#grid.bounds
#grid.plot(smooth_shading=True, show_edges=True)

'''
pl = pv.Plotter()
pl.add_mesh(grid, scalars = [i for i in range(grid.n_points)], show_edges=True)
label_coords = grid.points + [0, 0, 0.02]
point_labels = [f'Point {i}' for i in range(grid.n_points)]
pl.add_point_labels(label_coords, point_labels,
                    font_size=25, point_size=20)
pl.camera_position = 'yz'
pl.show()
'''

# ----- Unstructured Grid -----
'''
dh = np.min([dx, dy, dz])
si, sj, sk = dx/dh, dy/dh, dz/dh

 # grid size: ni*nj*nk cells; si, sj, sk steps
grid_ijk = np.mgrid[
            : (Nx+1) * si : si,
            : (Ny+1) * sj : sj,
            : (Nz+1) * sk : sk,
            ]
# repeat array along each Cartesian axis for connectivity
for axis in range(1, 4):
     grid_ijk = grid_ijk.repeat(2, axis=axis)

# slice off unnecessarily doubled edge coordinates
grid_ijk = grid_ijk[:, 1:-1, 1:-1, 1:-1]

# reorder and reshape to VTK order
corners = grid_ijk.transpose().reshape(-1, 3)
dims = np.array([Nx, Ny, Nz]) + 1
egrid = pv.ExplicitStructuredGrid(dims, corners)
egrid = egrid.compute_connectivity()
#egrid.plot(smooth_shading=True, show_edges=True)
# !! bounds are not correct !!
#egrid.bounds

# ----- Comparing grid and egrid -----

pl = pv.Plotter(shape=(1, 2), border=False)
actor00 = pl.add_mesh(grid, smooth_shading=True, show_edges=True)
pl.add_title("Structured grid", font='times')

pl.subplot(0, 1)
actor01 = pl.add_mesh(egrid, smooth_shading=True, show_edges=True)
pl.add_title("Explicit Structured grid", font='times')

pl.show()
'''

# ---- Add Scalar data ----

analyticH = Field(Nx, Ny, Nz)
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
#grid.plot(smooth_shading=True, show_edges=True, scalars='Hy', cmap='rainbow')
#grid.plot(smooth_shading=True, show_edges=True, scalars='Hx', cmap='rainbow')
#analyticH.inspect3D(cmap='rainbow')

'''
# Using np.zeros instead of field class = Equivalent
analyticHz = np.zeros((Nx, Ny, Nz))
dt = 1.0 / (c_light * np.sqrt(1 / dx ** 2 + 1 / dy ** 2 + 1 / dz ** 2))
Nt = 10

for ii in range(Nx):
    for jj in range(Ny):
        for kk in range(Nz):

            x = (ii+0.5) * dx + xmin
            y = (jj+0.5) * dy + ymin
            z = (kk+0.5) * dz + zmin
            analyticHz[ii, jj, kk] = analytic_sol_Hz(x, y, z, (Nt-0.5) * dt)

grid.cell_data['Hz'] = np.reshape(analyticHz[:, :, :], N)
grid.plot(smooth_shading=True, show_edges=True, scalars='Hz', cmap='rainbow')
'''

# ------ Clip scalar data with stl -------

# --- Read stl ----
unit = 1e-3
surf = pv.read('goniometer.stl')
surf = surf.rotate_x(90)    # z axis longitudinal
surf = surf.scale(unit)     # [m]
#surf = surf.subdivide(3, subfilter='linear') #if used, select.threshold() is empty

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

# ---- Cells inside surface ----
tol = unit*1.e-3
select = grid.select_enclosed_points(surf, tolerance=tol)
inside = select.threshold(0.1)
points_inside = np.where(select['SelectedPoints'] > 0.1)[0]
cells_inside = np.where(select.point_data_to_cell_data()['SelectedPoints'] > 0.1)[0]
grid['Solid1'] = select.point_data_to_cell_data()['SelectedPoints']

# ---- Add Scalar data ----

analyticH = Field(Nx, Ny, Nz)
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

# -------- Clip data -------

grid.cell_data['Hz'][cells_inside] = 0.0

# -------- Plot -------
pl = pv.Plotter()
pl.add_mesh(grid, show_edges=True, style='wireframe', color='w', opacity=0.15)
pl.add_mesh(grid.slice(normal=[1,0,0]), scalars='Hz', cmap='rainbow')
#pl.add_mesh(grid.extract_cells(cells_inside), scalars='Solid1', cmap='Blues', opacity=0.6)
#pl.add_mesh(surf, color='blue', opacity=0.35)

pl.show()

# ------ Clip analyticH instead -----

analyticH[cells_inside, 'y'] = 0.0

grid.cell_data['Hy'] = np.reshape(analyticH[:, :, :, 'y'], N)

# -------- Plot -------
pl = pv.Plotter()
pl.add_mesh(grid, show_edges=True, style='wireframe', color='w', opacity=0.15)
pl.add_mesh(grid.slice(normal=[1,0,0]), scalars='Hz', cmap='rainbow')
#pl.add_mesh(grid.extract_cells(cells_inside), scalars='Solid1', cmap='Blues', opacity=0.6)
#pl.add_mesh(surf, color='blue', opacity=0.35)

pl.show()