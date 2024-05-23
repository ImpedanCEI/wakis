# User's Guide

This section aims to showcase `wakis`capabilities together with useful recipes to use in the simulation scripts.

Since `wakis` has been developed for computing bea, coupling impedance for particle accelerator components, the example that will serve as a conductive thread for the explanation is a **pillbox cavity with a passing proton beam**. 

The guide will go into detailed step-by-step on how to write the simulation script and visualize or access the computed data.

## Import modules

The first part of a python script always includes importing external sources of code e.g., `packages` or `modules`. In `wakis`, we use:

* `numpy`: Used for numerical operations, especially for matrix operations.
* `scipy.constants`: to import physical constants easily like vacuum permittivity `eps_0` or the speed of light `c`
* `matplotlib`: Used for 1d and 2d plotting and visualization.
* `h5py`: To store data in the memory-efficient format HDF5
* `tqdm`: This package is used for displaying progress bars in loops.
* `pyvista`: For handling and visualizing 3D CAD geometries and vtk-based 3D plotting.

Optionally, one can use `os` or `sys` packages to handle the PATH and directory creations. The first part of any simulaiton script could look similar to:

```python
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import pyvista as pv
import h5py
from tqdm import tqdm
from scipy.constants import c as c_light
```

Next step is to import the `wakis` classes that will allow to run the electromagnetic simulations:

```python
from gridFIT3D import GridFIT3D        # Grid generation
from solverFIT3D import SolverFIT3D    # EM field simulation
from wakeSolver import WakeSolver      # Wake and Impedance calculation
```

## Simulation domain, geometry and materials setup

`wakis` is a numerical electromagnetic solver that uses the Finite Integration Technique. The grid used is a structured grid composed by rectangular cells. The simulation domain is a rectangular box that will be broken into cells where the electromagnetic fields will be computed.

### Number of mesh cells

 In every `wakis` simulation example, the script starts specifiyng the number of mesh cells per direction the user wants to use:

```python
# ---------- Domain setup ---------
# Number of mesh cells
Nx = 50
Ny = 50
Nz = 100
```

```{note}
Note that the number of cells will heavily affect the simulation time (in particular, the timestep due to CFL condition), and also the memory requirements.
```
 
### `STL` geometry importing
In beam-coupling impedance simulations one is usually interested in the geometric impedance, together with the impedance coming from material properties. In `wakis`, the geometry to simulate (sometimes referred as Embedded boundaries) can be imported from a `.stl` file containing a CAD model.

```python
# stl geometry files (add path to them if necessary)
stl_cavity = 'cavity.stl' 
stl_shell = 'shell.stl'
stl_solids = {'cavity': stl_cavity, 'shell': stl_shell}

# Optional: plot the geometry imported using PyVista
geometry = pv.read(stl_shell) + pv.read(stl_cavity)
geometry.plot() # add pyvista **kwargs to make a fancier plot
```
![geometry plot example using a few pyvista **kwargs](img/geometry_plot.png)

```{seealso}
Check [PyVista's documentation](https://docs.pyvista.org/version/stable/user-guide/simple.html#plotting) for more advanced 3d plotting
```

Imported `.stl` files can also be translated, rotated and scaled in x, y, z by providing a list. E.g., `stl_scale['cavity'] = [1., 2, 1.]` will duplicate the y dimension of the imported cavity.stl.

```python
stl_scale = {'cavity': [1., 1., 2.], 'shell': [1., 1., 2.]} # scale factor
stl_rotate = {'cavity': [1., 1., 90.], 'shell': [1., 1., 90.]}  # rotate angle (degrees)
stl_translate = {'cavity': [1., 0, 0], 'shell': [1., 0, 0]} # displacement in [m]
```

### Associating a material to each solid
Each `stl` solid can be associated with a material by indicating `[eps, mu, conductivity]`. In `materials.py`, the most used materials are available in the material library. The user can specify a value for the relative permittivity $\varepsilon$, the relative permeability $\mu$ and the value for the conductivity $\sigma$ in [S/m] or use the material name from the libary:

```python
# Indicate relative permittivity, relative permeability and conductivity [S/m]
# [eps_r, mu_r, conductivity] 
# or the material name from the library:
stl_materials = {'cavity': 'vacuum',       # equivalent to [1.0, 1.0, 0]
                'shell': [1e3, 1.0, 1e3]}  # equivalent to a 'lossy metal'
```

### Defining the `grid` object
After this, the user needs to indicate the domain bounds to simulate. They can be calculated from the imported geometry or just hardcoded:

``` python
# Domain bounds
geometry = pv.read(stl_shell) + pv.read(stl_cavity)
xmin, xmax, ymin, ymax, zmin, zmax = geometry.bounds
```

Finally, we can build our simulation grid using `GridFIT3D` class including all the information above:

```python
# initialize grid object
grid = GridFIT3D(xmin, xmax, ymin, ymax, zmin, zmax, 
                Nx, Ny, Nz, 
                stl_solids=stl_solids, 
                stl_materials=stl_materials,
                stl_translate=stl_translate,
                stl_scale=stl_scale
                stl_rotate=stl_rotate
                )
```

Optionally, the `grid` of the simulation can be inspected interactivelly in 3D (thanks to PyVista):

```python
grid.inspect(add_stl=['cavity', 'shell'] #default is all stl solids
             stl_opacity=0.5             #default is 0.5
             stl_colors=['blue', 'red']) #default is `white`
```

 ![Gif showing how the grid.inspect() method works](img/grid_inspect.gif)

`wakis` python's API is fully exposed, so all the `grid` class parameters relevant for the simulation can be accessed as class attributes and modified once the class has been instantiated. E.g., `grid.dx` gives the mesh step size in x direction, `grid.N` gives the total number of cells. Attributes can be checked by typing `grid.` and then pressing `TAB` when running on `ipython`.


## Setting up the electromagnetic solver

Once the simulation domain has been defined through the geometry and the grid, the electromagnetic (EM) solver can be created. The solver implemented in `wakis` uses the Finite Integration Technique (FIT) [^1]. The recipe on how to instantiate the `SolverFIT3D` class is given below:

```python
solver = SolverFIT3D(grid=grid,     # pass grid object
                     dt=dt,         # (OPTIONAL) define timestep
                     cfl=0.5,       # Default if no dt is defined
                     bc_low=bc_low, 
                     bc_high=bc_high, 
                     use_stl=True,     # Enables or disables geometry import
                     bg='vacuum',      # Background material 
                     )

```

### Simulation timestep
The simulation timestep is calculated following the CFL condition, that mainly depends on the cell size. A default value of 0.5 ensures simulation stability:
```
# Extract of SolverFIT3D __init__
self.dt = cfln / (c_light * np.sqrt(1 / self.grid.dx ** 2 + 1 / self.grid.dy ** 2 +
                                            1 / self.grid.dz ** 2))
```

### Boundary conditions
The required parameters `bc_low` and `bc_high` allow to choose the 6 boundary conditions for our simulation box. For the lower-end boundaries bc_low = [x-, y-, z-] and for the high-end boundaries bc_high = [x+, y+, z+] is used. The supported values to give for -/+ boundaries are:

* `pec` stands for Perfect Electric Conductor: it is a Dirichlet boundary conditions that forces the tangent electric $E$ field at the boundary to be 0. 
* `pmc` stands for Perfect Magnetic Conductor: similarly to pec, it is a Dirichlet boundary conditions that forces the tangent magnetic $H$ field at the boundary to be 0.
* `periodic`: The field values of the high-end boundary are passed to the lower-end boundary, simulating a periodic structure where the fields re-enter the simulaiton domain.
* `abc` First order extrapolation (FOEXTRAP) absorbing boundary condition (ABC)[^2]: a tipe of Dirichlet absorbing boundary condition that allows to absorb longitudinally propagating fields when they reach the boundary. Only works under specific circunstamces.
* `pml` Perfect Matching Layer: A more advanced type of ABC, capable of absorbing electromagnetic waves propagating in a wider range of propagation angles [^3]. This boundary conditions are needed e.g., for simulating accelerator beampipe transitions and RF cavities above beampipe cutoff. *Currently under development*

An example of the boundary conditions that are typically used for the pillbox cavity example are:

```python
# boundary conditions
bc_low=['pec', 'pec', 'pml']  #or z=`pec` if below cutoff
bc_high=['pec', 'pec', 'pml'] #or z=`pec` if below cutoff
```
### Accessing `solver`'s fields, matrices and simulation parameters
Similarly to the `grid` object from `GridFIT3D`, `wakis`'s `SolverFIT3D` class is fully exposed. This means that once the `solver` object is instantiated, all the parameters of the EM solver can be accessed as class atributtes `solver.attr` and modifyed. E.g., one can modify the simulation timestep after the instantiation by doing:

```python
solver.dt = 1e-9 #[s]
```

#### Electromagnetic fields `E`, `J`, `H`
Once the solver has been instantiated, the electromagnetic fields electric $E$, magnetic $H$ and current $J$ can be accessed and modified to e.g., add initial conditions to the simulation. The fields are 3D vectorial matrices of sizes [Nx, Ny, Nz]x3. The times 3 comes from their vectorial nature, since there are values for each simulation cell in $x$, $y$, and $z$ direction. Below an example on how to access the electric field component $E_z$ to add an initial condition to the field:

```python
# modify the lower left bottom corner (cell [0,0,0]) of the simulation domain.
solver.E[0,0,0,'z'] = c_light 

# modify the x component of H for a XY plane at a particular z 
iz = Nz//3
solver.H[:, :, iz, 'x'] = np.ones((Nx, Ny))

# modify the y component of the J on the z axis at a particular x, y
ix, iy = Nx//2, Ny//2
solver.J[ix, iy, :,'y']

# get the absolute value of the electric field
E_abs = solver.E.get_abs() #size [Nx, Ny, Nz]
```

The routines `inspect()` for 2D and `inspect3d()` allow for quick visualization of the field values:
``` python
solver.E.inspect(plane='YZ',            # 2d plane, cut at the domain center
                 cmap='bwr',            # colormap
                 dpi=100,               # plot DPI value (pixel resolution)
                 # also, the plane can be specified as a 2D slice in x,y,z
                 # e.g., for the YZ plane at x = Nx//2
                 x=Nx//2, 
                 y=slice(0,Ny), 
                 z=slice(0,Nz) 
                )

```

#### Material tensors `ieps`, `imu`, `sigma`

The same applies for the material tensors for permittivity $\varepsilon$, permeability $\mu$, and conductivity $\sigma$. For computational cost reasons, the values saved in memory correspont to $\varepsilon^{-1}$ and $\mu^{-1}$.  To access a specific value or a slice of values (and modify them if desired):

```python
# permittibity(^-1) tensor in x direction
solver.ieps[:, :, :, 'x']

# permittibity(^-1) tensor in y direction
solver.imu[:, :, :, 'y']

# Modify first 10 cells in x of the conductivity tensor:
for d in ['x', 'y', 'z']:
    solver.sigma[:10, :, :, d] = np.ones(10)*1e3 #S/m 
```

Since `wakis` supports anisotropy, the tensors are 3D matrices of sizes `[Nx, Ny, Nz]x3` since there are values for each simulation cell in $x$, $y$, and $z$ direction. Similarly, one can inspect the values given to the tensors by using e.g., `solver.sigma.inspect()`


```{hint}
Material tensors $\varepsilon$, $\mu$, and $\sigma$, and electromagnetic fields $E$, $H$ and $J$ are created as `Field` objects, the class in `field.py`. This class allows for optimized access to matrix values, conversion to array format using the lexicograpihc index and inspection methods ìnspect()` via 2D plots to confirm that the tensor are built correctly before running the simulation.
```
### Running a simulation

```{caution}
This guide is in development at the moment. More content will come very soon!
```

## Adding time-dependent sources




## Using `wakis` as a EM+Wakefield solver