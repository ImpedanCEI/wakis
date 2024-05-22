# Installation guide

This section covers how to install `wakis` package for developers and users. 

The installation guide is writen for Linux, but `wakis` code and dependencies are 100% python so it can run in other operating systems. For developers using Windows, we recommend checking the [WSL setup guide](#WSL) to install and setup the Windows subsystem for linux.

## Installing from Github

This is the recommended installation for developers. To start using wakis, you can `git clone`it from the main repository:

```
# SSH:
git clone git@github.com:ImpedanCEI/FITwakis.git

# or HTTPS:
git clone https://github.com/ImpedanCEI/FITwakis.git
```

However, if you would like to improve and make changes in `wakis`, we encorage to create a [fork](https://github.com/ImpedanCEI/wakis/fork) from wakis' `main` branch: https://github.com/ImpedanCEI/wakis on your personal GitHub. 

To contribute, first fork the repository, create a new branch, and submit a pull request. Step-by-step:

1. Fork the repository: https://github.com/ImpedanCEI/wakis/fork
2. Create a new branch: `git checkout -b my-feature-branch`
3. Make your changes and commit them:
    `git add my-changed-script.py`
    `git commit -m 'Explain the changes'`
4. Push to the branch: git push origin my-feature-branch
5. Submit a pull request: https://github.com/ImpedanCEI/wakis/pulls

## Dependencies

`wakis` is a 100% pyhton code that relies only on a few renowed python packages:

* `numpy`: Used for numerical operations, especially for matrix operations.
* `scipy`: Provides additional functionality for sparse matrices and other scientific computations.
* `matplotlib`: Used for 1d and 2d plotting and visualization.
* `h5py`: This package provides a Python interface to the HDF5 binary data format for storing the field 3D data for several timesteps
* `tqdm`: This package is used for displaying progress bars in loops.
* `pyvista`: For handling and visualizing 3D CAD geometries and vtk-based 3D plotting.

To install the dependencies in a conda python environment, simply run:

```
pip install -r requirements.txt
```

## Python installation

If a python installation has not been setuo yet, we recommend using [miniconda](https://docs.anaconda.com/free/miniconda/index.html) [^2]. Miniconda can be installed and activated by:

```
# get, install and activate miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh 
source miniconda3/bin/activate

# create dev python environment
conda create --name wakis-env python=3.9
conda activate wakis-env
```

Once the conda environment is created, one can proceed with the dependencies. On  `wakis` main directory do:

```
pip install ipython
pip install -r requirements.txt
```


[PyVista](https://github.com/pyvista/pyvista) a python package for 3D plotting and mesh analysis through a streamlined interface for the Visualization Toolkit (VTK) [^1]. 
We rely on [PyVista](https://github.com/pyvista/pyvista) to import CAD/STL geometry as embedded boundaries or/and as solids with different materials into the domain. This allows `wakis` to efficiently render 3D plots of the electromagnetic fields and visualize the computational mesh interactively.



[^1]: Sullivan and Kaszynski, (2019). PyVista: 3D plotting and mesh analysis through a streamlined interface for the Visualization Toolkit (VTK). Journal of Open Source Software, 4(37), 1450, https://doi.org/10.21105/joss.01450
[^2]: Anaconda Software Distribution. (2020). Anaconda Documentation. Anaconda Inc. Retrieved from https://docs.anaconda.com/