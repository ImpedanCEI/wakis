# v0.5.1 Draft
*Comming soon!*

## 🚀 New Features
* Plotting
  * Allow passing camera position to solver's 3D plots `plot3D` and `plor3DnSTL`
  
## 💗 Other Tag highlights
* 🔁 Nightly tests with GitHub actions: 
  * 003 -> coverage for grid `inspect` and `plot_solid`
* 📁 Examples: 
  * 003 -> MPI wakefield simulation with `mpi4py`
* 📁 Notebooks: 
  * 005 -> MPI example in jupyter notebooks with `ipyparallel`+ `mpi4py`

## 🐛 Bugfixes 
* `__version__` now matches PyPI release and tag
* `gridFIT3D.plot_solids()` fix typo in the opacity assignment
* `example/001` fixed stl_solids parameter in grid.inspect() call

## 👋👩‍💻New Contributors


## 📝Full changelog
`git log v0.5.0... --date=short --pretty=format:"* %ad %d %s (%aN)" | copy`
* 2025-03-12  fix: typo in ocpacity, variable overwritten (elenafuengar)
* 2025-03-12  test: add coverage for grid.inspect and grid.plot_solids (elenafuengar)
* 2025-03-12  fix: add_stl use keys not stl file path, add default parameters and uncomment (elenafuengar)
* 2025-03-11  feature: allow passing camera position as argument on 3D plotting routines (elenafuengar)
* 2025-03-11  test: study different pml_func profiles and compare smoothness and derivatives (elenafuengar)
* 2025-03-10  docs: update release.md for next version (elenafuengar)
* 2025-03-07  docs: update supported python versions 3.8+ <3.12 (elenafuengar)
* 2025-03-07  docs: update version (elenafuengar)
* 2025-03-07  docs: update CITATION.cff with v0.5.0 doi (Elena de la Fuente García)
* 2025-03-07  docs: add zenodo badge to readme.md (Elena de la Fuente García)
