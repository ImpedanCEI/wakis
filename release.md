# v0.6.1

This release introduces major improvements to performance and usability: **running single-precision simulations** allowing x100 speedup on mid-range GPUs, **MKL backend integration** for multithreaded time-stepping (sparse-matrix times vector operations), **adaptive mesh refinement** (first steps, WIP), **STEP geometry unit extraction and scaling** and more robust parsing, added **IDDEFIX wrapper** for streamlined simulation extrapolation, and updated **interactive 3D visualization tools** of imported solids with widgets.
It also enhances multi-GPU compatibility, and testing workflows.

---

## 🚀 New Features

* 🧱 **SolverFIT3D**
  * Performed data-type tracking to enable running single-precision simulations on both CPU and GPU simply by passing `dtype=np.float32` to solver constructor. 
  * Added **MKL backend** for optimized CPU computations, with automatic fallback if unavailable.
  * Introduced environment variable to control MKL threads and improved sparse matrix–vector operations.
  * Added **callback function** argument (`fun(solver, t)`) executed after each timestep for flexible simulation monitoring.
  * Implemented **absorbing boundary conditions (ABC)** for EM simulations with updated testing routine.
  * Added **single-precision support** for solver initialization and data type tracking.

* ⚙️ **Mesh Refinement**
  * Introduced **adaptive mesh refinement** based on OpenFOAM's snappy hexmesh with automatic CFL-stable grid recalculation - *Work in progress*
  * Added example `notebook_006` showcasing refined mesh simulation.

* 🪶 **Geometry & CAD Tools**
  * Added **unit extraction from STEP files** and automatic **STL scaling to meters**.
  * `geometry.load_stp()` now supports **file paths** and **lowercased material names** for consistency.
  * Added tests for geometry unit handling and material case normalization.

* 🎛️ **Visualization**
  * Added interactive `inspect3D` visualization supporting both **PyVista** and **Matplotlib** backends.
  * Introduced new `plot_stl_mask()` tool with interactive 3D sliders to visualize solid occupancy in the computational domain.
  * Added offscreen plotting support for **headless servers** (export to HTML).

* 🌊 **WakeSolver**
  * Added **on-the-fly wake potential computation** and **IDDEFIX** wrapper for wake extrapolation.
  * Improved extrapolated wake consistency with CST/Wakis conventions.
  * Implemented `wakelength` attribute loading when using `load_results()`.

* 🧩 **Miscellaneous**
  * Enhanced GPU/CPU integration — unified timestepping (`one_step`) and backend detection.
  * Enabled multi-GPU test cases and performance optimizations.

---

## 💗 Other Tag Highlights

* 🔁 **Tests**
  * Added **GitHub Action** to trigger tests automatically on PR open or sync.
  * Added MKL vs SciPy backend tests and hardware info retrieval (threads, sockets, affinity).
  * Added test coverage for geometry scaling, lowercase material import, and ABC boundary handling.

* 📚 **Documentation**
  * Updated **multi-GPU** and **MKL installation** guides.
  * Added **CSG geometry** and PyVista snippets to the User’s Guide.
  * Refined **Physics Guide**, improved clarity on SI base units.
  * Added SWAN badge, DOI, and tutorial repo to README.
  * Simplified **issue** and **feature request templates** for contributors.

* ⚙️ **Build & Compatibility**
  * Ensured **NumPy 2.0+ compatibility**.
  * Upgraded **PyVista** dependency to enable `|` (union) operator for CSG modeling.

---

## 🐛 **Bugfixes**

* Fixed crash in GPU memory pinning and added `to_gpu()` routine for reliable field transfer.
* Fixed axis allocation in grid and tensors `inspect()`.
* Fixed robustness of `load_results()` to ensure trailing slash consistency and automatic loading of simulated wakelength.
* Corrected transverse impedance save to logfile.
* Fixed synchronization in MKL backend initialization when GPU is disabled.
* Fixed minor doc typos and link issues (e.g. WSL installation link).
* Fixed nightly test failures caused by lowercase material names.

---

## 👋👩‍💻 **New Contributors**

* [**@Antoniahuber**](https://github.com/Antoniahuber) — Implemented geometry unit extraction, STL scaling, material normalization, and related tests.  
* [**@Elleanor-Lamb**](https://github.com/Elleanor-Lamb) — Updated documentation for HTCondor and GPU installation.  

---

## 📝 **Full changelog**

| **83 commits** | 📚 Docs | 🧪 Tests | 🐛 Fixes | 🎨 Style | ✨ Features | Other |
|-----------------|---------|----------|-----------|------------|--------------|-------|
| % of Commits    | 26.5%   | 10.8%    | 9.6%      | 8.4%       | 35.0%        | 9.7%  |


`git log v0.6.0...v0.6.1 --date=short --pretty=format:"* %ad %d %s (%aN)*


* 2025-11-04  test: added action to trigger tests on PR open or sync (elenafuengar)
* 2025-11-04  Allow lists as color inputs --> merge #28 from Antoniahuber/main (Elena de la Fuente García)
* 2025-11-03  Allow lists as color inputs (Antonia Huber)
* 2025-10-31  feature: extract units from STEP file and scale the generated STL geometry to be in meters --> Merge pull request #27 from Antoniahuber/main (Elena de la Fuente García)
* 2025-10-31  Added possibility to give a filepath to stl-files (Antonia Huber)
* 2025-10-31  Merge pull request #1 from Antoniahuber/Documentation-Change-geometry Scale units function (Antoniahuber)
* 2025-10-31  units Test (Antoniahuber)
* 2025-10-31  Update test_006_geometry_utils.py (Antoniahuber)
* 2025-10-30  Test for units function (Antoniahuber)
* 2025-10-30  Merge branch 'ImpedanCEI:main' into Documentation-Change-geometry (Antoniahuber)
* 2025-10-30  bugfix: fixing nightly test failing after lowercase fix --> merge #26 from Antoniahuber/main (Elena de la Fuente García)
* 2025-10-30  lowercaseInRightFunction.py (Antoniahuber)
* 2025-10-30  Merge branch 'ImpedanCEI:main' into main (Antoniahuber)
* 2025-10-30  Test function for lowercase materials (Antoniahuber)
* 2025-10-28  bugfix: ensure material in lower case + docs: minor fixes #20 (Elena de la Fuente García)
* 2025-10-28  Update geometry.py (Antoniahuber)
* 2025-10-21  Recognize unit in .stp file, completed docstring, converts materialnames to lowercase (Antoniahuber)
* 2025-10-20  Import materials from .stp in lowercase (Antoniahuber)
* 2025-10-20  Added commas in usersguide (Antoniahuber)
* 2025-10-17  style: simplify feature request template (elenafuengar)
* 2025-10-17  style: simplify issue template (elenafuengar)
* 2025-09-26  style: added call to `inspect3D`, allowing to visualize interactively in 3d the material tensors or electromagnetic fields (elenafuengar)
* 2025-09-26  feature: enhance inspect3D method to support interactive visualization with PyVista and Matplotlib backends (elenafuengar)
* 2025-09-26  feature: fix slider rendering to save the slider bounds after every callback (elenafuengar)
* 2025-09-25  style: add new plot_stl_mask method to the notebook (elenafuengar)
* 2025-09-25  feature: `plot_stl_mask` to show the cells occupied by a certain solid in the computational domain. The plot is interactive with 3 sliders in x, y, z (elenafuengar)
* 2025-09-22  WIP: gradient based extraction of solid boundaries for SIBC (elenafuengar)
* 2025-08-20  feature: load wakelength attr when using load_results -needed for extrapolation (elenafuengar)
* 2025-08-20  docs: update multi-gpu from notebooks guide (elenafuengar)
* 2025-08-20  fix: add flag not on GPU for the MKL backend (elenafuengar)
* 2025-08-20  style: include results in notebook 005, run on multi-gpu (elenafuengar)
* 2025-08-14  style: revision of notebook 005, added lines for multi-GPU, use iddefix wrappers for extrapolation (elenafuengar)
* 2025-08-14  docs: add MKL installation and customization instructions (elenafuengar)
* 2025-08-13  tests: retrieve num sockets and cores from lscpu for omp num threads + mem. pinning via KMP affinity (elenafuengar)
* 2025-08-13  tests: add MKL vs scipy test (elenafuengar)
* 2025-08-13  docs: add SWAN badge and tutorial repo to readme (elenafuengar)
* 2025-08-12  refact: one_step routine to private, GPU/CPU share same routine (elenafuengar)
* 2025-08-12  feature: adding MKL backend, refact:one_step routine assignment handled inside __init__ (elenafuengar)
* 2025-08-12  feature: WIP on-the-fly wake potential calculation (elenafuengar)
* 2025-08-12  build: compatibility with numpy2.0+ (elenafuengar)
* 2025-08-12  refactor: one_step func assignment is handled inside solverFIT3D (elenafuengar)
* 2025-08-12  fix: bug in `to_gpu()` routine, bug in `inspect()` when allocating the axes, enforcing memory pinning in `fromarray()` (elenafuengar)
* 2025-08-08  feature: speedup by avoiding sparse diag operations during timestepping (elenafuengar)
* 2025-08-08  feature: Add number of threads env variable for MKL backend (elenafuengar)
* 2025-08-08  feature: MKL backend working and added to routines -will be used if it can be imported (elenafuengar)
* 2025-08-08  feature: WIP, explore multithreaded sparsemat-vec operation using MKL backend for scipy (elenafuengar)
* 2025-08-08  feature: add option for grid plotting offscreen for headless-servers. It exports the scene to html file instead (elenafuengar)
* 2025-08-05  test: implementing the 2-timestep ABC BCs and testing with against a planewave (WIP) (elenafuengar)
* 2025-08-05  fix: allow to pass custom transverse slices to WavePacket (elenafuengar)
* 2025-08-05  feature: add `callback` arg that allows to pass a custom function in the form fun(solver, t) right after the timestep update (elenafuengar)
* 2025-08-04  feature: ABC boundaries implemented in the `emsolve` routine for testing (elenafuengar)
* 2025-08-04  feature: WIP updated version of the ABC boundaries (elenafuengar)
* 2025-07-25  feature: data type tracking to enable passing desired precision to solverFIT3D constructor -> support for single-precision simulations! (elenafuengar)
* 2025-07-25  fix: add extra check for field on gpu when calling inspect (elenafuengar)
* 2025-07-25  tests: update 006 to new key naming convention (elenafuengar)
* 2025-07-23  feature: working simlation with mesh refinement -WIP (elenafuengar)
* 2025-07-23  style: revision of 004 notebook (elenafuengar)
* 2025-07-23  feature: improved STEP file parsing to avoid buffering the stp file but instead regex line-by-line (elenafuengar)
* 2025-07-23  fix: make load_results func more robust by adding end slash if the name does not end with it (elenafuengar)
* 2025-07-22  feature: recalculate mesh spacing after refinement to improve cfl stability + WIP notebook 006 (elenafuengar)
* 2025-07-22  feature: add notebook 006 to showcase/test mesh refinement (WIP) (elenafuengar)
* 2025-07-22  feature: mesh refinement bug fixes, got first simulation running! (elenafuengar)
* 2025-07-22  feature: used newly implemented IDDEFIX wrapper functions to extrapolated the simulated wake (elenafuengar)
* 2025-07-22  refact: splitted DE model fitting and added new functions to retrieve extrapolated wake potential, function and impedance -applying convention and unit changes to be consistent with Wakis/CST (elenafuengar)
* 2025-07-22  feature: finalized first version of mesh refinement (elenafuengar)
* 2025-07-22  docs: emphazise on inputs units - Wakis always uses SI base units (elenafuengar)
* 2025-07-22  Merge pull request #17 from elleanor-lamb/gpu_docs_update (Elena de la Fuente García)
* 2025-07-22  updated docs (Elleanor Lamb)
* 2025-07-22  updated docs for HTCondor (Elleanor Lamb)
* 2025-07-22  feature: wrapping function for easy wake extrapolation using IDDEFIX (elenafuengar)
* 2025-07-22  fix: bug in transverse impedance save to logfile (elenafuengar)
* 2025-07-15  docs: minor fizes in physics guide and releases (elenafuengar)
* 2025-07-03  fix: broken link for WSL installation (elenafuengar)
* 2025-06-30  docs: add release notes to docs and modify index (elenafuengar)
* 2025-06-30  feature: first steps towards smart snappy grid (elenafuengar)
* 2025-06-19  build: upgrade PyVista version to enable use of | (union) operator for CSG modelling (elenafuengar)
* 2025-06-20  feat: add error handling for Nz<N_mpi_proc + more verbose output about MPI domain splitting (elenafuengar)
* 2025-06-11  docs: polish installatio guide for Windows (elenafuengar)
* 2025-06-05  docs: add CSG geometry to the users guide + pyvista code snippet (elenafuengar)
* 2025-05-27  docs: update README with v0.6.0 features and DOI (elenafuengar)