# v0.6.0 Draft
*Coming soon!*

## 🚀 New Features

* 🖼️ **Plotting**
  * Unified plotting tools for both MPI and non-MPI simulations.
  * `plot1D` now supports field visualization independently of MPI use.
  * `Plot2D` supports parallel execution.
  * Error handling added for `plot3D` and `plot3DonSTL` when `use_mpi = True`.
  * Support for `dpi` and `return_handles` in plot utilities to further customize plots.
  * Enhanced examples and notebook suite visualization cells.

* 🧱 **GridFIT3D**
  * Added `mpi_initialize()` to handle domain decomposition (Z-slicing).
  * New method `mpi_gather_asGrid()` to retrieve the full global grid from distributed subdomains.
  * Full support for multi-GPU domain decomposition through `cupy` (CUDA-aware, Linux only).
  * Improved communication layer for subdomain synchronization via ghost cells.
  
* ⚡ **SolverFIT3D**
  * MPI-compatible time-stepping routine `mpi_one_step()` using a leapfrog scheme.
  * `mpi_communicate()` to send/receive boundary field values between subdomains.
  * `mpi_gather()` to retrieve field data as a NumPy array and `mpi_gather_asField()` to reconstruct a `Field` object.
  * MPI-safe support integrated into `update()` and field getter logic.
  * Introduced `save_state()` method for checkpointing during MPI runs together with `load_state()`. Added support for MPI save state too.
  * Feature in progress: multiGPU support (`use_GPU=True` when `use_MPI=True`) for distributed simulations (*Linux only!*).
  
* 📥 **Sources**
  * **Beam**:
    * Added `plot(t)` to visualize beam current evolution.
    * Generalized `update()` to work with or without MPI.
    * Enhanced support for time-aligned injection with beta and MPI shifts.
    * New example for MPI+GPU simulation (*topic in progress*).

* 🌊 **WakeSolver**
  * Refactored to internally store the full longitudinal domain.
  * `skip_cells` now acts only at analysis level, preserving resolution.
    * `add_space` and `use_edt` retained for compatibility, but `add_space` is deprecated for new parameter `skip_cells`, more adequate to its utility (i.e. skip cells in the integration path).
  * Future-ready structure for distributed wake solving with MPI-aware GPU.
  * Improved numerical robustness by preventing indexing errors in `WakePotential` integration.
  * Enhanced extrapolation method with `iddefix`.
  * New example `004` for wakefield simulation with MPI+GPU configuration.

* 🛡️ **Security & Documentation**
  * Added `SECURITY.md` to describe supported versions and vulnerability reporting.
  * Improved installation guide with Miniforge (supports both Windows/Linux) and MPI setup instructions.
  * Added new issue templates for **Bug Report** and **Feature Request** with markdown formatting and emojis for readability.
  * Addition of the **Physics Guide**, with detailed physics models and numerical methods explanations.
  * User's guide updated to include Wake extrapolation with `iddefix`, Wake function calculation with `neffint`, and power loss calculation with `BIHC`.
  * Added a **Table of Contents (ToC)** to the documentation for easier navigation.
  * Expanded installation guide with multiGPU configuration and MPI-aware domain partitioning.

## 💗 Other Tag highlights

* 🔁 Nightly tests with GitHub Actions:
  * Enabled infrastructure for MPI-based test cases (`test_003`, `test_005`).
  * Improved test coverage for MPI and GPU simulations.
  * MultiGPU end-to-end tests for distributed domain synchronization.

* 📁 **Examples**:
  * `003` → MPI wakefield simulation using `mpi4py`.
  * `004a` → Fit wake potential data directly with the wake potential resonator formalism.
  * `004b` → Fit impedance from wake potential data using `compute_deconvolution()`.
  * `005` → Full MPI simulation inside Jupyter using `ipyparallel` + `mpi4py`.
  * New example with MPI + GPU configuration for large-scale simulations.

* 📁 **Notebooks**:
  * `005` → Full MPI simulation inside Jupyter using `ipyparallel` + `mpi4py`.
  * New Jupyter notebook showcasing multiGPU configuration.

## 🐛 **Bugfixes**
* Fixed crash in `plot3D` and `plot3DonSTL` when `use_mpi=True`.
* Fixed default `use_mpi=True` to now default to `False` for general usage.
* Fixed a typo in beam injection routine `solver.z.min()`.
* Fixed potential rounding error in wake potential integration with negligible performance impact (~0.1ns) -> solves [issue #12](https://github.com/ImpedanCEI/wakis/issues/12)
* Corrected default beam injection time to align with CST Wakefield Solver reference in beta<1 cases.
* Fixed minor typos in the documentation.
* Fixed synchronization issues with MPI runs when saving states.
* Resolved encoding issues when installing in Windows editable mode.
* Corrected result folder naming in GPU example `002`.

## 👋👩‍💻 **New Contributors**
* [**@mctfr**](https://github.com/mctfr) – Contributed improvements to installation instructions in the documentation.

## 📝 **Full changelog**
`git log v0.5.1... --date=short --pretty=format:"* %ad %d %s (%aN)" | copy`


## 📝 **Full changelog**
`git log v0.5.1... --date=short --pretty=format:"* %ad %d %s (%aN)" | copy`

* 2025-05-09  feature: multiGPU working (Ubuntu) -but needs optimization (elenafuengar)
* 2025-05-09  style: revise GPU example 002, fix folder result name (Elena De La Fuente Garcia)
* 2025-05-09  build: fix encoding when installing editable in Windows (Elena De La Fuente Garcia)
* 2025-05-08  docs: add miniforge (Windows/Linux) to python installation guide (elenafuengar)
* 2025-05-08  docs: fix reference typo (elenafuengar)
* 2025-05-07  feature: support `save_state` for MPI runs (elenafuengar)
* 2025-05-06  docs: fix few mistakes spotted after RTD deployment (elenafuengar)
* 2025-05-06  docs: minor changes to adapt to the physics guide content (elenafuengar)
* 2025-05-06  docs: revised physics guide, `make html` passed (elenafuengar)
* 2025-05-05  docs: first version of physics guide (elenafuengar)
* 2025-05-02  feat: new example for MPI+GPU simulation (in progress) (elenafuengar)
* 2025-04-29  docs: update installation with MPI-GPU findings (elenafuengar)
* 2025-04-24  Update README.md (Elena de la Fuente García)
* 2025-04-24  Create SECURITY.md (Elena de la Fuente García)
* 2025-04-24  Update and rename feature-request-💡.md to feature-request.md (Elena de la Fuente García)
* 2025-04-24  Update bug-report.md (Elena de la Fuente García)
* 2025-04-24  Update and rename bug-report-🐛.md to bug-report.md (Elena de la Fuente García)
* 2025-04-24  Update and rename 💡feature-request-.md to feature-request-💡.md (Elena de la Fuente García)
* 2025-04-24  Add issue templates (bug & feature) (Elena de la Fuente García)
* 2025-04-09  docs: small typo in readme (Elena de la Fuente García)
* 2025-04-07  docs: update readme with playground contents (elenafuengar)
* 2025-04-02  build: update release draft version to 0.6.0 (elenafuengar)
* 2025-04-02  tests: add MPI test files and test script in progress (elenafuengar)
* 2025-04-02  build: add neffint, iddefix and bihc as dependencies (elenafuengar)
* 2025-04-02  style: fix results folder and plot kwargs (elenafuengar)
* 2025-03-29  docs: update release.md (elenafuengar)
* 2025-03-29  docs: small typo (elenafuengar)
* 2025-03-27  style: revised notebook 004, in particular the iddefix extrapolation (elenafuengar)
* 2025-03-27  feature: include wakefield simulation in example 003 (elenafuengar)
* 2025-03-27  docs: fix sidebar TOC in conf.py, add TOC to installation and user guide, minor fixes (elenafuengar)
* 2025-03-27  feature: add wakefield simulation to MPI example and extrapolation to fully decayed (elenafuengar)
* 2025-03-27  feature: MPI wakefield simulation with `solver.wakesolve` is now working (elenafuengar)
* 2025-03-27  docs: update MPI installation after testing on imp machines (elenafuengar)
* 2025-03-27  docs: fix in `index.md` for missing logo (elenafuengar)
* 2025-03-26  docs: prepare v0.5.2 release (elenafuengar)
* 2025-03-26  feature: support for MPI in `wakesolve` in progress + refact: wakesolve saves now all longitudinal values, `skip_cells` will only be applied inside `WakeSolver`. `add_space` and `use_edt` kept for legacy (elenafuengar)
* 2025-03-26  test: adjust tes_001 to `WakeSolver` refactor in previous commit (elenafuengar)
* 2025-03-26  fix: add `if` statement in wake potential calculation to catch rounding errors (profiler indicates only 0.1ns overhead) + refact: `add_space` now deprecated for `skip_cells'. This parameter now adjusts the slicing of z instead of modyfing zmin and zmax, since `solver.wakesolve` will now save all the longitudinal data (elenafuengar)
* 2025-03-26  fix: typo in `solver.z.min()` (elenafuengar)
* 2025-03-25  fix: change default to False for MPI (elenafuengar)
* 2025-03-21  feature: notebook 005 containing working MPI example using `ipyparallel` and `mpi4py`, using MPI methods inside `Grid` and `Solver`. Beautiful <3 (elenafuengar)
* 2025-03-21  feature: working MPI script for an electromagnetic simulation, using new MPI methods inside Grid and Solver (elenafuengar)
* 2025-03-21  style: change BCs for MPI subdmains to `mpi` for code readability (elenafuengar)
* 2025-03-21  feature: MPI implementation in SolverFIT3D, with `mpi_one_step()` to perform MPI leapfrog update, `mpi_communicate()` to send/recv information between mpisubdomains, `mpi_gather()` to retrieve the global field as a np.array for the specific x,y,z and componen, and `mpi_gather_asField()` to retrieve the global field as a `Field` instance (elenafuengar)
* 2025-03-21  feature: MPI implementation in GridFIT3D inside __init__, with `mpi_initialize()` to generate the z-subdomains and `mpi_gather_asGrid()` to retrieve the global grid (elenafuengar)
* 2025-03-21  fix: error handling for `plot3D` and `plot3DonSTL` when `use_mpi = True` -not supported (elenafuengar)
* 2025-03-21  feature: support for MPI and non-MPI inside `plot1D` -input script agnostic! (elenafuengar)
* 2025-03-21  feature: support for MPI and non MPI plots in Plot2D (elenafuengar)
* 2025-03-21  refactor: include MPI support inside `uptade()` (elenafuengar)
* 2025-03-21  style: remove check for component Abs, using __getitem__ supported key (elenafuengar)
* 2025-03-20  feature: support key[3] = 'abs' or 'Abs' in __getitem__ (elenafuengar)
* 2025-03-20  feature: update script to use MPI functions inside solver (elenafuengar)
* 2025-03-20  feature: include MPI functions inside solver to run MPI simulations with `mpi4py` and openmpi (elenafuengar)
* 2025-03-19  refact: change MPI update to use global ZMIN when solver hassattr, add plot(t) function (elenafuengar)
* 2025-03-18  Merge pull request #11 from mctfr/patch-1 (Elena de la Fuente García)
* 2025-03-18  Update installation.md (Manuel Cotelo Ferreiro)
* 2025-03-18  docs: revise docstrings (elenafuengar)
* 2025-03-18  docs: major revision of user guide, update index and installation guide to include MPI setup (elenafuengar)
* 2025-03-16  docs: update README & citation (elenafuengar)
* 2025-03-14  docs: add PR template (elenafuengar)
* 2025-03-13  fix: update default injection time to account for beta (elenafuengar)
* 2025-03-13  feature: add `dpi` and `return_handles` as kwargs (elenafuengar)
