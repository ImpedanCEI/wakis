# v0.5.2 Draft
*Comming soon!*

## 🚀 New Features

* 🖼️ **Plotting**
  * Unified plotting tools for both MPI and non-MPI simulations.
  * `plot1D` now supports field visualization independently of MPI use.
  * `Plot2D` supports parallel execution.
  * Error handling added for `plot3D` and `plot3DonSTL` when `use_mpi = True`.
  * Support for `dpi` and `return_handles` in plot utilities.

* 🧱 **GridFIT3D**
  * Added `mpi_initialize()` to handle domain decomposition (Z-slicing).
  * New method `mpi_gather_asGrid()` to retrieve the full global grid from distributed subdomains.

* ⚡ **SolverFIT3D**
  * MPI-compatible time-stepping routine `mpi_one_step()` using a leapfrog scheme.
  * `mpi_communicate()` to send/receive boundary field values between subdomains.
  * `mpi_gather()` to retrieve field data as a NumPy array and `mpi_gather_asField()` to reconstruct a `Field` object.
  * MPI-safe support integrated into `update()` and field getter logic.

* 📥 **Sources**
  * **Beam**:
    * Added `plot(t)` to visualize beam current evolution.
    * Generalized `update()` to work with or without MPI.
    * Enhanced support for time-aligned injection with beta and MPI shifts.

* 🌊 **WakeSolver**
  * Refactored to internally store the full longitudinal domain.
  * `skip_cells` now acts only at analysis level, preserving resolution.
  * `add_space` and `use_edt` retained for compatibility, but `add_space` is deprecated for new parameter `skip_cells`
  * Future-ready structure for distributed wake solving with MPI.
  * Improved numerical robustness by preventing indexing errors in `WakePotential` integration.

## 💗 Other Tag highlights

* 🔁 Nightly tests with GitHub Actions:
  * Enabled infrastructure for MPI-based test cases (`test_003`, `test_005`).

* 📁 Examples:
  * `003` → MPI wakefield simulation using `mpi4py`.

* 📁 Notebooks:
  * `005` → MPI simulation inside Jupyter using `ipyparallel` + `mpi4py`.

## 🐛 Bugfixes
* Fixed crash in `plot3D` and `plot3DonSTL` when `use_mpi=True`.
* Fixed default `use_mpi=True` to now default to `False` for general usage.
* Fixed a typo in `solver.z.min()`.
* Fixed potential rounding error in wake potential integration with negligible performance impact (~0.1ns).
* Corrected default beam injection time to align with CST Wakefield Solver reference.

## 👋👩‍💻 New Contributors

* [**@mctfr**](https://github.com/mctfr) – Contributed improvements to installation instructions in the documentation.

## 📝 Full changelog
`git log v0.5.1... --date=short --pretty=format:"* %ad %d %s (%aN)" | copy`

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
* 2025-03-13  feature: add `dpi` and `return_handles` as kwargs (elen_*
