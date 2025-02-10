# v0.5.0 Draft
*Comming soon!*

## New Features
* Perfect Matching Layers (PML) boundary conditions: First version out!

* Ploting:
    * `solver.plot3DonSTL` Field on STL solid using `pyvista.sample` interpolation algorithm 
        * Interactive plane clipping on `plot3DonSTL`
        * Field shown on clipping plane
    * `grid.plot_solids()` 3D plot with the imported solids and the position in the simulation bounding box when `bounding_box=True`

* Sources:
    * Add `plot(t)` method to plot the source over the simulation time `t` 
    * Custom amplitude as an attribute `self.amplitude`.

* Wake extrapolation of partially decayed wakes coupling with [`IDDEFIX`]: https://github.com/ImpedanCEI/IDDEFIX: 
    * IDDEFIX is a physics-informed machine learning framework that fits a resonator-based model (parameterized by R, f, Q) to wakefield simulation data using Evolutionary Algorithms. It leverages Differential Evolution to optimize these parameters, enabling efficient classification and extrapolation of electromagnetic wakefield behavior. This allows for reduced simulation time while maintaining long-term accuracy, akin to time-series forecasting in machine learning

* Impedance to wake function conversion using non-equidistant fourier transform with: [`neffint]: https://github.com/ImpedanCEI/neffint
    * Neffint is an acronym for Non-equidistant Filon Fourier integration. This is a python package for computing Fourier integrals using a method based on Filon's rule with non-equidistant grid spacing.

* Beam-induced power loss calculations for different beam shapes and filling schemes using ['BIHC`]: https://github.com/ImpedanCEI/BIHC
    * Beam Induced Heating Computation (BIHC) tool is a package that allows the estimation of the dissipated power due to the passage of a particle beam inside an accelerator component. The dissipated power value depends on the characteristics of the particle beam (beam spectrum and intensity) and on the characteristics of the consdiered accelerator component (beam-coupling impedance).

## Other Tag highlights
* Nightly tests with Github actions 
* :file_folder: notebooks: containing interactive examples
* :file_folder: examples: major cleanup, examples on CPU and GPU

## Bugfixes
* Patch representation when a list is passed in `Plot2D`
* `ipympl` added as a dependency to `wakis['notebook']` installation
* Injection time to account for relativistic beta

## Full Changelog
`git log v0.4.0... --date=short --pretty=format:"* %ad %d %s (%aN)" | copy`

* 2025-02-07  (HEAD -> main, origin/main, origin/HEAD) fix: use tilde quantities (elenafuengar)
* 2025-02-07  fix: pass anti_aliasing argument to pyvista (elenafuengar)
* 2025-02-07  fix: add bc_high to logical checks in `apply_bc_to_c` (elenafuengar)
* 2025-02-06  docs: update new features (elenafuengar)
* 2025-02-06  refactor: move benchmarks to satellite repository (elenafuengar)
* 2025-02-06  fix: revised notebook with embedded html firgures (2) (elenafuengar)
* 2025-02-06  docs: add troubleshooting for headless monitors (elenafuengar)
* 2025-02-06  fix: revised notebook with embedded html firgures (elenafuengar)
* 2025-02-06  feat: allow to pass amplitude as attribute (elenafuengar)
* 2025-02-06  feat: in `plot_solids()` add SSAA atialiasing, bounding box and camera position. in `inspect()` add SSAA antialiasing and allow empty meshes (elenafuengar)
* 2025-02-01  add iddefix and bihc references (elenafuengar)
* 2025-01-28  remove admonition (elenafuengar)
* 2025-01-28  finalize contribution guide and add a snippet to Code of conduct (elenafuengar)
* 2025-01-27  Update 004_Wakefield_simulation_and_extrapolation.ipynb (Elena de la Fuente García)
* 2025-01-27  Update 004_Wakefield_simulation_and_extrapolation.ipynb (Elena de la Fuente García)
* 2025-01-24  Change in DE results to pass directly ndarray and not dict (Elena de la Fuente García)
* 2025-01-24  Merge pull request #7 from ImpedanCEI/wakis-iddefix-example (Elena de la Fuente García)
* 2025-01-24  (origin/wakis-iddefix-example) Added iddefix to example (Malthe Raschke Nielsen)
* 2025-01-24  add code of conduct (elenafuengar)
* 2025-01-24  first version of contributing.md (elenafuengar)
* 2025-01-23  change in load results (elenafuengar)
* 2025-01-23  remove cell outputs (elenafuengar)
* 2025-01-23  Add example to shouwcase IDDEFIX wake extrapolation (elenafuengar)
* 2025-01-23  revision and restructuring of the example (elenafuengar)
* 2025-01-23  minor update (elenafuengar)
* 2025-01-22  add logo to 3d plots when possible (elenafuengar)
* 2025-01-22  add logo and fix bug in grid.inspect() colors dict (elenafuengar)
* 2025-01-22  bugfix in gridFIT3 (elenafuengar)
* 2025-01-21  small typo (elenafuengar)
* 2025-01-21  add grid.plot_solids() to the example (elenafuengar)
* 2025-01-21  add assign_colors() routine and plot_solids to 3d plot imported solids (elenafuengar)
* 2025-01-21  add material_colors library (elenafuengar)
* 2025-01-21  remove pyvista plots since can cause seg fault on headless servers without x11 (elenafuengar)
* 2025-01-21  add GPU example (elenafuengar)
* 2025-01-21  add animation command (elenafuengar)
* 2025-01-21  add example of each plot type (elenafuengar)
* 2025-01-17  new release draft description (elenafuengar)
* 2025-01-16  fix typo and add 3d plotting snippet (elenafuengar)
* 2025-01-16  add example 001 content (elenafuengar)
* 2025-01-16  add stl files for example 001 (elenafuengar)
* 2025-01-15  update README (elenafuengar)
* 2025-01-14  add test for wake to impedance conversion (elenafuengar)
* 2025-01-14  add functions to go from wake function to impedance and viceversa, for any frequency span, resolution and number of samples (elenafuengar)
* 2025-01-10  new examples placeholders (elenafuengar)
* 2025-01-10  Moving scripts to subfolder (elenafuengar)
* 2025-01-10  minor fixes (elenafuengar)
* 2025-01-10  minor fixes (elenafuengar)
* 2025-01-10  Update requirements.txt (Elena de la Fuente García)
* 2025-01-02  Create CITATION.cff (Elena de la Fuente García)
* 2024-12-20  Update README.md (Elena de la Fuente García)
* 2024-12-20  fix version (elenafuengar)
* 2024-12-20  fix token name (elenafuengar)
* 2024-12-20  change to v3 (elenafuengar)
* 2024-12-20  remove slug (elenafuengar)
* 2024-12-20  change to dispatch (elenafuengar)
* 2024-12-20  change trigger (elenafuengar)
* 2024-12-20  change trigger (elenafuengar)
* 2024-12-20  fix comments (elenafuengar)
* 2024-12-20  fix comments (elenafuengar)
* 2024-12-20  change trigger (elenafuengar)
* 2024-12-20  fix identation (elenafuengar)
* 2024-12-20  adding codecov tests - trigger in 2 minutes (elenafuengar)
* 2024-12-20  adding codecov tests (elenafuengar)
* 2024-12-19  reorganize badges (elenafuengar)
* 2024-12-19  add nightly test badge (elenafuengar)
* 2024-12-19  revert to midnight (elenafuengar)
* 2024-12-19  test trigger UTC time (elenafuengar)
* 2024-12-19  test trigger (elenafuengar)
* 2024-12-19  delete typo (elenafuengar)
* 2024-12-19  add nightly tests (elenafuengar)
* 2024-12-18  add nightly tests (elenafuengar)
* 2024-12-18  generate h5 file inside tests folder (elenafuengar)
* 2024-12-18  fix actions (elenafuengar)
* 2024-12-18  mark all pyvista tests as interactive (elenafuengar)
* 2024-12-18  fix actions (elenafuengar)
* 2024-12-18  fix actions (elenafuengar)
* 2024-12-18  update troubleshooting (elenafuengar)
* 2024-12-09  adding static methods for wake to impedance conversion and viceversa (ELENA DE LA FUENTE)
* 2024-12-06  Merge pull request #4 from ImpedanCEI/MaltheRaschke-patch-1 (Elena de la Fuente García)
* 2024-12-03  (origin/MaltheRaschke-patch-1) Small code comment correction (MaltheRaschke)
* 2024-11-22  improve description (elenafuengar)
* 2024-11-22  support matplotlib **kwargs (elenafuengar)
* 2024-11-22  add python notebook examples (elenafuengar)
* 2024-11-22  add notebook troubleshooting (elenafuengar)
* 2024-11-20  Update index.md (Elena de la Fuente García)
* 2024-10-23  add GPU setup instructions (elenafuengar)
* 2024-10-22  update docstring plot1D (elenafuengar)
* 2024-10-22  fix bugs in WavePaket + time plot feature (elenafuengar)
* 2024-10-22  minor changes (elenafuengar)
* 2024-10-22  numpy version (elenafuengar)
* 2024-10-05  setup manual github action (elenafuengar)
* 2024-10-03  Add lines of code! in green (Elena de la Fuente García)
* 2024-10-03  Add lines of code! (Elena de la Fuente García)
* 2024-10-03  Add lines of code! (Elena de la Fuente García)
* 2024-10-03  Update manual_tests_CPU.yml (Elena de la Fuente García)
* 2024-10-03  Create manual_tests_CPU.yml (Elena de la Fuente García)
* 2024-10-03  Create manual.yml (Elena de la Fuente García)
* 2024-09-30  Update index.md (Elena de la Fuente García)
* 2024-09-28  adding 3d plot (elenafuengar)
* 2024-09-26  new feautures in plot3DonSTL: field on plane capabilities (elenafuengar)
* 2024-09-26  bugfix in plot2D: use patch_reverse flag when a list of stl patches is passed (elenafuengar)
* 2024-09-26  bigfix: injection time in Beam source + add beta to speed (elenafuengar)
* 2024-09-26  add clip_box option to plot3DonSTL (elenafuengar)
* 2024-09-26  fix typo in verbose (elenafuengar)
* 2024-09-25  Merge branch 'main' of github.com:ImpedanCEI/wakis into main (elenafuengar)
* 2024-09-25  update version to match pyPI (elenafuengar)
* 2024-09-24  add license badge (Elena de la Fuente García)
* 2024-09-24  add pypi badge (Elena de la Fuente García)
* 2024-09-24  Update README.md (Elena de la Fuente García)
* 2024-09-24  Update README.md (Elena de la Fuente García)