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
