# v0.5.0 Draft
*Comming soon!*

## New Features
* Perfect Matching Layers (PML) boundary conditions
* Plot field on STL solid using `pyvista.sample` interpolation algorithm 
    * Interactive plane clipping on `plot3DonSTL`
    * Field shown on clipping plane
* Wake extrapolation of partially decayed wakes coupling new package `ImpedanCEI/iddefix` 
* Conversion from impedance to wake function and viceversa with user-defined arbitrary sampling

## Tag highlights
* Nightly tests with Github actions 
* :file_folder: notebooks containing interactive examples
* :file_folder: examples: major cleanup, examples on CPU and GPU

## Bugfixes
* Patch representation when a list is passed in `Plot2D`
* `ipympl` added as a dependency to `wakis['notebook']` installation
* Injection time to account for relativistic beta
