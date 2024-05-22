# `wakis`
> **Wak**e and **I**mpedance **S**olver

[![Documentation Status](https://readthedocs.org/projects/wakis/badge/?version=latest)](https://wakis.readthedocs.io/en/latest/?badge=latest)

`wakis` is a **3D Time-domain Electromagnetic solver** that solves the Integral form of Maxwell's equations using the Finite Integration Technique (FIT) numerical method. It computes the longitudinal and transverse **wake potential and beam-coupling impedance** from the simulated electric and magnetic fields. It is also a multi-purpose solver, capable of simulating planewaves interaction with nano-structures, optical diffraction, and much more!

## About
:rocket: Some of `wakis` features:
* Material tensors: permittivity $\varepsilon$, permeability $\mu$, conductivity $\sigma$. Possibility of anisotropy.
* CAD geometry importer (`.stl` format) for definition of embedded boundaries and material regions, based on [pyvista](https://github.com/pyvista/pyvista) 
* Boundary conditions: PEC, PMC, Periodic, ABC-FOEXTRAP
* Different time-domain sources: particle beam, planewave, gaussian wavepacket
* 100% python, fully exposed API (material tensors, fields $E$, $H$, $J$). Matrix operators based on `numpy` and `scipy.sparse` routines ensure fast calculations.
* 1d, 2d, 3d built-in plotting on-the-fly
* Optimized memory consumption
* GPU acceleration: _coming soon_

## How to use
:book: Documentation, powered by `sphinx`, is available at [wakis.readthedocs.io](wakis.readthedocs.io) 

Check :file_folder: `benchmarks/` for beam-coupling impedance calculations & comparisons to the commercial tool CST® Wakefield solver:
* PEC cubic cavity below cutoff (mm) and above cutoff (cm)
* Conductive cubic cavity below cutoff
* Lossy pillbox cavity (cylindrical) above cutoff
* Simulations using beams with different relativistic $\beta$

Check :file_folder: `examples/` for different physical applications
* Planewave interacting with a PEC or dielectric sphere
* Gaussian wavepacket travelling through vacuum / dielectric
* Custom perturbation interacting with PEC geometry 

For specific needs, please contact the developer :woman_technologist: :wave:
* elena.de.la.fuente.garcia@cern.ch

## Motivation
The determination of electromagnetic wakefields and their impact on accelerator performance is a significant issue in current accelerator components. These wakefields, which are generated within the accelerator vacuum chamber as a result of the interaction between the structure and a passing beam, can have significant effects on the machine. 
These effects can be characterized through the beam coupling impedance in the frequency domain and wake potential in the time domain. Accurate evaluation of these properties is essential for predicting dissipated power and maintaining beam stability. 
`wakis` is an open-source tool that can compute wake potential and impedance for both longitudinal and transverse planes for general 3D structures. `wakis` was first presented at IPAC23 conference as a post-processing tool: 

* https://doi.org/10.18429/JACoW-IPAC2023-WEPL170
  
It has now evolved from a post-processing tool to a full 3D electromagnetic, time domain solver.

## Installation
This section explains how to set up the environment to start using `wakis` 3d electromagnetic time domain simulations, and beam-coupling impedance computations

#### Developers: Download wakis repository from Github
```
# SSH:
git clone git@github.com:ImpedanCEI/FITwakis.git

# or HTTPS:
git clone https://github.com/ImpedanCEI/FITwakis.git
```

#### Users: pip install from PyPI

_Coming soon_


<table style="border-collapse: collapse; width: 100%; height: 30.3828px;" border="1"><colgroup><col style="width: 33.2997%;"><col style="width: 33.2997%;"><col style="width: 33.2997%;"></colgroup>
<tbody>
<tr style="height: 30.3828px;">
<td style="height: 30.3828px;"><img src="https://indico.cern.ch/event/1383771/attachments/2805685/4895678/H_3D.gif" alt=""></td>
<td style="height: 30.3828px;"><img style="display: block; margin-left: auto; margin-right: auto;" src="https://indico.cern.ch/event/1383771/attachments/2805685/4895677/abc_abs.gif"></td>
<td style="height: 30.3828px;"><img src="https://indico.cern.ch/event/1383771/attachments/2805685/4895682/H2d_ZX_ABCz+.gif" alt=""></td>
</tr>
</tbody>
</table>


