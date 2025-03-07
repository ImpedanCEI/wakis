<img src="https://github.com/ImpedanCEI/wakis/blob/main/docs/img/wakis-logo-pink.png" alt="wakis-logo-light-background" width="240">

> **Wak**e and **I**mpedance **S**olver

[![Documentation Status](https://readthedocs.org/projects/wakis/badge/?version=latest)](https://wakis.readthedocs.io/en/latest/?badge=latest)
![Tests badge](https://github.com/impedanCEI/wakis/actions/workflows/nightly_tests_CPU.yml/badge.svg)
[![codecov](https://codecov.io/github/elenafuengar/wakis/graph/badge.svg?token=7QPYJC23A0)](https://codecov.io/github/elenafuengar/wakis)

![PyPI - Version](https://img.shields.io/pypi/v/wakis?style=flat-square&color=blue)
![PyPI - License](https://img.shields.io/pypi/l/wakis?style=flat-square&color=pink)
![Tokei - LOC](https://tokei.rs/b1/github/ImpedanCEI/wakis?category=code?/style=square&color=green)
![Zenodo DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14988677.svg)


`wakis` is a **3D Time-domain Electromagnetic solver** that solves the Integral form of Maxwell's equations using the Finite Integration Technique (FIT) numerical method. It computes the longitudinal and transverse **wake potential and beam-coupling impedance** from the simulated electric and magnetic fields. It is also a multi-purpose solver, capable of simulating planewaves interaction with nano-structures, optical diffraction, and much more!

## About
:rocket: Some of `wakis` features:
* Wake potential and impedance calculations for particle beams with different relativistic $\beta$
* Material tensors: permittivity $\varepsilon$, permeability $\mu$, conductivity $\sigma$. Possibility of anisotropy.
* CAD geometry importer (`.stl` format) for definition of embedded boundaries and material regions, based on [`pyvista`](https://github.com/pyvista/pyvista) 
* Boundary conditions: PEC, PMC, Periodic, ABC-FOEXTRAP
* Different time-domain sources: particle beam, planewave, gaussian wavepacket
* 100% python, fully exposed API (material tensors, fields $E$, $H$, $J$). Matrix operators based on `numpy` and `scipy.sparse` routines ensure fast calculations.
* 1d, 2d, 3d built-in plotting on-the-fly
* Optimized memory consumption
* GPU acceleration using `cupy/cupyx`
* Perfect matching layer (PML) 
* Wakefield extrapolation with PIML [`iddefix`](https://github.com/ImpedanCEI/IDDEFIX) evolutionary algorithm
* Non-equidistant Filon Fourier integration with [`neffint`](https://github.com/ImpedanCEI/neffint)
* Beam-induced heating estimation due to impedance with [`bihc`](https://github.com/ImpedanCEI/BIHC)

## How to use
:book: Documentation, powered by `sphinx`, is available at [wakis.readthedocs.io](https://wakis.readthedocs.io/en/latest/index.html)

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
`wakis` is an open-source tool that can compute wake potential and impedance for both longitudinal and transverse planes for general 3D structures. 

* `wakis` was first presented at IPAC23 conference as a post-processing tool: https://doi.org/10.18429/JACoW-IPAC2023-WEPL170
  
* It has now evolved from a post-processing tool to a full 3D electromagnetic, time domain solver; and has been presented at the 14th International Computational Accelerator Physics Conference (ICAP24): https://indico.gsi.de/event/19249/contributions/82636/

## Installation
This section explains how to set up the environment to start using `wakis` 3d electromagnetic time domain simulations, and beam-coupling impedance computations

#### Developers: Download wakis repository from Github
```
# SSH:
git clone git@github.com:ImpedanCEI/wakis.git

# or HTTPS:
git clone https://github.com/ImpedanCEI/wakis.git

# Create python environment
cd wakis/
conda create --name wakis-env python=3.9
conda activate wakis-env
pip install -r requirements.txt
```

#### Users: pip install from PyPI
For basic usage, simply do:
```
pip install wakis
```
For some extra features, including interactive 3d plots in python notebooks, use: 
```
pip install wakis['notebook']
```




