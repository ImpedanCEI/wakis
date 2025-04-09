<img src="https://github.com/ImpedanCEI/wakis/blob/main/docs/img/wakis-logo-pink.png" alt="wakis-logo-light-background" width="240">

> Open-source **Wak**e and **I**mpedance **S**olver

[![Documentation Status](https://readthedocs.org/projects/wakis/badge/?version=latest)](https://wakis.readthedocs.io/en/latest/?badge=latest)
![Tests badge](https://github.com/impedanCEI/wakis/actions/workflows/nightly_tests_CPU.yml/badge.svg)
[![codecov](https://codecov.io/github/elenafuengar/wakis/graph/badge.svg?token=7QPYJC23A0)](https://codecov.io/github/elenafuengar/wakis)

![PyPI - Version](https://img.shields.io/pypi/v/wakis?style=flat-square&color=blue)
![PyPI - License](https://img.shields.io/pypi/l/wakis?style=flat-square&color=pink)
![Tokei - LOC](https://tokei.rs/b1/github/ImpedanCEI/wakis?category=code?/style=square&color=green)
![Zenodo DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14988677.svg)


`wakis` is a **3D Time-domain Electromagnetic solver** that solves the Integral form of Maxwell's equations using the Finite Integration Technique (FIT) numerical method. It computes the longitudinal and transverse **wake potential and beam-coupling impedance** from the simulated electric and magnetic fields. It is also a multi-purpose solver, capable of simulating planewaves interaction with nano-structures, optical diffraction, and much more!

## About
🚀 Some of `wakis` features:
* Wake potential and impedance calculations for particle beams with different relativistic $\beta$
* Material tensors: permittivity $\varepsilon$, permeability $\mu$, conductivity $\sigma$. Possibility of anisotropy.
* CAD geometry importer (`STL` & `STEP` format) for definition of embedded boundaries and material regions, based on [`pyvista`](https://github.com/pyvista/pyvista) 
* Boundary conditions: PEC, PMC, Periodic, ABC-FOEXTRAP, Perfect Matched Layers (PML)
* Different time-domain sources: particle beam, planewave, gaussian wavepacket
* 100% python, fully exposed API (material tensors, fields $E$, $H$, $J$). Matrix operators based on `numpy` and `scipy.sparse` routines ensure fast calculations.
* 1d, 2d, 3d built-in plotting on-the-fly
* Optimized memory consumption
* GPU acceleration using `cupy/cupyx`
* CUDA-aware MPI parallelization with `mpi4py` and `ipyparallel` *coming soon!*

🧩 Other complementary tools in the ecosystem:
* Wakefield extrapolation via broadband resonator fitting with PIML [`iddefix`](https://github.com/ImpedanCEI/IDDEFIX) evolutionary algorithms
* Non-equidistant Filon Fourier integration with [`neffint`](https://github.com/ImpedanCEI/neffint)
* Beam-induced heating estimation due to impedance with [`bihc`](https://github.com/ImpedanCEI/BIHC)

## How to use
📖 Documentation, powered by `sphinx`, is available at [wakis.readthedocs.io](https://wakis.readthedocs.io/en/latest/index.html)

Check 📁 `examples/` and `notebooks/` for different physical applications:
* Planewave interacting with a PEC or dielectric sphere
* Gaussian wavepacket travelling through vacuum / dielectric
* Custom perturbation interacting with PEC geometry 
* Wakefield simulation of accelerator cavity on CPU, GPU and with MPI

Check 🌐📁 [`wakis-benchmarks/`](https://github.com/ImpedanCEI/wakis-benchmarks) for beam-coupling impedance calculations & comparisons to the commercial tool CST® Wakefield solver:
* PEC cubic cavity below cutoff (mm) and above cutoff (cm)
* Conductive cubic cavity below cutoff
* Lossy pillbox cavity (cylindrical) above cutoff
* Simulations using beams with different relativistic $\beta$

Check 🌐📁 [`CEI-logo/`](https://github.com/ImpedanCEI/CEI-logo) for a fun & complete beam-coupling impedance workflow tutorial, including:
* **001**: Electromagnetic simulation preparation, inspection & 1d, 2d, 3d plotting
* **002**: Wakefield simulation on GPU
* **003**: Extrapolation of wakefield simulation to fully decayed with [`iddefix`](https://github.com/ImpedanCEI/IDDEFIX). Conversion to wake function for beam dynamics simulation with [`neffint`](https://github.com/ImpedanCEI/neffint).
* **004**: Beam-induced heating estimation due to impedance with [`bihc`](https://github.com/ImpedanCEI/BIHC)
This playground was used to generate the CERN [ABP-CEI section](https://indico.cern.ch/event/1519352/contributions/6393071/attachments/3031114/5351148/CEIinfo-20250313.pdf) logo 🎨🖌️

🔁 To be informed of the latest features/bug fixes pushed to `main`, check the [`release.md`](https://github.com/ImpedanCEI/wakis/blob/main/release.md)

For specific needs, please contact the developer 👩‍💻👋
* elena.de.la.fuente.garcia@cern.ch

## Installation
Wakis supports **Python 3.9 - 3.11** and can be installed in any `conda` environment.

📖 **For a detailed installation guide (GPU, MPI setup, FAQs), check our [documentation](https://wakis.readthedocs.io/en/latest/installation.html).**

### Users: Install via PyPI  
For basic usage, simply run:
```bash
pip install wakis
```
For additional features, including **interactive 3D plots in Jupyter notebooks**, use:
```bash
pip install wakis['notebook']
```
To install **complementary tools** in the Wakis ecosystem:
```bash
pip install neffint iddefix bihc
```
💡 **Have a bug, feature request, or suggestion?** Open a [GitHub Issue](https://github.com/ImpedanCEI/wakis/issues) so the community can track it.

### Developers: Contribute to Wakis  
First, [Fork](https://github.com/ImpedanCEI/wakis/fork) the repository and clone it from `main`:
```bash
# SSH:
git clone git@github.com:YourUserName/wakis.git

# or HTTPS:
git clone https://github.com/YourUserName/wakis.git
```
Create a dedicated **conda environment** and install dependencies:
```bash
cd wakis/
conda create --name wakis-env python=3.11
conda activate wakis-env
pip install -r requirements.txt
pip install neffint iddefix bihc  # Optional tools
```
🛠️ **Want to contribute?**  To merge your changes into `main`, create a **Pull Request (PR)** following our [PR template](https://github.com/ImpedanCEI/wakis/blob/main/.github/pull_request_template.md).

## Motivation
🎯 The determination of electromagnetic wakefields and their impact on accelerator performance is a significant issue in current accelerator components. These wakefields, which are generated within the accelerator vacuum chamber as a result of the interaction between the structure and a passing beam, can have significant effects on the machine. 
These effects can be characterized through the beam coupling impedance in the frequency domain and wake potential in the time domain. Accurate evaluation of these properties is essential for predicting dissipated power and maintaining beam stability. 
`wakis` is an open-source tool that can compute wake potential and impedance for both longitudinal and transverse planes for general 3D structures. 

* 🌱 `wakis` was firstly presented at the **International Particle Accelerator Conference in 2023** (IPAC23) as a post-processing tool: https://doi.org/10.18429/JACoW-IPAC2023-WEPL170
  
* 🌳 It has now evolved from a post-processing tool to a full 3D electromagnetic, time domain solver; and has been presented at the **14th International Computational Accelerator Physics Conference in 2024** (ICAP24): https://indico.gsi.de/event/19249/contributions/82636/

## Citing `Wakis`
🔖 Each Wakis release is linked to a [Zenodo](https://zenodo.org/records/15011421) publication under a unique [DOI](https://doi.org/10.5281/zenodo.15011421). If you are using Wakis in your scientific research, please help our scientific visibility by citing this work:

> [1] E. de la Fuente Garcia et. al., “Wakis: 3D Electromagnetic Time-Domain Wakefield and Impedance Solver”. Zenodo, Mar. 12, 2025. doi: [10.5281/zenodo.15011421](https://doi.org/10.5281/zenodo.15011421). 



