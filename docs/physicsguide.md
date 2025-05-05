# Physic's guide

## 1. Introduction

### 🎯 Motivation

In modern accelerators, precise knowledge of beam-coupling impedance and wakefields is essential to ensure beam quality, mitigate heating, and optimize component design. Analytical methods, while powerful, often fall short for realistic 3D geometries — this is where full electromagnetic solvers like Wakis become indispensable.


### 📚 Background

The evaluation of beam-coupling impedance and wakefields is critical in the design and operation of particle accelerators. As charged particle bunches traverse beamline structures, they interact with geometric discontinuities and material changes, generating electromagnetic fields known as **wakefields**.

These wakefields can:
- Degrade beam transmission and stability (causing coherent instabilities)
- Induce energy spread and emittance growth
- Cause beam-induced heating and power loss on the accelerator devices

The **frequency-domain** description of these interactions is the **beam-coupling impedance**, an intrinsic property of each accelerator structure that quantifies its wakefield response to the beam. Obtaining an accurate representation of the **beam-coupling impedance** of each device -and of its time-domain inverse fourier transforms the wake function-, is crucial to evaluate the collective effects of the beam in the accelerator, and serves as an input quantity for beam dynamics simulations performed to study beam stability and properties under different scenarios. 

::: tip
Beam-dynamic multiparticle simulations and tracking to study collective effects can be performed with another CERN in-house python suite of packages [Xsuite](https://github.com/xsuite/xsuite)
:::


Furthermore, the impedance of each accelerator device is needed to estimate the beam induced power deposition on the devices parts, being crucial for the mechanical design, vacuum compliance and cooling requirements. 

::: info
The beam induced heating estimation can be performed easily with a package of the Wakis ecosystem [`bihc`](https://github.com/ImpedanCEI/BIHC)
Beam Induced Heating Computation (BIHC) tool is a package that allows the estimation of the dissipated power due to the passage of a particle beam inside an accelerator component.

The dissipated power value depends on the characteristics of the particle beam (beam spectrum and intensity) and on the characteristics of the consdiered accelerator component (beam-coupling impedance).
:::

#### Analytical vs Numerical Approaches


Analytical methods offer elegant and accurate solutions for the beam-coupling impedance of **simplified geometries**. These are invaluable for physical insight and quick estimates in idealized cases such as:

- **Resistive wall pipes**  
  Analytical expressions exist for round and elliptical beam pipes, characterizing impedance due to finite wall conductivity [Yokoya, 1993](https://cds.cern.ch/record/248630), [Migliorati et al., 2019](https://cds.cern.ch/record/2705426).

- **RF cavities**  
  Resonant mode impedance and wake contributions can be calculated using modal expansion or equivalent circuits [Hofmann & Zotter, 1990](https://cds.cern.ch/record/196446), [Jensen, 2014](https://cds.cern.ch/record/1982429).

- **Tapers and step transitions**  
  Important for matching sections and bellows, these can be described with slowly varying cross-section approximations [Yokoya, 1990](https://cds.cern.ch/record/210347), [Stupakov, 2007](https://link.aps.org/doi/10.1103/PhysRevSTAB.10.094401), [Palumbo et al., 1994](https://arxiv.org/abs/physics/0309023).


However, real-world accelerator components like:
- Beam Position Monitors (BPMs)
- Injection kickers
- RF-shielded bellows
- Complex collimators or cavities

...often have **no analytical solution**. These must be addressed through **full 3D numerical simulations** of Maxwell’s equations [Maxwell, 1997](https://cds.cern.ch/record/2769595), using finite differences, finite elements, or finite integration.


::: tip 🧑‍🏫 Why Use Time-Domain Solvers?

Wakis employs a **time-domain approach** using the Finite Integration Technique (FIT), which offers key benefits:
- **Broadband response** in a single simulation
- Natural support for transient excitation (e.g., Gaussian bunch)
- Efficient use of explicit solvers with GPU and MPI support

This makes Wakis well-suited for impedance characterization across a **wide frequency range**, complementing frequency-domain solvers like CST or HFSS.

For a broader overview of impedance modeling, see [Metral et al., 2020](https://cds.cern.ch/record/2743945).

:::


## 2. Electromagnetic Formulation

### ⚡🧲 Maxwell's Equations (Integral Form)

Wakis numerically solves Maxwell's equations in their **integral form**, which is fundamental to the Finite Integration Technique (FIT). This approach preserves the physical laws in their conservative form and naturally fits the discretization on structured grids.

The time-domain integral form of Maxwell's equations is:

$$
\begin{subequations}\label{eq:Maxwell}
\begin{align}
\oint_{\partial A} \mathbf{E}\cdot \mathrm{d}\mathbf{s} &= -\iint_{A}\frac{\partial \mathbf{B}}{\partial t}\cdot \mathrm{d}\mathbf{A} \tag{1a}\\[6pt]
\oint_{\partial A} \mathbf{H}\cdot \mathrm{d}\mathbf{s} &= \iint_{A}\left(\frac{\partial \mathbf{D}}{\partial t} + \mathbf{J}\right)\cdot \mathrm{d}\mathbf{A} \tag{1b}\\[6pt]
\oiint_{\partial V} \mathbf{B}\cdot \mathrm{d}\mathbf{A} &= 0 \tag{1c}\\[6pt]
\oiint_{\partial V} \mathbf{D}\cdot \mathrm{d}\mathbf{A} &= \iiint_{V}\rho\, \mathrm{d}V \tag{1d}\\[6pt]
\mathbf{D} = \varepsilon \mathbf{E},\quad 
\mathbf{B} &= \mu \mathbf{H},\quad 
\mathbf{J} = \sigma \mathbf{E} + \rho\mathbf{v} \tag{1e}
\end{align}
\end{subequations}
$$

These equations relate:
- The **circulation** of fields around surfaces to the **flux** of their time derivatives through those surfaces.
- The **flux** of $\mathbf{D}$ and $\mathbf{B}$ through closed surfaces to the **charge and magnetic monopole content** (the latter being zero).

Wakis assumes **initially charge-free conditions**, where:
- $\rho = 0$ and $\mathbf{v} = 0$ (static beam)
- The divergence equations (1c–1d) are **satisfied implicitly** by construction.

### 🧱 Discretization with the Finite Integration Technique (FIT)

Wakis discretizes the integral form of Maxwell's equations using the **Finite Integration Technique (FIT)** on a structured three-dimensional Cartesian grid:

$$
N_\text{cells} = N_x \times N_y \times N_z
$$

This approach maps:
- Line integrals → to grid **edges**
- Surface integrals → to grid **faces**
- Volume integrals → to grid **cells**

The resulting discretization yields the **Maxwell Grid Equations (MGE)**, which evolve the fields on a **staggered Yee-like lattice**. Specifically:
- $\vec{E}$ and $\vec{H}$ components are stored on **edges**
- $\vec{D}$ and $\vec{B}$ components are defined on **faces**
- Scalar quantities such as charge density reside at **cell centers**

This structure ensures that discrete curl, divergence, and gradient operators obey their continuous counterparts' conservation properties, which is critical for numerical stability and accuracy.


#### Maxwell Grid Equations (MGE)

In FIT, the continuous Maxwell equations are converted into discrete update rules for the electric and magnetic fields:

$$
\mathbf{C} \vec{E} = -\frac{d}{dt} \vec{B}
\qquad
\mathbf{C}^T \vec{H} = \frac{d}{dt} \vec{D} + \vec{J}
$$

Where:
- $\mathbf{C}$ is the discrete **curl matrix**
- $\mathbf{C}^T$ is its transpose (used for magnetic curl)
- $\vec{E}, \vec{H}, \vec{D}, \vec{B}, \vec{J}$ are **1D vectors** of length $3N_\text{cells}$ stored in **lexicographic order**

The curl matrices are huge sparse matrices with bands of +1 and -1. They are implemented efficiently in Wakis using `scipy.sparse` CSR format. The electromagnetic fields are stored in a structured `Field` object in Wakis:
- Supports `.toarray()` and `.fromarray()` for 1d lexicographic to 3d grid. Automatically reshaped to the simulation grid
- Interoperates with CuPy and MPI through magic methods and flags.
- `.inspect()` and other handy plotting methods 
- Custom magic methods for multiplication, addition, division.
- Getters and setters for easy and optimized access on-the-fly: apply intial conditions, sources, save states... 

This formulation enables stable, explicit time stepping and modular extensions to lossy media, materials, and sources.

#### Material tensors and grid information

Wakis distinguishes between **primal** and **dual** grid geometries as part of its Finite Integration Technique (FIT) formulation. Each quantity is mapped to a geometric entity and stored as a sparse diagonal matrix to enable fast, memory-efficient computations:

| Quantity                  | Description                                    |
|--------------------------|------------------------------------------------|
| $\mathbf{M}_\varepsilon$ | Diagonal matrix of permittivities              |
| $\mathbf{M}_\mu^{-1}$    | Diagonal matrix of inverse permeabilities      |
| $\mathbf{M}_\sigma$      | Diagonal matrix of electrical conductivities   |
| $\mathbf{M}_l$, $\mathbf{M}_A$ | Edge lengths and face areas (primal/dual) |

To support **anisotropic materials** and **imported geometries**, Wakis stores the raw material data in structured `Field` objects — similar to 3D tensors — where values can be specified independently along the **x, y, and z directions** for each cell.

Before time-stepping, these directional fields are assembled into the corresponding **sparse diagonal matrices** using `scipy.sparse.diags`. This preserves the **locality of FIT updates** while enabling efficient CPU and GPU execution.

Wakis also supports spatially varying media, embedded materials from CAD imports, and subpixel smoothing, ensuring accurate representation of complex geometries and composite structures.

### 🕒 Time-Stepping Routine

Wakis uses the **Leapfrog scheme**, a second-order accurate and explicit time integrator. This method updates the magnetic and electric fields in a staggered fashion:

$$
\begin{subequations}\label{eq:timestepping}
\begin{align}
\mathbf{h}^{n+1} &= \mathbf{h}^n - \Delta t \, \widetilde{\mathbf{D}}_s \, \mathbf{M}_\mu^{-1} \, \mathbf{D}_A^{-1} \, \mathbf{C} \, \mathbf{e}^{n+0.5} \tag{2a} \\[6pt]
\mathbf{e}^{n+1.5} &= \mathbf{e}^{n+0.5} + \Delta t \, \mathbf{D}_s \, \widetilde{\mathbf{M}}_\varepsilon^{-1} \, \widetilde{\mathbf{D}}_A^{-1} \, \widetilde{\mathbf{C}} \, \mathbf{h}^n 
- \widetilde{\mathbf{M}}_\varepsilon^{-1} \, \mathbf{j}_{\text{src}}^n 
- \widetilde{\mathbf{M}}_\varepsilon^{-1} \, \widetilde{\mathbf{M}}_\sigma \, \mathbf{e}^{n+0.5} \tag{2b}
\end{align}
\end{subequations}
$$

Where:
- $\mathbf{C}$ and $\widetilde{\mathbf{C}}$ are discrete curl matrices on the primal and dual grids
- $\mathbf{D}_s$, $\mathbf{D}_A$ are diagonal matrices containing metric coefficients (edge lengths and face areas)
- Material properties enter through $\mathbf{M}_\mu$, $\mathbf{M}_\varepsilon$, and $\mathbf{M}_\sigma$

The timestep $\Delta t$ is constrained by:
- The **Courant–Friedrichs–Lewy (CFL)** condition
- Material relaxation times (for dispersive or lossy media)

Most matrix operations are precomputed and cached, enabling large-scale simulations with modest memory usage (~20M cells in <8 GB RAM/GPU).

### 🔌 Sources and Initial Conditions

Wakis allows arbitrary initial conditions on $\vec{E}$, $\vec{H}$, and $\vec{J}$. Sources can be defined in multiple ways:
- **User-defined time-dependent callbacks** injected at each step
- **Predefined source types**: Gaussian beams, dipoles, plane waves, laser pulses, available in `sources.py`

#### Example: Gaussian Beam Current $J_z$

A rigid Gaussian bunch current is modeled as a line distribution:

$$
\mathbf{J}_z(x_{\text{src}}, y_{\text{src}}, \vec{z}) = 
\frac{q \beta c}{\sqrt{2\pi} \sigma_z} \, 
\exp\left( -\frac{(\vec{s} - s_0)^2}{2\sigma_z^2} \right)
$$

with:
- $\vec{s} = \vec{z} - \beta c t$: beam-frame coordinate
- $s_0 = z_{\min} - \beta c t_{\text{inj}}$: center of bunch

This supports both **ultra-relativistic** ($\beta \approx 1$) and **low-beta** scenarios.


### 🧊🔚 Boundary Conditions

Wakis supports several boundary condition types:
- **PEC (Perfect Electric Conductor)**: enforces $\vec{E}_{\parallel} = 0$
- **PMC (Perfect Magnetic Conductor)**: enforces $\vec{H}_{\parallel} = 0$
- **Periodic BCs**: implemented with synchronized ghost cells
- **PML (Perfectly Matched Layers)**: absorbing layers using graded conductivity profiles for reflection-free truncation

PMLs follow the formulation by Berenger [1994] and are ramped using smooth profiles [Oskooi et al., 2008] to reach adiabatic reflection.


### 📥🗿 Geometry Importing & Embedded Boundaries

Wakis integrates with [**PyVista**](https://docs.pyvista.org/) to import CAD geometries in `.STL`, `.STEP`, or `.OBJ` formats. The mesh is overlaid onto the simulation domain and mapped onto the Cartesian grid using:
- First-order subpixel smoothing (MEEP-inspired)
- Assignment of material properties ($\varepsilon_r$, $\mu_r$, $\sigma$) to each cell in $x$, $y$, and $z$

Future versions aim to include a more advanced meshing algorithm for improved fidelity near corners and edges.

### 🚀 GPU and MPI Parallelization

Wakis supports:
- **GPU acceleration** using [**CuPy**](https://cupy.dev/) and `cupyx.scipy.sparse`
- Drop-in replacement of NumPy/SciPy operations when `use_gpu=True`
- **MPI parallelization** using [**mpi4py**](https://mpi4py.readthedocs.io/)
- Efficient longitudinal domain decomposition with ghost-cell synchronization
- Seamless integration with **multi-GPU** setups, with performance benefits on 100k+ timesteps

::: info 👩‍💻 Developer Notes

Wakis is:
- Fully open-source and available on [GitHub](https://github.com/ImpedanCEI/wakis)
- Packaged on [PyPI](https://pypi.org/project/wakis/)
- Documented with `Sphinx` and hosted on `ReadTheDocs`: https://wakis.readthedocs.io/ 
- Includes **CI/CD**, with end-to-end tests running nightly, tagged **versioned releases**, and numerous **ready-to-run examples** in both Python scripts and notebooks

:::

## 3. Wake Potential and Impedance calculation

Wakis computes beam coupling impedance from time-domain electromagnetic field simulations by evaluating the wakefields generated by a moving charged particle (or bunch) as it traverses an accelerator structure.

### 📚 Physical Definition: Wake function and Impedance

Let:
- $\vec{r_s} = (x_s, y_s, z_s)$ be the position of a **source particle**
- $\vec{r_t} = (x_t, y_t, z_t)$ the **test particle** position
- $s = z_t - z_s$ the **longitudinal separation**

The **wake function** $w(\vec{r_s}, \vec{r_t}, s)$ quantifies the electromagnetic interaction between particles:

\[
w(\vec{r_s}, \vec{r_t}, s) = \frac{1}{q_s q_t} \int_{-\infty}^{\infty} \vec{F}_{\text{Lorentz}} \cdot d\vec{z}
\]

Its Fourier transform yields the **longitudinal impedance**:

\[
Z_{\parallel}(\omega) = \int_{-\infty}^{\infty} w_{\parallel}(s) \, e^{-i \omega s / c} \frac{ds}{c}
\]

However, since a beam is a **distributed source**, the wake function is not directly accessible. Instead, we compute the **wake potential** $W(s)$ generated by the full bunch distribution.

### 📈 Wake Potential from 3D electromganetic simulations

The **wake potential** is calculated by integrating the electric and magnetic fields seen by a test particle as it follows behind the source:

\[
W(s) = \frac{1}{q_s} \int_{-\infty}^{\infty} \left[ E_z(z, t) + c \, \vec{e}_z \times \vec{B}(z, t) \right]_{t = (s + z)/c} \, dz
\]

For ultra-relativistic beams, the transverse component vanishes, and the expression simplifies to:

\[
W_\parallel(s) = \frac{1}{q_s} \int_{-\infty}^{\infty} E_z(z, t = (s + z)/c) \, dz
\]

The **transverse wake potential** is recovered via the **Panofsky-Wenzel theorem**:

\[
W_{\perp,\alpha}(s) = \frac{\partial}{\partial \alpha} \int_{-\infty}^{s} W_\parallel(s') \, ds', \quad \alpha = x, y
\]

Wakis implements this gradient using second-order finite differences.

#### Transverse Decomposition

Wakis supports transverse wake analysis:
\[
W_{\perp,x}(x, y, s) = W_C(s) + W_D(s) \Delta x_s + W_Q(s) \Delta x_t + \mathcal{O}(x^2)
\]

- $W_D$: **dipolar wake**, linear in source offset
- $W_Q$: **quadrupolar wake**, linear in test offset
- $W_C$: **coherent term**, for asymmetric geometries

These are extracted by sampling field responses at multiple $(x_s, x_t)$ combinations.


### 🔁 From Wake to Impedance

Given the bunch profile $\lambda(s)$ and the wake potential $W(s)$, the beam coupling impedance is computed in Fourier space:

- **Longitudinal impedance**:

\[
Z_\parallel(\omega) = c \cdot \frac{\mathcal{F}[W_\parallel(s)]}{\mathcal{F}[\lambda(s)]}
\]

- **Transverse impedance**:

\[
Z_\perp(\omega) = -i c \cdot \frac{\mathcal{F}[W_\perp(s)]}{\mathcal{F}[\lambda(s)]}
\]

where $\mathcal{F}$ denotes the Fourier transform. Wakis uses `numpy.fft` with a Hanning window and zero-padding for smooth frequency analysis.


::: warning 🧪 Open-Source Solver Compatibility

The wake and impedance analysis is performed within the `WakeSolver` class in `Wakesolver.py`. It is modular and can be used with other EM solvers besides Wakis' `SolverFIT3D` output:
- It has been tested with **WarpX** EM fields. WarpX is a powerful open-source PIC solver for full 3D EM fields
- It has been benchmarked with **CST** Wakefield solver, using both EM field ouput (time-domain field monitors) and calculated wake potential and impedance. > See the full wake analysis benchmark in [IPAC'23 proceedings, E. de la Fuente](https://doi.org/10.18429/JACoW-IPAC2023-WEPL170).
- **Interoperability**: Wakis can read field maps in HDF5, CSV, or NumPy format
- **Subvolume extraction** and interpolation are supported for field-based post-processing

:::
