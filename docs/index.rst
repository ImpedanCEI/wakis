Welcome to ``wakis`` documentation
-----

> **wakis**: 3D Time-domain **Wak**\e and **I**\mpedance **S**\olver

``wakis`` is a **3D Time-domain Electromagnetic solver** that solves the Integral form of Maxwell's equations using the Finite Integration Technique (FIT) numerical method. It computes the longitudinal and transverse **wake potential and beam-coupling impedance** from the simulated electric and magnetic fields. It is hence focused on simulations for particle accelerator components, but it is also a multi-purpose solver; capable of simulating planewaves interaction with nano-structures, optical diffraction, and much more!

Some of ``wakis`` features:

* Material tensors: permittivity :math:`\varepsilon`, permeability :math:`\mu`, conductivity :math:`\sigma`. Possibility of anisotropy.
* CAD geometry importer (``.stl`` format) for definition of embedded boundaries and material regions, based on `pyvista <https://github.com/pyvista/pyvista>`_
* Boundary conditions: PEC, PMC, Periodic, ABC-FOEXTRAP
* Different time-domain sources: particle beam, planewave, gaussian wavepacket
* 100% python, fully exposed API (material tensors, fields :math:`E`, :math:`H`, :math:`J`). Matrix operators based on ``numpy`` and ``scipy.sparse`` routines ensure fast calculations.
* 1d, 2d, 3d built-in plotting on-the-fly
* Optimized memory consumption
* GPU acceleration: *coming soon*

The source code is available in the ``wakis`` `GitHub repository <https://github.com/ImpedanCEI/wakis>`_.

.. toctree::
   :caption: Table of contents
   :maxdepth: 3

   installation
   usersguide
   physicsguide
   wakis 
