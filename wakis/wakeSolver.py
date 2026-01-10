# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

import time
import os
import glob
import shutil
import h5py
import numpy as np
from tqdm import tqdm
from scipy.constants import c as c_light

from .logger import Logger


class WakeSolver:
    """Class for wake potential and impedance
    calculation from 3D time domain E fields
    '''

    def __init__(self, wakelength=None, q=1e-9, sigmaz=1e-3, beta=1.0,
                 xsource=0., ysource=0., xtest=0., ytest=0., 
                 chargedist=None, ti=None, 
                 compute_plane='both', skip_cells=0, add_space=None, 
                 Ez_file='Ez.h5', save=True, results_folder='results/',
                 verbose=0, counter_moving=False):
        '''
        Parameters
        ----------
        wakelength : float, optional
            Wakelength to be simulated. If not provided, it will be calculated from the Ez field data.
        q : float
            Beam total charge in [C]
        sigmaz : float
            Beam sigma in the longitudinal direction [m]
        beta : float, deafult 1.0
            Ratio of beam's velocity to the speed of light c [a.u.]
        xsource : float, default 0.
            Beam center in the transverse plane, x-dir [m]
        ysource : float, default 0.
            Beam center in the transverse plane, y-dir [m]
        xtest : float, default 0.
            Integration path center in the transverse plane, x-dir [m]
        ytest : float, default 0.
            Integration path center in the transverse plane, y-dir [m]
        ti : float, optional
            Injection time, when beam enters domain [s]. If not provided,
            the default value ti=8.53*sigmaz will be used
        chargedist : dict or str, default None
            If not provided, an analytic gaussian with sigmaz and q will be used.
            When str, specifies the filename containing the charge distribution data
            When dict, should contain the charge distribution data in keys (e.g.): {'X','Y'}
            'X' : longitudinal coordinate [m]
            'Y' : charge distribution in [C/m]
        Ez_file : str, default 'Ez.h5'
            hdf5 file containing Ez(x,y,z) data for every timestep
        save: bool, default True
            Flag to enable saving the wake potential, impedance and charge distribution
            results in `.txt` files.
            - Longitudinal: WP.txt, Z.txt.
            - Transverse: WPx.txt, WPy.txt, Zx.txt, Zy.txt
            - Charge distribution: lambda.txt, spectrum.txt
        verbose: bool, default 0
            Controls the level of verbose in the terminal output
        counter_moving: bool, default False
            If the test charge is moving in the same or opposite direction to the source

        Attributes
        ----------
        Ezt : ndarray
            Matrix (nz x nt) containing Ez(x_test, y_test, z, t)
            where nz = len(z), nt = len(t)
        s : ndarray
            Wakelegth vector s=c_light*t-z [m] representing the distance between
            the source and the integration point. Goes from -ti*c_light
            to the simulated wakelength where ti is the beam injection time.
        WP : ndarray
            Longitudinal wake potential WP(s) [V/pC]
        WP_3d : ndarray
            Longitudinal wake potential in 3d WP(x,y,s). Shape = (2*n+1, 2*n+1, len(s))
            where n = n_transverse_cells and s the wakelength array [V/pC]
        Z : ndarray
            Longitudinal impedance [Ohm] computed by the fourier-transformation of the
            longitudinal component of the wake potential, which is divided by the
            fourier-transformed charge distribution line function lambda(s) using a
            single-sided DFT with 1000 samples.
        WPx : ndarray
            Trasnverse wake potential in x direction WPx(s) [V/pC]
        WPy : ndarray
            Transverse wake potential in y direction WP(s) [V/pC]
        Zx : ndarray
            Trasnverse impedance in x-dir Zx(f) [Ohm]
        Zy : ndarray
            Transverse impedance in y-dir Zy(f) [Ohm]
        lambdas : ndarray
            Linear charge distribution of the passing beam λ(s) [C/m]
        lambdaf : ndarray
            Charge distribution spectrum λ(f) [C]
        dx : float
            Ez field mesh step in transverse plane, x-dir [m]
        dy : float
            Ez field mesh step in transverse plane, y-dir [m]
        x : ndarray
            vector containing x-coordinates for field monitor [m]
        y : ndarray
            vector containing y-coordinates for field monitor [m]
        n_transverse_cells : int, default 1
            Number of transverse cells used for the 3d calculation: 2*n+1
            This determines de size of the 3d wake potential

        """

        # beam
        self.q = q
        self.sigmaz = sigmaz
        self.beta = beta
        self.v = self.beta * c_light
        self.xsource, self.ysource = xsource, ysource
        self.xtest, self.ytest = xtest, ytest
        self.chargedist = chargedist
        self.ti = ti
        self.skip_cells = skip_cells
        self.compute_plane = compute_plane
        self.DE_model = None

        self.counter_moving = counter_moving

        if add_space is not None:  # legacy support for add_space
            self.skip_cells = add_space

        # Injection time
        if ti is not None:
            self.ti = ti
        else:
            # ti = 8.548921333333334*self.sigmaz/self.v  #injection time as in CST for beta = 1
            ti = (
                8.548921333333334 * self.sigmaz / (np.sqrt(self.beta) * self.v)
            )  # injection time as in CST for beta <=1
            self.ti = ti

        # field
        self.Ez_file = Ez_file
        self.Ez_hf = None
        self.Ezt = None  # Ez(x_t, y_t, z, t)
        self.t = None
        self.xf, self.yf, self.zf = None, None, None  # field subdomain
        self.x, self.y, self.z = None, None, None  # full simulation domain

        # solver init
        self.wakelength = wakelength
        self.s = None
        self.lambdas = None
        self.WP = None
        self.WP_3d = None
        self.n_transverse_cells = 1
        self.WPx, self.WPy = None, None
        self.f = None
        self.Z = None
        self.Zx, self.Zy = None, None
        self.lambdaf = None

        # user
        self.verbose = verbose
        self.logger = Logger()
        self.save = save
        self.folder = results_folder
        if not self.folder.endswith("/"):
            self.folder += "/"
        if self.Ez_file is None:
            self.Ez_file = self.folder + "Ez.h5"
        if self.save:
            if not os.path.exists(self.folder):
                os.mkdir(self.folder)

        self.assign_logs()

    def solve(self, compute_plane=None, **kwargs):
        """
        Perform the wake potential and impedance for
        longitudinal and transverse plane
        """
        if compute_plane is None:
            compute_plane = self.compute_plane

        for key, val in kwargs.items():
            setattr(self, key, val)

        t0 = time.time()

        if compute_plane.lower() == "both" or "transverse":
            # Obtain longitudinal Wake potential
            self.calc_long_WP_3d()

            # Obtain transverse Wake potential
            self.calc_trans_WP()

            # Obtain the longitudinal impedance
            self.calc_long_Z()

            # Obtain transverse impedance
            self.calc_trans_Z()

        elif compute_plane == "longitudinal":
            # Obtain longitudinal Wake potential
            self.calc_long_WP()

            # Obtain the longitudinal impedance
            self.calc_long_Z()

        # Elapsed time
        t1 = time.time()
        totalt = t1 - t0
        self.log("Calculation terminated in %ds" % totalt)

    def update_long_WP(self, t):
        """WIP
        calculation of wake potential on the fly
        TODO: simplify logic, add transverse WP
        """

        it = int(t / self.dt)
        if it == 0:
            # --- setup once before time loop ---
            # self.s already computed as in your calc_long_WP
            self.s = np.arange(
                -self.ti * self.v, self.wake.wakelength, self.dt * self.v
            )  # 1D array of s-values (m)
            self.WP = np.zeros_like(self.s, dtype=np.float64)  # accumulator

        # --- inside your time loop, after Ez is updated for current timestep `it` ---
        # get Ez at the probe (shape nz,)
        # if you store Ez as Ezt[k, it] style, do:
        Ez_curr = self.E[
            self.Nx // 2, self.Ny // 2, :, "z"
        ]  # or extract from self.E at (xmid, ymid, :) if not using Ezt

        # compute s values for each z where current Ez contributes
        s_vals = self.v * t - self.v * self.ti + np.min(self.z) - self.z
        # convert to fractional index in s-array
        idxf = (s_vals - self.s[0]) / (self.v * self.dt)  # float indices

        # mask in-range contributions
        mask = (idxf >= 0.0) & (idxf <= (len(self.s) - 1))
        if not np.any(mask):
            # no contributions fall into s range this timestep
            pass
        else:
            idxf_m = idxf[mask]
            Ez_m = Ez_curr[mask]

            # integer floor indices
            i0 = np.floor(idxf_m).astype(int)
            frac = idxf_m - i0

            # handle points that land exactly on last bin: we add all to last bin
            last_bin_mask = i0 >= len(self.s) - 1
            if np.any(last_bin_mask):
                # assign entirely to last bin (no i0+1 available)
                self.WP[-1] += np.sum(Ez_m[last_bin_mask]) * self.dz
                # drop those from other accumulation
                keep = ~last_bin_mask
                i0 = i0[keep]
                frac = frac[keep]
                Ez_m = Ez_m[keep]

            # accumulate with linear interpolation weights
            if i0.size:
                # faster: use bincount
                left = np.bincount(
                    i0, weights=Ez_m * (1.0 - frac) * self.dz, minlength=len(self.s)
                )
                right = np.bincount(
                    i0 + 1, weights=Ez_m * frac * self.dz, minlength=len(self.s)
                )
                self.WP += left + right

        if it == self.Nt - 1:
            self.WP = self.WP / (self.q * 1e12)

    def calc_long_WP(self, Ezt=None, **kwargs):
        """
        Obtains the wake potential from the pre-computed longitudinal
        Ez(z,t) field from the specified solver.
        Parameters can be passed as **kwargs.

        Parameters
        ----------
        t : ndarray
            vector containing time values [s]
        z : ndarray
            vector containint z-coordinates [m]
        sigmaz : float
            Beam longitudinal sigma, to calculate injection time [m]
        q : float
            Beam charge, to normalize wake potential
        ti : float, default 8.53*sigmaz/c
            Injection time needed to set the negative part of s vector
            and wakelength
        Ezt : ndarray, default None
            Matrix (nz x nt) containing Ez(x_test, y_test, z, t)
            where nz = len(z), nt = len(t)
        Ez_file : str, default 'Ez.h5'
            HDF5 file containing the Ez(x, y, z) field data
            for every timestep. Needed only if Ezt is not provided.
        """
        for key, val in kwargs.items():
            setattr(self, key, val)

        # Read h5
        if Ezt is not None:
            self.Ezt = Ezt

        elif self.Ez_hf is None:
            self.read_Ez()

        # time variables
        nt = len(self.t)
        dt = self.t[2] - self.t[1]
        ti = self.ti

        # longitudinal variables
        if self.zf is None: self.zf = self.z
        zmax = np.max(self.zf) #should it be domain's edge instead?
        zmin = np.min(self.zf)

        if self.skip_cells != 0:
            zz = slice(self.skip_cells, -self.skip_cells)
        else:
            zz = np.s_[:]
        z = self.zf[zz]
        nz = len(z)

        # Set Wake length and s
        if self.wakelength is not None:
            wakelength = self.wakelength
        else:
            wakelength = nt * dt * self.v - (zmax - zmin) - ti * self.v
            self.wakelength = wakelength

        s = np.arange(-self.ti * self.v, wakelength, dt * self.v)

        self.log("Max simulated time = " + str(round(self.t[-1] * 1.0e9, 4)) + " ns")
        self.log("Wakelength = " + str(round(wakelength, 3)) + "m")

        # Initialize
        WP = np.zeros_like(s)
        keys = list(self.Ez_hf.keys())

        # check for rounding errors
        if nt > len(keys) - 4:
            nt = len(keys) - 4
            self.log("*** rounding error in number of timesteps")

        # Assembly Ez field
        self.log("Assembling Ez field...")
        Ezt = np.zeros((nz, nt))  # Assembly Ez field
        Ez = self.Ez_hf[keys[0]]

        if len(Ez.shape) == 3:
            for n in range(nt):
                Ez = self.Ez_hf[keys[n]]
                Ezt[:, n] = Ez[Ez.shape[0] // 2 + 1, Ez.shape[1] // 2 + 1, zz]

        elif len(Ez.shape) == 1:
            for n in range(nt):
                Ezt[:, n] = self.Ez_hf[keys[n]]
        self.Ezt = Ezt

        # integral of (Ez(xtest, ytest, z, t=(s+z)/c))dz
        print("Calculating longitudinal wake potential WP(s)...")
        with tqdm(total=len(s) * len(z)) as pbar:
            for n in range(len(s)):
                for k in range(nz):
                    ts = (z[k] + s[n]) / self.v - zmin / self.v - self.t[0] + ti
                    it = int(ts / dt)  # find index for t
                    if it < nt:
                        WP[n] = WP[n]+(Ezt[k, it])*self.dz[k]   #compute integral
                    pbar.update(1)

        WP = WP / (self.q * 1e12)  # [V/pC]

        self.s = s
        self.WP = WP

        if self.save:
            np.savetxt(
                self.folder + "WP.txt",
                np.c_[self.s, self.WP],
                header="   s [m]" + " " * 20 + "WP [V/pC]" + "\n" + "-" * 48,
            )

    def calc_long_WP_3d(self, **kwargs):
        """
        Obtains the 3d wake potential from the pre-computed Ez(x,y,z)
        field from the specified solver. The calculation
        Parameters can be passed as **kwargs.

        Parameters
        ----------
        Ez_file : str, default 'Ez.h5'
            HDF5 file containing the Ez(x,y,z) field data for every timestep
        t : ndarray
            vector containing time values [s]
        z : ndarray
            vector containing z-coordinates [m]
        q : float
            Total beam charge in [C]. Default is 1e9 C
        n_transverse_cells : int, default 1
            Number of transverse cells used for the 3d calculation: 2*n+1
            This determines de size of the 3d wake potential
        """
        self.log("\n")
        self.log("Longitudinal wake potential")
        self.log("-" * 24)

        for key, val in kwargs.items():
            setattr(self, key, val)

        # Read h5
        if self.Ez_hf is None:
            self.read_Ez()

        # time variables
        nt = len(self.t)
        dt = self.t[2] - self.t[1]
        ti = self.ti

        # longitudinal varianles
        if self.zf is None: self.zf = self.z
        zmax = np.max(self.zf) 
        zmin = np.min(self.zf)              

        if self.skip_cells != 0:
            zz = slice(self.skip_cells, -self.skip_cells)
        else:
            zz = np.s_[:]
        z = self.zf[zz]
        nz = len(z)

        # Set Wake length and s
        if self.wakelength is not None:
            wakelength = self.wakelength
        else:
            wakelength = nt * dt * self.v - (zmax - zmin) - ti * self.v
            self.wakelength = wakelength

        s = np.arange(-self.ti * self.v, wakelength, dt * self.v)

        self.log(f"* Max simulated time = {np.max(self.t)} s")
        self.log(f"* Wakelength = {wakelength} m")

        # field subvolume in No.cells for x, y
        i0, j0 = self.n_transverse_cells, self.n_transverse_cells
        WP = np.zeros_like(s)
        WP_3d = np.zeros((i0 * 2 + 1, j0 * 2 + 1, len(s)))
        Ezt = np.zeros((nz, nt))
        keys = list(self.Ez_hf.keys())

        # check for rounding errors
        if nt > len(keys) - 4:
            nt = len(keys) - 4
            self.log("*** rounding error in number of timesteps")

        print("Calculating longitudinal wake potential WP(s)")
        with tqdm(total=len(s) * (i0 * 2 + 1) * (j0 * 2 + 1)) as pbar:
            for i in range(-i0, i0 + 1, 1):
                for j in range(-j0, j0 + 1, 1):
                    # Assembly Ez field
                    for n in range(nt):
                        Ez = self.Ez_hf[keys[n]]
                        Ezt[:, n] = Ez[Ez.shape[0] // 2 + i, Ez.shape[1] // 2 + j, zz]

                    # integral of (Ez(xtest, ytest, z, t=(s+z)/c))dz
                    if self.counter_moving:
                        for n in range(len(s)):
                            for k in range(nz):
                                ts = (
                                    (z[-k - 1] - s[n]) / (-1 * self.v)
                                    - zmax / (-1 * self.v)
                                    - self.t[0]
                                    + ti
                                )
                                it = int(ts / dt)  # find index for t
                                if it < nt:
                                    WP[n] = WP[n] + (Ezt[-k - 1, it]) * (-1 * self.dz[k])  # compute integral
                            pbar.update(1)

                    else:
                        for n in range(len(s)):
                            for k in range(nz):
                                ts = (
                                    (z[k] + s[n]) / self.v
                                    - zmin / self.v
                                    - self.t[0]
                                    + ti
                                )
                                it = int(ts / dt)  # find index for t
                                if it < nt:
                                    WP[n] = WP[n] + (Ezt[k, it]) * self.dz[k]  # compute integral

                            pbar.update(1)

                    WP = WP / (self.q * 1e12)  # [V/pC]
                    WP_3d[i0 + i, j0 + j, :] = WP

        self.s = s
        self.WP = WP_3d[i0, j0, :]
        self.WP_3d = WP_3d

        self.log(f"Elapsed time {pbar.format_dict['elapsed']} s")

        if self.save:
            np.savetxt(
                self.folder + "WP.txt",
                np.c_[self.s, self.WP],
                header="   s [m]" + " " * 20 + "WP [V/pC]" + "\n" + "-" * 48,
            )

    def calc_trans_WP(self, **kwargs):
        """
        Obtains the transverse wake potential from the longitudinal
        wake potential in 3d using the Panofsky-Wenzel theorem using a
        second-order scheme for the gradient calculation

        Parameters
        ----------
        WP_3d : ndarray
            Longitudinal wake potential in 3d WP(x,y,s). Shape = (2*n+1, 2*n+1, len(s))
            where n = n_transverse_cells and s the wakelength array
        s : ndarray
            Wakelegth vector s=c*t-z representing the distance between
            the source and the integration point. Goes from -8.53*sigmat to WL
            where sigmat = sigmaz/c and WL is the Wakelength
        dx : float
            Ez field mesh step in transverse plane, x-dir [m]
        dy : float
            Ez field mesh step in transverse plane, y-dir [m]
        x : ndarray, optional
            vector containing x-coordinates [m]
        y : ndarray, optional
            vector containing y-coordinates [m]
        n_transverse_cells : int, default 1
            Number of transverse cells used for the 3d calculation: 2*n+1
            This determines de size of the 3d wake potential
        """

        for key, val in kwargs.items():
            setattr(self, key, val)

        self.log("\n")
        self.log("Transverse wake potential")
        self.log("-" * 24)
        self.log(f"* No. transverse cells = {self.n_transverse_cells}")

        ds = self.s[2]-self.s[1]
        i0, j0 = self.n_transverse_cells, self.n_transverse_cells

        # Initialize variables
        WPx = np.zeros_like(self.s)
        WPy = np.zeros_like(self.s)
        int_WP = np.zeros_like(self.WP_3d)

        print("Calculating transverse wake potential WPx, WPy...")
        # Obtain the transverse wake potential
        with tqdm(total=len(self.s) * (i0 * 2 + 1) * (j0 * 2 + 1)) as pbar:
            for n in range(len(self.s)):
                for i in range(-i0, i0 + 1, 1):
                    for j in range(-j0, j0 + 1, 1):
                        # Perform the integral
                        int_WP[i0 + i, j0 + j, n] = (
                            np.sum(self.WP_3d[i0 + i, j0 + j, 0:n]) * ds
                        )
                        pbar.update(1)

                # Perform the gradient (second order scheme)
                WPx[n] = - (int_WP[i0+1,j0,n]-int_WP[i0-1,j0,n])/(self.dx[i0-1]+self.dx[i0])
                WPy[n] = - (int_WP[i0,j0+1,n]-int_WP[i0,j0-1,n])/(self.dy[j0-1]+self.dy[j0])
    
        self.WPx = WPx
        self.WPy = WPy

        self.log(f"Elapsed time {pbar.format_dict['elapsed']} s")

        if self.save:
            np.savetxt(
                self.folder + "WPx.txt",
                np.c_[self.s, self.WPx],
                header="   s [m]" + " " * 20 + "WP [V/pC]" + "\n" + "-" * 48,
            )
            np.savetxt(
                self.folder + "WPy.txt",
                np.c_[self.s, self.WPx],
                header="   s [m]" + " " * 20 + "WP [V/pC]" + "\n" + "-" * 48,
            )

    def calc_long_Z(self, samples=1001, fmax=None, **kwargs):
        """
        Obtains the longitudinal impedance from the longitudinal
        wake potential and the beam charge distribution using a
        single-sided DFT with 1000 samples.
        Parameters can be passed as **kwargs

        Parameters
        ----------
        WP : ndarray
            Longitudinal wake potential WP(s)
        s : ndarray
            Wakelegth vector s=c*t-z representing the distance between
            the source and the integration point. Goes from -8.53*sigmat to WL
            where sigmat = sigmaz/c and WL is the Wakelength
        lambdas : ndarray
            Charge distribution λ(s) interpolated to s axis, normalized by the beam charge
        chargedist : ndarray, optional
            Charge distribution λ(z). Not needed if lambdas is specified
        q : float, optional
            Total beam charge in [C]. Not needed if lambdas is specified
        z : ndarray
            vector containing z-coordinates [m]. Not needed if lambdas is specified
        sigmaz : float
            Beam sigma in the longitudinal direction [m].
            Used to calculate maximum frequency of interest fmax=c/(3*sigmaz)
        """
        self.log("\n")
        self.log("Longitudinal impedance")
        self.log("-" * 24)

        for key, val in kwargs.items():
            setattr(self, key, val)

        print("Calculating longitudinal impedance Z...")
        self.log(f"Single sided DFT with number of samples = {samples}")

        # setup charge distribution in s
        if self.lambdas is None and self.chargedist is not None:
            self.calc_lambdas()
        elif self.lambdas is None and self.chargedist is None:
            self.calc_lambdas_analytic()
            try:
                self.log(
                    "! Using analytic charge distribution λ(s) since no data was provided"
                )
            except Exception:  # ascii encoder error handling
                self.log(
                    "! Using analytic charge distribution since no data was provided"
                )

        # Set up the DFT computation
        ds = np.mean(self.s[1:] - self.s[:-1])
        if fmax is None:
            fmax = self.v / self.sigmaz / 3  # max frequency of interest
        N = int(
            (self.v / ds) // fmax * samples
        )  # to obtain a 1000 sample single-sided DFT

        # Obtain DFTs - is it v or c?
        lambdafft = np.fft.fft(self.lambdas * self.v, n=N)
        WPfft = np.fft.fft(self.WP * 1e12, n=N)
        ffft = np.fft.fftfreq(len(WPfft), ds / self.v)

        # Mask invalid frequencies
        mask = np.logical_and(ffft >= 0, ffft < fmax)
        WPf = WPfft[mask] * ds
        lambdaf = lambdafft[mask] * ds
        self.f = ffft[mask]  # Positive frequencies

        # Compute the impedance
        self.Z = -WPf / lambdaf
        self.lambdaf = lambdaf

        if self.save:
            np.savetxt(
                self.folder + "Z.txt",
                np.c_[self.f, self.Z],
                header="   f [Hz]" + " " * 20 + "Z [Ohm]" + "\n" + "-" * 48,
            )
            np.savetxt(
                self.folder + "spectrum.txt",
                np.c_[self.f, self.lambdaf],
                header="   f [Hz]"
                + " " * 20
                + "Charge distribution spectrum [C/s]"
                + "\n"
                + "-" * 48,
            )

    def calc_trans_Z(self, samples=1001, fmax=None):
        """
        Obtains the transverse impedance from the transverse
        wake potential and the beam charge distribution using a
        single-sided DFT with 1000 samples
        Parameters can be passed as **kwargs
        """
        self.log("\n")
        self.log("Transverse impedance")
        self.log("-" * 24)

        print("Calculating transverse impedance Zx, Zy...")
        self.log(f"Single sided DFT with number of samples = {samples}")

        # Set up the DFT computation
        ds = np.mean(self.s[1:] - self.s[:-1])
        if fmax is None:
            fmax = self.v / self.sigmaz / 3  # max frequency of interest
        N = int(
            (self.v / ds) // fmax * samples
        )  # to obtain a 1000 sample single-sided DFT

        # Obtain DFTs

        # Normalized charge distribution λ(w)
        lambdafft = np.fft.fft(self.lambdas * self.v, n=N)
        ffft = np.fft.fftfreq(len(lambdafft), ds / self.v)
        mask = np.logical_and(ffft >= 0, ffft < fmax)
        lambdaf = lambdafft[mask] * ds

        # Horizontal impedance Zx⊥(w)
        WPxfft = np.fft.fft(self.WPx * 1e12, n=N)
        WPxf = WPxfft[mask] * ds

        self.Zx = 1j * WPxf / lambdaf

        # Vertical impedance Zy⊥(w)
        WPyfft = np.fft.fft(self.WPy * 1e12, n=N)
        WPyf = WPyfft[mask] * ds

        self.Zy = 1j * WPyf / lambdaf

        self.fx = ffft[mask]
        self.fy = ffft[mask]

        if self.save:
            np.savetxt(
                self.folder + "Zx.txt",
                np.c_[self.fx, self.Zx],
                header="   f [Hz]" + " " * 20 + "Zx [Ohm]" + "\n" + "-" * 48,
            )
            np.savetxt(
                self.folder + "Zy.txt",
                np.c_[self.fy, self.Zy],
                header="   f [Hz]" + " " * 20 + "Zy [Ohm]" + "\n" + "-" * 48,
            )

    def calc_lambdas(self, **kwargs):
        """Obtains normalized charge distribution in terms of s
        λ(s) to use in the Impedance calculation

        Parameters
        ----------
        s : ndarray
            Wakelegth vector s=c*t-z representing the distance between
            the source and the integration point. Goes from -8.53*sigmat to WL
            where sigmat = sigmaz/c and WL is the Wakelength
        chargedist : ndarray, optional
            Charge distribution λ(z)
        q : float, optional
            Total beam charge in [C]
        z : ndarray, optional
            vector containing z-coordinates of the domain [m]
        zf : ndarray, optional
            vector containing z-coordinates of the field monitor [m]. N
        """
        for key, val in kwargs.items():
            setattr(self, key, val)

        if type(self.chargedist) is str:
            d = self.read_txt(self.chargedist)
            keys = list(d.keys())
            z = d[keys[0]]
            chargedist = d[keys[1]]

        elif (self.chargedist) is dict:
            keys = list(self.chargedist.keys())
            z = self.chargedist[keys[0]]
            chargedist = self.chargedist[keys[1]]

        else:
            chargedist = self.chargedist
            if len(self.z) == len(self.chargedist):
                z = self.z
            elif len(self.zf) == len(self.chargedist):
                z = self.zf
            else:
                self.log("Dimension error: check input dimensions")

        self.lambdas = np.interp(self.s, z, chargedist / self.q)

        if self.save:
            np.savetxt(
                self.folder + "lambda.txt",
                np.c_[self.s, self.lambdas],
                header="   s [Hz]"
                + " " * 20
                + "Charge distribution [C/m]"
                + "\n"
                + "-" * 48,
            )

    def calc_lambdas_analytic(self, **kwargs):
        """Obtains normalized charge distribution in s λ(z)
        as an analytical gaussian centered in s=0 and std
        equal sigmaz

        Parameters
        ----------
        s : ndarray
            Wakelegth vector s=c*t-z representing the distance between
            the source and the integration point. Goes from -8.53*sigmat to WL
            where sigmat = sigmaz/c and WL is the Wakelength
        sigmaz : float
            Beam sigma in the longitudinal direction [m]
        """

        for key, val in kwargs.items():
            setattr(self, key, val)

        self.lambdas = (
            1
            / (self.sigmaz * np.sqrt(2 * np.pi))
            * np.exp(-(self.s**2) / (2 * self.sigmaz**2))
        )

        if self.save:
            np.savetxt(
                self.folder + "lambda.txt",
                np.c_[self.s, self.lambdas],
                header="   s [Hz]"
                + " " * 20
                + "Charge distribution [C/m]"
                + "\n"
                + "-" * 48,
            )

    def get_SmartBounds(
        self,
        freq_data=None,
        impedance_data=None,
        minimum_peak_height=1.0,
        distance=3,
        inspect_bounds=True,
        Rs_bounds=[0.8, 10],
        Q_bounds=[0.5, 5],
        fres_bounds=[-0.01e9, +0.01e9],
    ):
        import iddefix

        self.log("\nCalculating bounds using the Smart Bound Determination...")
        # Smart bounds
        # Find the main resonators and estimate the bounds -courtesy of Malthe Raschke!
        bounds = iddefix.SmartBoundDetermination(
            freq_data,
            np.real(impedance_data),
            minimum_peak_height=minimum_peak_height,
            Rs_bounds=Rs_bounds,
            Q_bounds=Q_bounds,
            fres_bounds=fres_bounds,
        )

        bounds.find(minimum_peak_height=minimum_peak_height, distance=distance)

        if inspect_bounds:
            bounds.inspect()
            bounds.to_table()
            return bounds

        bounds.to_table()
        return bounds

    def get_DEmodel_fitting(
        self,
        freq_data=None,
        impedance_data=None,
        plane="longitudinal",
        dim="z",
        parameterBounds=None,
        N_resonators=None,
        DE_kernel="DE",
        maxiter=1e5,
        cmaes_sigma=0.01,
        popsize=150,
        tol=1e-3,
        use_minimization=True,
        minimization_margin=[0.3, 0.2, 0.01],
        minimum_peak_height=1.0,
        distance=3,
        inspect_bounds=False,
        Rs_bounds=[0.8, 10],
        Q_bounds=[0.5, 5],
        fres_bounds=[-0.01e9, +0.01e9],
    ):
        import iddefix

        if freq_data is None or impedance_data is None:
            if plane == "longitudinal" and dim == "z":
                freq_data = self.f
                impedance_data = self.Z
            elif plane == "transverse":
                if dim == "x":
                    freq_data = self.fx
                    impedance_data = self.Zx
                elif dim == "y":
                    freq_data = self.fy
                    impedance_data = self.Zy
                else:
                    raise ValueError('Invalid dimension. Use dim = "x" or "y".')
            else:
                raise ValueError(
                    'Invalid plane or dimension. Use plane = "longitudinal" or "transverse" and choose the dimension dim = "z", "x" or "y".'
                )

        if parameterBounds is None or N_resonators is None:
            bounds = self.get_SmartBounds(
                parameterBounds=parameterBounds,
                N_resonators=N_resonators,
                minimum_peak_height=minimum_peak_height,
                distance=distance,
                inspect_bounds=inspect_bounds,
                Rs_bounds=Rs_bounds,
                Q_bounds=Q_bounds,
                fres_bounds=fres_bounds,
            )
            N_resonators = bounds.N_resonators
            parameterBounds = bounds.parameterBounds

        # Build the differential evolution model
        print("Fitting the impedance using Differential Evolution...")
        self.log("\nExtrapolating wake potential using Differential Evolution...")

        objectiveFunction = iddefix.ObjectiveFunctions.sumOfSquaredErrorReal
        DE_model = iddefix.EvolutionaryAlgorithm(
            freq_data,
            np.real(impedance_data),
            N_resonators=N_resonators,
            parameterBounds=parameterBounds,
            plane=plane,
            fitFunction="impedance",
            wake_length=self.wakelength,  # in [m]
            objectiveFunction=objectiveFunction,
        )

        if DE_kernel == "DE":
            DE_model.run_differential_evolution(
                maxiter=int(maxiter),
                popsize=popsize,
                tol=tol,
                mutation=(0.3, 0.8),
                crossover_rate=0.5,
            )

        elif DE_kernel == "CMAES":  # TODO: fix UnboundLocalError
            DE_model.run_cmaes(
                maxiter=int(maxiter),
                popsize=popsize,
                sigma=cmaes_sigma,
            )

        if use_minimization:
            self.log("Running minimization algorithm...")
            DE_model.run_minimization_algorithm(minimization_margin)

        self.DE_model = DE_model
        self.log(DE_model.warning)
        if self.verbose:
            print(DE_model.warning)

        return DE_model

    def get_extrapolated_wake(self, wakelength, sigma=None, use_minimization=True):
        """
        Get the extrapolated wake potential [V/pC] from the DE model
        """
        if self.DE_model is None:
            raise AttributeError("Run get_DEmodel() first to obtain the DE model")

        if sigma is None:
            sigma = self.sigmaz / c_light

        # Get the extrapolated wake potential
        # TODO: add beta
        t = np.arange(
            self.s[0] / c_light, wakelength / c_light, (self.s[2] - self.s[1]) / c_light
        )
        wake_potential = self.DE_model.get_wake_potential(
            t, sigma=sigma, use_minimization=use_minimization
        )

        s = t * c_light  # Convert time to distance [m]
        return s, -wake_potential * 1e-12  # in [V/pC] + CST convention

    def get_extrapolated_wake_function(self, wakelength, use_minimization=True):
        """
        Get the extrapolated wake function (a.k.a. Green function) from the DE model
        """
        if self.DE_model is None:
            raise AttributeError("Run get_DEmodel() first to obtain the DE model")

        t = np.arange(
            self.s[0] / c_light, wakelength / c_light, (self.s[2] - self.s[1]) / c_light
        )
        wake_function = self.DE_model.get_wake(t, use_minimization=use_minimization)
        return t, wake_function

    def get_extrapolated_impedance(
        self, f=None, use_minimization=True, wakelength=None
    ):
        """
        Get the extrapolated impedance [Ohm] from the DE model
        """
        if self.DE_model is None:
            raise AttributeError("Run get_DEmodel() first to obtain the DE model")

        if f is None:
            f = self.DE_model.frequency_data

        impedance = self.DE_model.get_impedance(
            frequency_data=f, use_minimization=use_minimization, wakelength=wakelength
        )
        return f, impedance

    @staticmethod
    def calc_impedance_from_wake(
        wake, s=None, t=None, fmax=None, samples=None, verbose=True
    ):
        if type(wake) is list:
            t = wake[0]
            wake = wake[1]
        if s is not None:
            t = s / c_light
        elif s is None and t is None:
            raise AttributeError(
                'Provide time data through parameter "t" [s] or "s" [m]'
            )
        dt = np.mean(t[1:] - t[:-1])

        # Maximum frequency: fmax = 1/dt
        if fmax is not None:
            aux = np.arange(t.min(), t.max(), 1 / fmax / 2)
            wake = np.interp(aux, t, wake)
            dt = np.mean(aux[1:] - aux[:-1])
            del aux
        else:
            fmax = 1 / dt
        # Time resolution: fres=(1/len(wake)/dt/2)

        # Obtain DFTs
        if samples is not None:
            Wfft = np.fft.fft(wake, n=2 * samples)
        else:
            Wfft = np.fft.fft(wake)

        ffft = np.fft.fftfreq(len(Wfft), dt)

        # Mask invalid frequencies
        mask = np.logical_and(ffft >= 0, ffft < fmax)
        Z = Wfft[mask] / len(wake) * 2
        f = ffft[mask]  # Positive frequencies

        if verbose:
            print(f"* Number of samples = {len(f)}")
            print(f"* Maximum frequency = {f.max()} Hz")
            print(f"* Maximum resolution = {np.mean(f[1:] - f[:-1])} Hz")

        return [f, Z]

    @staticmethod
    def calc_wake_from_impedance(
        impedance, f=None, tmax=None, samples=None, pad=0, verbose=True
    ):
        if len(impedance) == 2:
            f = impedance[0]
            Z = impedance[1]
        elif f is None:
            raise AttributeError('Provide frequency data through parameter "f"')
        else:
            Z = impedance
        df = np.mean(f[1:] - f[:-1])

        # Maximum time: tmax = 1/(f[2]-f[1])
        if tmax is not None:
            aux = np.arange(f.min(), f.max(), 1 / tmax)
            Z = np.interp(aux, f, Z)
            df = np.mean(aux[1:] - aux[:-1])
            del aux
        else:
            tmax = 1 / df

        # Time resolution: tres=(1/len(Z)/(f[2]-f[1]))
        # pad = int(1/df/tres - len(Z))
        # wake = np.real(np.fft.ifft(np.pad(Z, pad)))
        wake = np.real(-1 * np.fft.fft(Z, n=samples))
        wake = np.roll(wake, -1)
        # Inverse fourier transform of impedance
        t = np.linspace(0, tmax, len(wake))

        if verbose:
            print(f"* Number of samples = {len(t)}")
            print(f"* Maximum time = {t.max()} s")
            print(f"* Maximum resolution = {np.mean(t[1:] - t[:-1])} s")

        return [t, wake]

    def read_Ez(self, filename=None, return_value=False):
        """
        Read the Ez.h5 file containing the Ez field information
        """

        if filename is None:
            filename = self.Ez_file

        hf = h5py.File(filename, "r")
        print(f"Reading h5 file {filename}")
        self.log(
            "Size of the h5 file: "
            + str(round((os.path.getsize(filename) / 10**9), 2))
            + " Gb"
        )

        # Set attributes
        self.Ez_hf = hf
        self.Ez_file = filename
        if 'x' in hf.keys():
            self.xf = np.array(hf['x'])
        if 'y' in hf.keys():
            self.yf = np.array(hf['y'])
        if 'z' in hf.keys():
            self.zf = np.array(hf['z'])
        if 'dx' in hf.keys():
            self.dx = np.array(hf['dx'])
        if 'dy' in hf.keys():
            self.dy = np.array(hf['dy'])
        if 'dz' in hf.keys():
            self.dz = np.array(hf['dz'])
        if 't' in hf.keys():
            self.t = np.array(hf['t'])

        if return_value:
            return hf

    def read_txt(self, txt, skiprows=2, delimiter=None, usecols=None):
        """
        Reads txt variables from ascii files and
        returns data in a dictionary. Header should
        be the first line.
        """

        try:
            load = np.loadtxt(
                txt, skiprows=skiprows, delimiter=delimiter, usecols=usecols
            )
        except Exception:
            if self.verbose:
                print(f"[!] Using dtype=np.complex128 to read {txt}")
            load = np.loadtxt(
                txt,
                skiprows=skiprows,
                delimiter=delimiter,
                usecols=usecols,
                dtype=np.complex128,
            )

        try:  # keys == header names
            with open(txt) as f:
                header = f.readline()

            header = header.replace(" ", "")
            header = header.replace("#", "")
            header = header.replace("\n", "")
            header = header.split("]")

            d = {}
            for i in range(len(load[0, :])):
                d[header[i] + "]"] = load[:, i]

        except Exception:  # keys == int 0, 1, ...
            print("[!] Using integer keys since no header was found")
            d = {}
            for i in range(len(load[0, :])):
                d[i] = load[:, i]

        return d

    def save_txt(
        self, f_name, x_data=None, y_data=None, x_name="X [-]", y_name="Y [-]"
    ):
        """
        Saves x and y data to a text file in a two-column format.

        This function exports the provided `x_data` and `y_data` to a `.txt` file,
        formatting the output with a header that includes custom column names.

        Parameters
        ----------
        f_name : str
            Name of the output file (without the `.txt` extension).
        x_data : numpy.ndarray, optional
            Array containing x-axis data. If None, the file is not saved.
        y_data : numpy.ndarray, optional
            Array containing y-axis data. If None, the file is not saved.
        x_name : str, optional
            Label for the x-axis column in the output file. Default is `"X [-]"`.
        y_name : str, optional
            Label for the y-axis column in the output file. Default is `"Y [-]"`.

        Notes
        -----
        - The data is saved in a two-column format where `x_data` and `y_data`
        are combined column-wise.
        - If `x_data` or `y_data` is missing, the function prints a warning and does not save a file.

        Examples
        --------
        Save two NumPy arrays to `data.txt`:

        >>> x = np.linspace(0, 10, 5)
        >>> y = np.sin(x)
        >>> save_txt("data", x, y, x_name="Time [s]", y_name="Amplitude")

        The saved file will look like:

            Time [s]               Amplitude
            --------------------------------
            0.00                   0.00
            2.50                   0.59
            5.00                   -0.99
            7.50                   0.94
            10.00                  -0.54
        """
        if x_data is not None and y_data is not None:
            np.savetxt(
                f_name + ".txt",
                np.c_[x_data, y_data],
                header="   " + x_name + " " * 20 + y_name + "\n" + "-" * 48,
            )
        else:
            print("txt not saved, please provide x_data and y_data")

    def load_results(self, folder=None):
        """Load all txt from a given folder

        The txt files are generated when
        the attribute`save = True` is used
        """
        if folder is None and self.folder is None:
            raise Exception("[!] Please provide a folder path")
        elif folder is None:
            folder = self.folder

        if not folder.endswith("/"):
            folder = folder + "/"

        _, self.lambdas = self.read_txt(folder + "lambda.txt").values()
        _, self.WPx = self.read_txt(folder + "WPx.txt").values()
        _, self.WPy = self.read_txt(folder + "WPy.txt").values()
        self.s, self.WP = self.read_txt(folder + "WP.txt").values()

        _, self.lambdaf = self.read_txt(folder + "spectrum.txt").values()
        _, self.Zx = self.read_txt(folder + "Zx.txt").values()
        _, self.Zy = self.read_txt(folder + "Zy.txt").values()
        self.f, self.Z = self.read_txt(folder + "Z.txt").values()

        self.f = np.abs(self.f)
        self.wakelength = self.s[-1]
        self.folder = folder
        self.Ez_file = folder + "Ez.h5"

    def copy(self):
        obj = type(self).__new__(self.__class__)
        obj.__dict__.update(self.__dict__)
        return obj

    def log(self, txt):
        if self.verbose:
            print("\x1b[2;37m" + txt + "\x1b[0m")

    def read_cst_3d(self, path=None, folder="3d", filename="Ez.h5", units=1e-3):
        """
        Read CST 3d exports folder and store the
        Ez field information into a matrix Ez(x,y,z)
        for every timestep into a single `.h5` file
        compatible with wakis.

        Parameters
        ----------
        path: str, default None
            Path to the field data
        folder: str, default '3d'
            Folder containing the CST field data .txt files
        filename: str, default 'Ez.h5'
            Name of the h5 file that will be generated
        """

        self.log("Reading 3d CST field exports")
        self.log("-" * 24)

        if path is None:
            path = folder + "/"

        # Rename files with E-02, E-03
        for file in glob.glob(path + "*E-02.txt"):
            file = file.split(path)
            title = file[1].split("_")
            num = title[1].split("E")
            num[0] = float(num[0]) / 100

            ntitle = title[0] + "_" + str(num[0]) + ".txt"
            shutil.copy(path + file[1], path + file[1] + ".old")
            os.rename(path + file[1], path + ntitle)

        for file in glob.glob(path + "*E-03.txt"):
            file = file.split(path)
            title = file[1].split("_")
            num = title[1].split("E")
            num[0] = float(num[0]) / 1000

            ntitle = title[0] + "_" + str(num[0]) + ".txt"
            shutil.copy(path + file[1], path + file[1] + ".old")
            os.rename(path + file[1], path + ntitle)

        for file in glob.glob(path + "*_0.txt"):
            file = file.split(path)
            title = file[1].split("_")
            num = title[1].split(".")
            num[0] = float(num[0])

            ntitle = title[0] + "_" + str(num[0]) + ".txt"
            shutil.copy(path + file[1], path + file[1] + ".old")
            os.rename(path + file[1], path + ntitle)

        # sort
        try:

            def sorter(item):
                num = item.split(path)[1].split("_")[1].split(".txt")[0]
                return float(num)

            fnames = sorted(glob.glob(path + "*.txt"), key=sorter)
        except Exception:
            print("[!] Using default sorting for the files")
            fnames = sorted(glob.glob(path + "*.txt"))

        # Get the number of longitudinal and transverse cells used for Ez
        i = 0
        with open(fnames[0]) as f:
            lines = f.readlines()
            n_rows = len(lines) - 3  # n of rows minus the header
            x1 = lines[3].split()[0]

            while True:
                i += 1
                x2 = lines[i + 3].split()[0]
                if x1 == x2:
                    break

        n_transverse_cells = i
        n_longitudinal_cells = int(n_rows / (n_transverse_cells**2))

        # Create h5 file
        if os.path.exists(path + filename):
            os.remove(path + filename)

        hf = h5py.File(path + filename, "w")

        # Initialize variables
        Ez = np.zeros((n_transverse_cells, n_transverse_cells, n_longitudinal_cells))
        x = np.zeros((n_transverse_cells))
        y = np.zeros((n_transverse_cells))
        z = np.zeros((n_longitudinal_cells))
        t = []

        nsteps, i, j, k = 0, 0, 0, 0
        skip = -4  # number of rows to skip
        rows = skip

        # Start scan
        self.log(f"Scanning files in {path}:")
        for file in tqdm(fnames):
            # self.log.debug('Scanning file '+ file + '...')
            title = file.split(path)
            title2 = title[1].split("_")

            try:
                num = title2[1].split(".txt")
                t.append(float(num[0]) * 1e-9)
            except Exception:
                print("[!] timestep not found, using step number instead")
                t.append(nsteps)

            with open(file) as f:
                for line in f:
                    rows += 1
                    columns = line.split()

                    if rows >= 0 and len(columns) > 1:
                        k = int(rows / n_transverse_cells**2)
                        j = int(rows / n_transverse_cells - n_transverse_cells * k)
                        i = int(
                            rows - j * n_transverse_cells - k * n_transverse_cells**2
                        )

                        if k >= n_longitudinal_cells:
                            k = int(n_longitudinal_cells - 1)

                        Ez[i, j, k] = float(columns[5])

                        x[i] = float(columns[0]) * units
                        y[j] = float(columns[1]) * units
                        z[k] = float(columns[2]) * units

            if nsteps == 0:
                prefix = "0" * 5
                hf.create_dataset("Ez_" + prefix + str(nsteps), data=Ez)
            else:
                prefix = "0" * (5 - int(np.log10(nsteps)))
                hf.create_dataset("Ez_" + prefix + str(nsteps), data=Ez)

            i, j, k = 0, 0, 0
            rows = skip
            nsteps += 1

            # close file
            f.close()

        hf["x"] = x
        hf["y"] = y
        hf["z"] = z
        hf["t"] = t

        hf.close()

        # set field info
        self.log(
            "Ez field is stored in a matrix with shape "
            + str(Ez.shape)
            + " in "
            + str(int(nsteps))
            + " datasets"
        )
        self.log(
            f"Finished scanning files - hdf5 file {filename} succesfully generated"
        )

        # Update self
        self.xf = x
        self.yf = y
        self.zf = z
        self.t = np.array(t)

    def assign_logs(self):
        """
        Assigns the parameters of the wake to the logger
        """
        self.logger.wakeSolver["ti"] = self.ti
        self.logger.wakeSolver["q"] = self.q
        self.logger.wakeSolver["sigmaz"] = self.sigmaz
        self.logger.wakeSolver["beta"] = self.beta
        self.logger.wakeSolver["xsource"] = self.xsource
        self.logger.wakeSolver["ysource"] = self.ysource
        self.logger.wakeSolver["xtest"] = self.xtest
        self.logger.wakeSolver["ytest"] = self.ytest
        self.logger.wakeSolver["chargedist"] = self.chargedist
        self.logger.wakeSolver["skip_cells"] = self.skip_cells
        self.logger.wakeSolver["results_folder"] = self.folder
