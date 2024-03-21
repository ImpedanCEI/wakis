'''
`sources.py`
---

Functions defining a time-dependent source to be installed 
in the electromagnetic simulation

It should be in the form `func(self, t, *args, **kwargs)`

TODO: each source needs to be a class, with a call function in the form of
`source.call(self, t)`
'''

import numpy as np
from scipy.constants import mu_0, c as c_light 

class Beam:
    def __init__(self, xsource=0., ysource=0.,  q=1e-9, sigmaz=None, ti=None):
        '''
        Updates the current J every timestep 
        to introduce a gaussian beam 
        moving in +z direction

        Parameters
        ---
        xsource, ysource: float, default 0.
            Transverse position of the source [m]
        q: float, default 1e-9
            Beam charge [C]
        sigmaz: float, default None
            Beam longitudinal sigma [m]
        ti: float, default 9.55*sigmaz
            Injection time [s]
        '''

        self.xsource, self.ysource = xsource, ysource
        self.sigmaz = sigmaz
        if ti is not None: 
            self.ti = ti
        else:  self.ti = 8.548921333333334*self.sigmaz
        self.is_first_update = True

    def update(self, solver, t):
        if self.is_first_update:
            self.ixs, self.iys = np.abs(solver.x-self.xsource).argmin(), np.abs(solver.y-self.ysource).argmin()
            self.is_first_update = False
        # reference shift
        s0 = solver.z.min() - c_light*self.ti
        s = solver.z - c_light*t
        # gaussian
        profile = 1/np.sqrt(2*np.pi*self.sigmaz**2)*np.exp(-(s-s0)**2/(2*self.sigmaz**2))
        # update 
        solver.J[self.ixs,self.iys,:,'z'] = self.q*c_light*profile/self.dx/self.dy

class PlaneWave:
    def __init__(self,   xs=None, ys=None, nodes=15, f=None, beta=1.0):
        '''
        Updates the fields E and H every timestep 
        to introduce a planewave excitation at the given 
        xs, ys slice, moving in z+ direction

        Parameters
        ---
        xs, ys: slice, default 0:N
            Transverse positions of the source [m]
        nodes: float, default 15
            Number of nodes between z.min and z.max
        f: float, default nodes/T
            Frequency of the plane wave [Hz]. It overrides nodes param.
        beta: float, default 1.
            Relativistic beta
        '''
        # Check inputs and update self
        self.nodes = nodes
        self.beta = beta
        self.xs, self.ys = xs, ys
        self.f = f
        self.is_first_update = True
     
    def update(self, solver, t):
        if self.is_first_update:
            if self.xs is None:
                self.xs = slice(0, solver.Nx)
            if self.ys is None:
                self.ys = slice(0, solver.Ny)
            if self.f is None:
                T = (solver.z.max()-solver.z.min())/c_light
                self.f = self.nodes/T
            self.vp = self.beta*c_light  # wavefront velocity beta*c
            self.w = 2*np.pi*self.f           # ang. frequency  
            self.kz = self.w/c_light     # wave number   
            self.is_first_update = False

        solver.H[self.xs,self.ys,0,'y'] = -1.0 * np.cos(self.w*t) 
        solver.E[self.xs,self.ys,0,'x'] = 1.0 * np.cos(self.w*t) /(self.kz/(mu_0*self.vp)) 

class WavePacket:
    def __init__(self, xs=None, ys=None, 
                 sigmaz=None, sigmaxy=None, 
                 wavelength=None, f=None, beta=1.0):
        '''
        Updates E and H fields every timestep to
        introduce a 2d gaussian wave packetat the 
        given xs, ys slice travelling in z+ direction
        
        Parameters
        ----
        xs, ys: slice, default 0:N
            Transverse positions of the source [m]
        sigmaz: float, default 10*dz
            Longitudinal gaussian sigma [m]
        sigmaxy: float, default 5*dx
            Longitudinal gaussian sigma [m]
        wavelength: float, default 10*dz
            Wave packet wavelength [m] f=c/wavelength
        f: float, default None
            Wave packet frequency [Hz], overrides wavelength
        beta: float, default 1.
            Relativistic beta
        '''

        self.beta = beta
        self.xs, self.ys = xs, ys
        self.f = f
        self.wavelength = wavelength
        self.sigmaxy = sigmaxy 
        self.sigmaz = sigmaz   
        self.is_first_update = True

    def update(self, solver, t):
        if self.is_first_update:
            if self.wavelength is None:
                self.wavelength = self.wavelength*solver.dz
            if self.xs is None:
                self.xs = slice(0, solver.Nx)
            if self.ys is None:
                self.ys = slice(0, solver.Ny)
            if self.sigmaz is None: 
                self.sigmaz = 10*solver.dz
            if self.sigmaxy is None: 
                self.sigmaxy = 5*solver.dx
            if self.f is None:
                self.f = c_light/self.wavelength
            else: self.f = f
            self.w = 2*np.pi*self.f 
        
        # reference shift
        s0 = self.z.min()-6*self.sigmaz
        s = self.z.min()-self.beta*c_light*t
        self.is_first_update = False
    
        # 2d gaussian
        X, Y = np.meshgrid(self.x, self.y)
        gaussxy = np.exp(-(X**2+Y**2)/(2*self.sigmaxy**2))
        gausst = np.exp(-(s-s0)**2/(2*self.sigmaz**2))

        # Update
        self.H[self.xs,self.ys,0,'y'] = -1.0*np.cos(self.w*t)*gaussxy*gausst
        self.E[self.xs,self.ys,0,'x'] = 1.0*mu_0*c_light*np.cos(self.w*t)*gaussxy*gausst