# copyright ################################# #
# This file is part of the wakis Package.     #
# Copyright (c) CERN, 2024.                   #
# ########################################### #

'''
The `sources.py` script containts different classes
defining a time-dependent sources to be installed 
in the electromagnetic simulation.

All sources need an update function that will be called
every simulation timestep, e.g.:
    def update(self, t, *args, **kwargs)`
'''

import numpy as np
from scipy.constants import mu_0, c as c_light 

class Beam:
    def __init__(self, xsource=0., ysource=0., beta=1.0, 
                 q=1e-9, sigmaz=None, ti=None):
        '''
        Updates the current J every timestep 
        to introduce a gaussian beam 
        moving in +z direction

        Parameters
        ---
        xsource, ysource: float, default 0.
            Transverse position of the source [m]
        beta: float, default 1.0
            Relativistic beta of the beam [0-1.0]
        q: float, default 1e-9
            Beam charge [C]
        sigmaz: float, default None
            Beam longitudinal sigma [m]
        ti: float, default 9.55*sigmaz
            Injection time [s]
        '''

        self.xsource, self.ysource = xsource, ysource
        self.sigmaz = sigmaz
        self.q = q
        self.beta = beta
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
        solver.J[self.ixs,self.iys,:,'z'] = self.q*c_light*profile/solver.dx/solver.dy

class PlaneWave:
    def __init__(self, xs=None, ys=None, zs=0, nodes=15, f=None, beta=1.0):
        '''
        Updates the fields E and H every timestep 
        to introduce a planewave excitation at the given 
        xs, ys slice, moving in z+ direction

        Parameters
        ---
        xs, ys: slice, default 0:N
            Transverse positions of the source (indexes)
        zs: int or slice, default 0
            Injection position in z 
        nodes: float, default 15
            Number of nodes between z.min and z.max
        f: float, default nodes/T
            Frequency of the plane wave [Hz]. It overrides nodes param.
        beta: float, default 1.
            Relativistic beta

        # TODO: support different directions
        '''
        # Check inputs and update self
        self.nodes = nodes
        self.beta = beta
        self.xs, self.ys = xs, ys
        self.zs = zs
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

        solver.H[self.xs,self.ys,self.zs,'y'] = -1.0 * np.cos(self.w*t) 
        solver.E[self.xs,self.ys,self.zs,'x'] = 1.0 * np.cos(self.w*t) /(self.kz/(mu_0*self.vp)) 

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

class Dipole:
    def __init__(self, field='E', component='z',
                xs=None, ys=None, zs=None, 
                nodes=10, f=None, amplitude=1.0):
        '''
        Updates the fields E and H every timestep 
        to introduce a planewave excitation at the given 
        xs, ys slice, moving in z+ direction

        Parameters
        ---
        field: str, default 'E'
            Field to add to source to. Supports component e.g. 'Ex'
        component: str, default 'z'
            If not specified in field, component of the field to add the source to
        xs, ys, zs: int or slice, default N/2
            Positions of the source (indexes)
        nodes: float, default 15
            Number of nodes between z.min and z.max
        f: float, default nodes/T
            Frequency of the plane wave [Hz]. It overrides nodes param.
        '''
        # Check inputs and update self
        self.nodes = nodes
        self.xs, self.ys, self.zs = xs, ys, zs
        self.f = f
        self.field = field
        self.component = component
        self.amplitude = amplitude

        if len(field) == 2: #support for e.g. field='Ex'
            self.component = field[1]
            self.field = field[0]

        self.is_first_update = True

     
    def update(self, solver, t):
        if self.is_first_update:
            if self.xs is None:
                self.xs = int(solver.Nx/2)
            if self.ys is None:
                self.ys = int(solver.Ny/2)
            if self.zs is None:
                self.zs = int(solver.Nz/2)
            if self.f is None:
                T = (solver.z.max()-solver.z.min())/c_light
                self.f = self.nodes/T
            
            self.w = 2*np.pi*self.f
            self.is_first_update = False

        if self.field == 'E':
            solver.E[self.xs,self.ys,self.zs,self.component] = self.amplitude*np.sin(self.w*t) 
        elif self.field == 'H':
            solver.H[self.xs,self.ys,self.zs,self.component] = self.amplitude*np.sin(self.w*t) 
        elif self.field == 'J':
            solver.J[self.xs,self.ys,self.zs,self.component] = self.amplitude*np.sin(self.w*t) 
        else:
            print(f'Field "{self.field}" not valid, should be "E", "H" or "J"]')

class Pulse:
    def __init__(self, field='E', component='z',
                xs=None, ys=None, zs=None, 
                shape='Harris', L=None, amplitude=1.0,
                delay=0.):
        '''
        Injects an electromagnetic pulse at the given 
        source point (xs, ys, zs), with the selected
        shape, length and amplitude.

        Parameters
        ----
        field: str, default 'E'
            Field to add to source to. Supports component e.g. 'Ex'
        component: str, default 'z'
            If not specified in field, component of the field to add the source to
        xs, ys, zs: int or slice, default N/2
            Positions of the source (indexes)
        shape: str, default 'Harris'
            Profile of the pulse in time: ['Harris', 'Gaussian']
        L: float, default 50*dt
            width of the pulse (~10*sigma)

        Note: injection time for the gaussian pulse t0=5*L to ensure smooth derivative.
        '''
        # Check inputs and update self

        self.xs, self.ys, self.zs = xs, ys, zs
        self.field = field
        self.component = component
        self.amplitude = amplitude
        self.shape = shape
        self.L = L
        self.delay = delay

        if len(field) == 2: #support for e.g. field='Ex'
            self.component = field[1]
            self.field = field[0]

        if shape.lower() == 'harris':
            self.tprofile = self.harris_pulse
        elif shape.lower() == 'gaussian':
            self.tprofile = self.gaussian_pulse
        elif shape.lower() == 'rectangular':
            self.tprofile = self.rectangular_pulse
        else:
            print('** shape does not, match available types: "Harris", "Gaussian", "Rectangular"')

        self.is_first_update = True

    def harris_pulse(self, t):
        t = t*c_light - self.delay
        try:
            if t<self.L: 
                return (10 - 15*np.cos(2*np.pi/self.L*t) + 6*np.cos(4*np.pi/self.L*t) - np.cos(6*np.pi/self.L*t))/32 #L dividing (working)
            else:
                return 0.
        except: #support for time arrays
            return (10 - 15*np.cos(2*np.pi/self.L*t) + 6*np.cos(4*np.pi/self.L*t) - np.cos(6*np.pi/self.L*t))/32 #L dividing (working)

    def gaussian_pulse(self, t):
        t = t*c_light - self.delay
        return np.exp(-(t-5*(self.L/10))**2/(2*(self.L/10)**2))
    
    def rectangular_pulse(self, t):
        t = t*c_light - self.delay
        if t<self.L and t>0.:
            return 1.0
        else:
            return 0.0

    def update(self, solver, t):
        if self.is_first_update:
            if self.xs is None:
                self.xs = int(solver.Nx/2)
            if self.ys is None:
                self.ys = int(solver.Ny/2)
            if self.zs is None:
                self.zs = int(solver.Nz/2)
            if self.L is None:
                self.L = 50*solver.dt

            self.is_first_update = False

        if self.field == 'E':
            solver.E[self.xs,self.ys,self.zs,self.component] = self.amplitude*self.tprofile(t)
        elif self.field == 'H':
            solver.H[self.xs,self.ys,self.zs,self.component] = self.amplitude*self.tprofile(t) 
        elif self.field == 'J':
            solver.J[self.xs,self.ys,self.zs,self.component] = self.amplitude*self.tprofile(t) 
        else:
            print(f'Field "{self.field}" not valid, should be "E", "H" or "J"]')