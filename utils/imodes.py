"""
Internal dynamic modes class
"""
#import numpy as np
#from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

from .isw import *
from .tools import grad_z
from .density import InterpDensity 

import gsw

import pdb

class IWaveModes(object):
    """
    Wrapper class for calculation of modes
    """
    density_func = 'single_tanh'
    order = None

    def __init__(self, rho, z, salt=None, density_class=InterpDensity, **kwargs):

        self.__dict__.update(**kwargs)

        if salt is None:
            self.rho = rho
        else:
            # Compute potential density from the nonlinear EOS
            self.rho = gsw.pot_rho_t_exact(salt, rho, -z, 0.)

        # Check monotonicity of z
        assert np.all(np.diff(z)>0),\
                'input z must be increasing, z=0 at surface and positive up'

        self.z = z

        self.Fi = density_class(self.rho, self.z,\
                density_func=self.density_func, order=self.order)
 
    def __call__(self, zmax, dz, mode):
        """
        Compute the mode eigenfunction/value on the new vertical grid
        """

        dz = float(dz) # Make sure of this (new numpy is unforgiving...)
        mode = int(mode)

        #assert zmax <= self.z.min(), 'zmax must be > %f'%self.z.min()
        assert zmax < 0, 'Maximum depth must be negative (< 0)'
        assert dz > 0, 'dz must be positive (> 0)'

        # Interpolate the stored density onto the new depths
        # Depth array needs to be ordered from the surface down
        Z = np.arange(0, zmax - dz, -dz)

        rhoZ = self.Fi(Z)

        # Z needs to be negative so N^2 is positive
        drho_dz = grad_z(rhoZ, Z,  axis=0)
        N2 = -GRAV*drho_dz/RHO0

        phi, cn = iwave_modes(N2, dz)

        # Extract the mode of interest
        phi_1 = phi[:,mode]
        c1 = cn[mode]
        
        # Normalize so the max(phi)=1
        phi_1 = phi_1 / np.abs(phi_1).max()
        phi_1 *= np.sign(phi_1.sum())

        # Compute the equivalent depth (see Vitousek and Fringer,2011)
        h_e = wave_he(phi_1, dz)

        # Store the structure
        self.mode = mode
        self.N2 = N2
        self.rhoZ = rhoZ
        self.phi = phi_1
        self.c1 = c1
        self.Z = Z
        self.dz = dz

        return phi_1, c1, h_e, Z


    def calc_nonlin_params(self):
        """
        Calculate the nonlinear wave parameters

        Returns
            delta - nonlinear term
            epsilon - nonhydrostasy
            he - equivalent height
        """
        # Compute the equivalent depth (see Vitousek and Fringer,2011)
        r10 = calc_r10(self.phi, self.c1, self.N2, self.dz)
        r01 = calc_r01(self.phi, self.c1, self.dz)
        r20 = calc_r20(self.phi, self.c1, self.N2, self.dz)
        h_e = wave_he(self.phi, self.dz)
        
        return r10, r01, r20, h_e

    def calc_nonlin_structure(self):
        """
        Calculate the nonlinear structure functions
        """
        # Structure function for higher powers of epsilon & mu
        phi01 = calc_phi01(self.phi, self.c1, self.N2, self.dz)
        phi10 = calc_phi10(self.phi, self.c1, self.N2, self.dz)
        phi20 = calc_phi20(self.phi, self.c1, self.N2, self.dz)

        # Holloway 99 nonlinear correction
        T10 = calc_T10(self.phi, self.c1, self.N2, self.dz)

        D01 = calc_D01(self.phi, self.c1, self.N2, self.dz)
        D10 = calc_D10(self.phi, self.c1, self.N2, self.dz)
        D20 = calc_D20(self.phi, self.c1, self.N2, self.dz)

        return phi01, phi10, phi20, T10, D01, D10, D20



    def plot_modes(self):

        Z = self.Z

        plt.subplot(131)
        plt.plot(self.rhoZ, Z)
        plt.plot(self.rho, self.z ,'kd')
        plt.xlabel(r'$\rho(z)$ [kg m$^{-3}$]')

        ax = plt.subplot(132)
        plt.plot(self.N2, Z)
        plt.xlabel('$N^2$ [s$^{-2}$]')
        ax.set_yticklabels([])

        ax = plt.subplot(133)
        plt.plot(self.phi, Z)

        #plt.text(0.1,0.1, \
        #        'c1 = %3.2f [m/s]\nr10 = %1.2e [m$^{-1}$]'%(c1,mykdv0.r10),\
        #        transform=ax.transAxes)
        plt.xlabel('$\phi(z)$')
        ax.set_yticklabels([])









