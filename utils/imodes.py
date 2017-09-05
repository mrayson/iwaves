"""
Internal dynamic modes class
"""
#import numpy as np
#from scipy.interpolate import interp1d

from .isw import *
from .tools import grad_z
from scipy.interpolate import PchipInterpolator

import pdb

class IWaveModes(object):
    """
    Wrapper class for calculation of modes
    """
    def __init__(self, rho, z, salt=None):
        if salt is None:
            self.rho = rho
        else:
            raise Exception, NotImplementError
            # Compute density from the nonlinear EOS

        self.z = z

        self.Fi = PchipInterpolator(z, rho, axis=0, extrapolate=True)
 
    def __call__(self, zmax, dz, mode):
        """
        Compute the mode eigenfunction/value on the new vertical grid
        """

        assert zmax <= self.z.max(), 'zmax must be < %f'%self.z.max()

        # Interpolate the stored density onto the new depths
        Z = np.arange(0,zmax+dz,dz)

        rhoZ = self.Fi(Z)


        # Z needs to be negative so N^2 is positive
	drho_dz = grad_z(rhoZ, -Z,  axis=0)
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
        raise Exception, NotImplementedError




