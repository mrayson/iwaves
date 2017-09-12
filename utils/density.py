"""
Density fitting and interpolation classes
"""

import numpy as np
from scipy.optimize import leastsq
from scipy.interpolate import PchipInterpolator

import pdb

# Idealised models
def sech(z):
    return 2./(np.exp(z) + np.exp(-z))

def ideal_rho_tanh(z, rho0, drho, dp, L):
    #return drho/2 - drho/2*np.tanh(dp + dp*z/L )
    return drho/2 * (1 - np.tanh(dp + dp*z/L ) ) + rho0
    #return drho/2 * (1 - np.tanh(z/L + 1 ) )

def lamb_tanh_rho(z, rho0, dp, z1, h1, H=None):
    # Assumes z is negative down
    if H is None:
        H = z.min()
    zhat = z-H
    return rho0*(1 - dp*(1 + np.tanh( (zhat-z1)/h1) ) )

def double_tanh_rho(z, rho1, rho2, z1, z2, h1, h2):
    H = z.min()
    H = np.abs(H)
    zhat = z - H
    z1 = z1*H
    z2 = z2*H
    h1 = h1*H
    h2 = h2*H

    return rho1/2*(1+np.tanh( (z+z1)/h1)) +\
        rho2/2*(1+np.tanh( (z+z2)/h2))

def fdiff(coeffs, rho, z ):
    #soln = ideal_rho_tanh(z, rho0, coeffs[0], coeffs[1], coeffs[2])
    soln = lamb_tanh_rho(z, coeffs[0], coeffs[1], coeffs[2], coeffs[3])
    return rho - soln

def fit_rho(rho, z):
    """
    Fits an analytical density profile to data

    Inputs:
    ---
        rho: vector of density [Nz]
        z : depth [Nz] w/ negative values i.e. 0 at surface, positive: up

    Returns:
    ---
        rhofit: best fit function at z locations
        f0: tuple with analytical parameters
    """

    rho0 = rho.max()
    #drho = rho.max()-rho.min()
    initguess = [rho0, 1e-3, 40., 100.]
    f0,cov_x,info,mesg,err =\
        leastsq(fdiff, initguess, args=(rho, z), \
        full_output=True)

    rhofit = lamb_tanh_rho(z, f0[0], f0[1], f0[2], f0[3])
    return rhofit, f0

class FitDensity(object):
    """
    Interpolate by fitting an analytical profile first
    """
    def __init__(self, rho, z):

        rhofit, self.f0 = fit_rho(rho, z)

    def __call__(self, Z):

        f0 = self.f0
        return lamb_tanh_rho(Z, f0[0], f0[1], f0[2], f0[3])

class InterpDensity(object):
    """
    Wrapper class for pchip function
    """
    
    def __init__(self, rho ,z):

        self.Fi = PchipInterpolator(z, rho, axis=0, extrapolate=True)

    def __call__(self, Z):
        
        return self.Fi(Z)
