"""
Density fitting and interpolation classes
"""

import numpy as np
from scipy.optimize import leastsq, least_squares
from scipy.interpolate import PchipInterpolator, CubicSpline

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

def double_tanh_rho(z, rho0, rho1, rho2, z1, z2, h1, h2):

    return rho0 + rho1/2*(1-np.tanh( (z+z1)/h1)) +\
        rho2/2*(1-np.tanh( (z+z2)/h2))

def fdiff(coeffs, rho, z ):
    #soln = lamb_tanh_rho(z, *coeffs)
    soln = double_tanh_rho(z, *coeffs)
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

    rho0 = rho.min()

    #rhotry = rho
    #initguess = [rho0, 1e-3, 40., 100.] # lamb stratification function

    # Use "least_squares" at it allows bounds on fitted parameters to be input
    rhotry = rho # - rho0
    initguess = [rho0, 0.01, 0.01, 1., 2., 10., 10.] # double tanh guess
    H = np.abs(z).max()
    #bounds = [(0,10.),(0,10.),(0,H),(0,H),(0,H/2),(0,H/2)]
    bounds = [(rho0-1,0.,0.,0.,0.,0.,0.),(rho0+1,10.,10.,H,H,H/2,H/2)]
    soln =\
        least_squares(fdiff, initguess, args=(rhotry, z), \
        bounds=bounds,\
        )
    f0 = soln['x']

    #soln =  leastsq(fdiff, initguess, args=(rhotry, z), \
    #    full_output=True)
    #f0 = soln[0]
    

    #rhofit = lamb_tanh_rho(z, *f0)
    rhofit = double_tanh_rho(z, *f0)# + rho0
    return rhofit, f0

class FitDensity(object):
    """
    Interpolate by fitting an analytical profile first
    """
    def __init__(self, rho, z):

        self.rho0 = rho.min()
        rhofit, self.f0 = fit_rho(rho, z)

    def __call__(self, Z):

        f0 = self.f0
        return double_tanh_rho(Z, *f0)# + self.rho0
        #return lamb_tanh_rho(Z, *f0) 

class InterpDensity(object):
    """
    Wrapper class for pchip function
    """
    
    def __init__(self, rho ,z):

        #self.Fi = PchipInterpolator(z, rho, axis=0, extrapolate=True)
        self.Fi = CubicSpline(z, rho, axis=0, bc_type='natural')

    def __call__(self, Z):
        
        return self.Fi(Z)
