"""
Density fitting and interpolation classes
"""

import numpy as np
from scipy.optimize import leastsq, least_squares, curve_fit
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

def single_tanh_rho(z, rho0, rho1, z1, h1,):

    #return rho0 + rho1/2*(1-np.tanh( (z+z1)/h1))
    return rho0 - rho1*np.tanh((z+z1)/h1)

def double_tanh_rho(z, rho0, rho1, rho2, z1, z2, h1, h2):

    #return rho0 + rho1/2*(1-np.tanh( (z+z1)/h1)) +\
    #    rho2/2*(1-np.tanh( (z+z2)/h2))

    return rho0 - rho1*np.tanh((z+z1)/h1) -\
        rho2*np.tanh((z+z2)/h2)


def fdiff(coeffs, rho, z,density_func):

    if density_func=='double_tanh':
        soln = double_tanh_rho(z, *coeffs)
    elif density_func=='single_tanh':
        soln = single_tanh_rho(z, *coeffs)

    #print coeffs[-4], coeffs[-3], coeffs[-2], coeffs[-1]
    return rho - soln

def fit_rho(rho, z, density_func='single_tanh'):
    """
    Fits an analytical density profile to data

    Uses a robust linear regression

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

    # Use "least_squares" at it allows bounds on fitted parameters to be input
    rhotry = rho # - rho0
    H = np.abs(z).max()

    if density_func=='double_tanh':
        initguess = [rho0, 0.01, 0.01, 1., 2., H/10., H/10.] # double tanh guess
        #bounds = [(0,10.),(0,10.),(0,H),(0,H),(0,H/2),(0,H/2)]
        bounds = [(rho0-5,0.,0.,0.,0.,H/20.,H/20.),(rho0+5,10.,10.,H,H,H/2,H/2)]
    elif density_func=='single_tanh':
        initguess = [rho0, 1e-3, 40., 100.] # single stratification function
        bounds = [(rho0-5,0.,0.,0.),(rho0+5,10.,2*H,2*H)]

    soln =\
        least_squares(fdiff, initguess, args=(rhotry, z, density_func), \
        bounds=bounds,\
        xtol=1e-10,
        ftol=1e-10,
        loss='cauchy', f_scale=0.1, # Robust
        verbose=0,
        )
    f0 = soln['x']

    #soln =  leastsq(fdiff, initguess, args=(rhotry, z), \
    #    full_output=True)
    #f0 = soln[0]

    if density_func=='double_tanh':
        rhofit = double_tanh_rho(z, *f0)# + rho0
    elif density_func=='single_tanh':
        rhofit = single_tanh_rho(z, *f0)
    return rhofit, f0

class FitDensity(object):
    """
    Interpolate by fitting an analytical profile first
    """

    density_func = 'single_tanh'

    def __init__(self, rho, z, **kwargs):
        
        self.__dict__.update(**kwargs)

        self.rho0 = rho.min()
        rhofit, self.f0 = fit_rho(rho, z, density_func=self.density_func)

    def __call__(self, Z):

        f0 = self.f0
        if self.density_func=='double_tanh':
            return double_tanh_rho(Z, *f0)# + self.rho0
        elif self.density_func=='single_tanh':
            return single_tanh_rho(Z, *f0) 

class InterpDensity(object):
    """
    Wrapper class for pchip function
    """
    
    density_func = None
    def __init__(self, rho ,z, **kwargs):
        
        self.__dict__.update(**kwargs)

        self.Fi = PchipInterpolator(z, rho, axis=0, extrapolate=True)
        #self.Fi = CubicSpline(z, rho, axis=0, bc_type='natural')

    def __call__(self, Z):
        
        return self.Fi(Z)

class ChebyFitDensity(object):
    """
    Wrapper class for Chebyshev Polynomial fit
    """
    order=None
    def __init__(self, rho ,z, **kwargs):
        
        self.__dict__.update(**kwargs)
        nz = z.size
        if self.order is None:
            self.order = int(max(3,nz -2))
        self.f0 = coefs = np.polynomial.chebyshev.chebfit(z, rho, self.order)

    def __call__(self, Z):
        
        return np.polynomial.chebyshev.chebval(Z, self.f0)
