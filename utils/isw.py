"""
Internal wave functions
"""

import numpy as np
from scipy import linalg
from scipy.interpolate import interp1d

GRAV = 9.81
RHO0 = 1020.

###########
# Stratification functions
###########
def ideal_rho_tanh(z, rho0, drho, dp, L):
    return drho/2 * (1 - np.tanh(dp + dp*z/L ) ) + rho0

def lamb_tanh_rho(z, rho0, dp, z1, h1, H=None):
    # Assumes z is negative down
    if H is None:
        H = z.min()
    zhat = z-H
    return rho0*(1 - dp*(1 + np.tanh( (zhat-z1)/h1) ) )

###########
# Wave shape
###########
def gaussian(x, a_0, L_w):
    sigma = L_w/4
    return -a_0 * np.exp( - (x/sigma)**2. )

def sine(x, a_0, L_w, x0=0.):
    
    k = 2*np.pi/L_w
    eta = -a_0/2 - a_0/2 * np.sin(k*x + k*x0 + np.pi/2)
    eta[x>x0+L_w/2] = 0.
    eta[x<x0-L_w/2] = 0.

    return eta

def wave_eta(x, a_0, c_n, L_w, wavefunc=gaussian, **kwargs):
    """
    Initial gaussian wave
    """
    #return -a_0 *c_n* np.exp( - (x/L_w)**2. )
    return wavefunc(x, a_0, L_w, **kwargs)

def wave_init(x, rhoz, dz, d, a_0, L_w, mode=0, wavefunc=gaussian, **kwargs):
    """
    Initialise a wavefield
    """
    
    phi, cn, drho_dz = iwave_modes(rhoz, dz, d)
    
    #drho_dz = np.gradient(rhoz, -dz)
    
    eta = wave_eta(x, a_0, np.real(cn[mode]), L_w, wavefunc=wavefunc, **kwargs)
    
    phi_n = phi[:,mode].squeeze()
    phi_n /= np.abs(phi_n).max()
    phi_n *= np.sign(phi_n[1])
    
    rho_pr = eta*drho_dz[:,np.newaxis]*phi_n[:,np.newaxis]
    
    return rhoz[:,np.newaxis] - rho_pr, phi_n

def wave_init_phi(x, rhoz, drho_dz, phi_n, cn, z, d, a_0, L_w, mode=0):
    """
    Proper way to initialize the wavefield
    """
    
    #phi, dphi, cn = iwave_modes(rhoz, dz, d)
    Z = z[...,np.newaxis]
    #drho_dz = np.gradient(rhoz, -dz)
    
    eta = wave_eta(x, a_0, cn, L_w)
    
    #phi_n = phi[:,mode].squeeze()
    phi = phi_n / np.abs(phi_n).max()
    phi *= np.sign(phi_n.sum())
    
    #rho_pr = eta*drho_dz[:,np.newaxis]*phi[:,np.newaxis]
    eta_pr = eta*phi[:,np.newaxis]
    
    #print z.shape, rhoz.shape
    # Interpolation function
    Frho = interp1d(z, rhoz, axis=0)
    
    eta = z[:,np.newaxis] - eta_pr
    
    eta[eta>0.] = 0.
    eta[eta<-d] = -d
    
    # Find rho by interpolating eta
    return Frho(eta), phi
    #return rhoz[:,np.newaxis] - rho_pr, phi

def wave_delta_star(phi, dz):
    """
    Nonlinearity parameter - Liu, 1988
    """
    
    dphi = np.gradient(phi, dz)
    
    num = np.trapz( np.power(dphi,3.), dx=dz)
    den = np.trapz( np.power(dphi,2.), dx=dz)

    return 1.5*num/den
 
def wave_delta(phi, dz, a0):
    """
    Nonlinearity parameter
    """
    dphi = np.gradient(phi, dz)
    
    num = np.trapz( np.power(dphi,3.), dx=dz)
    den = np.trapz( np.power(dphi,2.), dx=dz)

    return -a0 * num/den

def wave_epsilon_star(phi, dz, C0):
    """
    Wave dispersion (nonhydrostasy) parameter - Liu 1988
    """
    
    dphi = np.gradient(phi, dz)
    
    num = np.trapz( phi**2., dx=dz)
    den = np.trapz( dphi**2., dx=dz)
    
    return  C0/2 * num / den


def wave_epsilon(phi, dz, Lw):
    """
    Wave dispersion (nonhydrostasy) parameter
    """
    
    dphi = np.gradient(phi, dz)
    
    num = np.trapz( phi**2., dx=dz)
    den = np.trapz( dphi**2., dx=dz)
    
    return np.sqrt( 3.0/Lw**2 * num / den)

def wave_he(phi, dz):
    """
    Equivalent layer height
    """
    
    dphi = np.gradient(phi, dz)
    
    num = np.trapz( phi**2., dx=dz)
    den = np.trapz( dphi**2., dx=dz)
    
    return np.sqrt(3.0 * num / den)

def iwave_modes(N2, dz):
    """
    Calculates the eigenvalues and eigenfunctions to the internal wave eigenvalue problem:
    
    $$
    \left[ \frac{d^2}{dz^2} - \frac{1}{c_0} \bar{\rho}_z \right] \phi = 0
    $$
    
    with boundary conditions 
    """

    nz = N2.shape[0] # Remove the surface values
    dz2 = 1/dz**2

    # Construct the LHS matrix, A
    A = np.diag(-1*dz2*np.ones((nz-1)),-1) + \
        np.diag(2*dz2*np.ones((nz,)),0) + \
        np.diag(-1*dz2*np.ones((nz-1)),1)

    # BC's
    eps = 1e-10
    A[0,0] = -1.
    A[0,1] = 0.
    A[-1,-1] = -1.
    A[-1,-2] = 0.

    # Construct the RHS matrix i.e. put N^2 along diagonals
    B = np.diag(N2,0)

    # Solve... (use scipy not numpy)
    w, phi = linalg.eig(A, b=B)

    c = 1. / np.power(w, 0.5) # since term is ... + N^2/c^2 \phi

    # Sort by the eigenvalues
    idx = np.argsort(c)[::-1] # descending order

    # Calculate the actual phase speed
    cn = np.real( c[idx] )

    return phi[:,idx], cn

      
