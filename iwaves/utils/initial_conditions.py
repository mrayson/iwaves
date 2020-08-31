
import numpy as np
from scipy import linalg, sparse
from scipy.interpolate import interp1d
from scipy.integrate import solve_bvp
from scipy.optimize import least_squares, leastsq

#########################
## Initial wave amplitude
def eta_gaussian(x, a_0, L_w):

    sigma = L_w/4
    eta = -a_0 * np.exp( - (x/sigma)**2. )

    return eta

def eta_halfsinepulse(x, a_0, L_w, x0=0.):
    """
    Half pulse of a sinusoid.

    L_w is the width of the pulse.
    x0 is the start of the pulse.

    """

    k = 2*np.pi/L_w
    eta = -a_0/2 - a_0/2 * np.sin(k*(x + x0 - L_w/2) + np.pi/2)
    eta[x>x0+L_w] = 0.
    eta[x<x0] = 0.

    return eta

def eta_fullsine(x, a_0, L_w, x0=0.):
    
    k = 2*np.pi/L_w
    eta =  - a_0 * np.cos(k*x + k*x0 + np.pi/2)
    eta[x>x0+1*L_w/2] = 0.
    #eta[x<x0-4*L_w/2] = 0.
    eta[x<x0-L_w/2] = 0.

    return eta

#########################
## Density functions
def rho_double_tanh_rayson(beta, z):
    """
    Double hyperbolic tangent with Matthew W.M. Rayson's coefficient definitions.
    """

    return beta[0] - beta[1]*(np.tanh((z+beta[2])/beta[3])
        + np.tanh((z+beta[4])/beta[5]))

#########################
## Bathy functions
def depth_tanh(x_shelf, L_shelf, h0, h_shelf, x):
    """
    
    """

    h_norm = (np.tanh((x-x_shelf)/(L_shelf))-1)/-2
        
    h = h_norm*(h0 - h_shelf)+h_shelf

    return h

def depth_tanh2(beta, x):
    """
    Same as depth_tanh but with only 2 arguments
    """
    x_shelf = beta[0]
    L_shelf = beta[1]
    h0      = beta[2]
    h_shelf = beta[3]
    
    """
    Get depth using a hyperbolic tan function.
    """
    
    h = depth_tanh(x_shelf, L_shelf, h0, h_shelf, x)
    
    return h