from iwaves.utils import isw 
from iwaves.utils.tools import grad_z

import numpy as np
from scipy import linalg, sparse
from scipy.interpolate import interp1d
from scipy.integrate import solve_bvp
from scipy.optimize import least_squares, leastsq

GRAV = 9.81
RHO0=1000.

def bc_zero(t, a_bc, omega_bc):

    return 0


def bc_sine(t, a_bc, omega_bc):

    phase = 2*np.pi*omega_bc*t
    out = a_bc*np.sin(phase)

    return out

def ramp_up_factor(t, ramp_time=12.4*3600, plot=False):
    """
    Exponnential ramp for BC.
    """
    if plot:
        print(ramp_time)

        t_ = np.arange(0, 10*ramp_time)
        rf = 1-np.exp(-6*t_/ramp_time)
        rf = rf**2

        plt.plot(t_, rf)
        plt.plot((ramp_time, ramp_time), (0, 1), ':')
        plt.plot((t, t), (0, 1), '-')
    
    rf = 1-np.exp(-6*t/ramp_time)
    rf = rf**2
    
    if plot:
        print('Ramp factor = {}'.format(rf))
    
    return rf

def rampedsine_bc(t, a_bc, omega_bc=1/(12.4*3600), ramp_time=None, plot=False):
    
    out = bc_sine(t, a_bc, omega_bc)

    if ramp_time is None:
        ramp_time = 1/omega_bc

    out = out*ramp_up_factor(t, plot=plot, ramp_time=ramp_time)

    # print('Returning {}'.format(out))

    return out
