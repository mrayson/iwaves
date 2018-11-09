# Functions for 2D arrays
import numpy as np

import pdb

def grad_z(y, z, axis=0):
    """
    Compute the vertical gradient

    "z" can be an array same size as y, or vector along the first axis of "y"

    Takes the derivative along the dimension specified by axis(=0)
    """
    Nz = z.shape[0]

    # Reshape the y variable
    y = y.swapaxes(0, axis)
    #assert y.shape[0] == Nz

    z = z.swapaxes(0, axis)
    assert z.shape == (Nz,) or z.shape == y.shape

    dy_dz = np.zeros_like(y)
    
    # Second-order accurate for mid-points
    ymid = 0.5*(y[1:,...]+y[0:-1,...])

    zmid = 0.5*(z[1:,...]+z[0:-1,...])

    dzmid  = zmid[1:,...] - zmid[0:-1,...] 
    dzmidi = 1./dzmid

    dy_dz[1:-1, ...] = (ymid[1:,...] - ymid[0:-1,...])*\
            dzmidi[:,...]

    # First-order accurate for top and bottom cells
    dy_dz[0,...] = (y[1,...] - y[0,...])*dzmidi[0,...]
    dy_dz[-1,...] = (y[-1,...] - y[-2,...])*dzmidi[-1,...]

    return dy_dz.swapaxes(axis, 0)


