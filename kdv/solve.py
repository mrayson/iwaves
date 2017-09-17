"""
Wrapper function to solve the KdV equations
"""

from kdvimex import KdVImEx
from kdv import KdV

import numpy as np
import xarray as xray

def solve_kdv(rho, z, runtime,\
        solver='imex', ntout=None, outfile=None,\
        verbose=True, **kwargs):
    """
    function for generating different soliton scenarios
    """
    if ntout is None:
        ntout = runtime

    # Initialize the KdV object
    if solver=='imex':
        mykdv = KdVImEx(rho, z, **kwargs)
    elif solver=='explicit':
        mykdv = KdV(rho, z, **kwargs)
    else:
        raise Exception, 'unknown solver %s'%solver

    # Initialise an output array
    nout = int(runtime//ntout)
    B = np.zeros((nout, mykdv.Nx))
    tout = np.zeros((nout,))

    ## Run the model
    nsteps = int(runtime//mykdv.dt_s)
    nn=0
    for ii in range(nsteps):
        # Log output
        point = nsteps/100.
        if verbose:
            if(ii % (5 * point) == 0):
                 print '%3.1f %% complete...'%(float(ii)/nsteps*100)

        if mykdv.solve_step() != 0:
            print 'Blowing up at step: %d'%ii
            break
        
        # Output data
        if (mykdv.t%ntout) < mykdv.dt_s:
            #print ii,nn, mykdv.t
            B[nn,:] = mykdv.B[:]
            tout[nn] = mykdv.t

            # Calculate the velocity
            # No point outputting this as can be calculated on the fly
            #if full_output:
            #    u,w = mykdv.calc_velocity(nonlinear=True)
            #    density = mykdv.calc_density(nonlinear=True)

            nn+=1

    # Save to netcdf
    ds = mykdv.to_Dataset()

    # Create a dataArray from the stored data
    coords = {'x':mykdv.x, 'time':tout}
    attrs = {'long_name':'Wave amplitude',\
            'units':'m'}
    dims = ('time','x')

    Bda = xray.DataArray(B,
            dims = dims,\
            coords = coords,\
            attrs = attrs,\
        )
    ds.merge( xray.Dataset({'B_t':Bda}), inplace=True )

    #if output_us:
    #    Uda = xray.DataArray(us,
    #            dims = dims,\
    #            coords = coords,\
    #            attrs = attrsU,\
    #        )

    #if output_us:
    #    ds.merge( xray.Dataset({'B_t':Bda,'us':Uda}), inplace=True )
    #else:
    #    # Don't output the surface currents
    #    ds.merge( xray.Dataset({'B_t':Bda}), inplace=True )

    if outfile is not None:
        ds.to_netcdf(outfile)

    return mykdv, Bda


