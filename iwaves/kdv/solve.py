"""
Wrapper function to solve the KdV equations
"""

from .kdvdamped import KdVDamp
from .kdvimex import KdVImEx
from .kdv import KdV
from .vkdv import vKdV
import iwaves.utils.boundary_conditions as bcs

import numpy as np
import xarray as xray

def zerobc(t):
    return 0

def solve_kdv(rho, z, runtime,\
        solver='imex',
        x=None,
        h=None,
        mode=0,
        ntout=None, outfile=None, full_output=False,\
        myfunc=None,
        bcfunc=zerobc,
        a_bc_left=0,
        verbose=True, **kwargs):
    """
    function for generating different soliton scenarios

    solver: explicit, imex or vkdv (uses imex)
    """
    if ntout is None:
        ntout = runtime

    # Initialize the KdV object
    if solver=='imex':
        mykdv = KdVImEx(rho, z, x=x,**kwargs)
    elif solver=='explicit':
        mykdv = KdV(rho, z, x=x, **kwargs)
    elif solver=='vkdv':
        mykdv = vKdV(rho, z, h, x, mode, **kwargs)
    elif solver=='damped':
        mykdv = KdVDamp(rho, z, x=x, **kwargs)
    else:
        raise Exception('unknown solver %s'%solver)

    # Initialise an output array
    nout = int(runtime//ntout)
    B = np.zeros((nout, mykdv.Nx))
    tout = np.zeros((nout,))
    density = None
    if full_output:
        density = np.zeros((nout, mykdv.Nx, mykdv.Nz))
    output = []

    ## Run the model
    nsteps = int(runtime//mykdv.dt_s)
    nn=0
    for ii in range(nsteps):
        # Log output
        point = nsteps/100.
        if verbose:
            if(ii % (mykdv.print_freq * point) == 0):
                 print('%3.1f %% complete...'%(float(ii)/nsteps*100))

        if mykdv.solve_step(bc_left=bcs.rampedsine_bc(mykdv.t, a_bc_left)) != 0:
            print('Blowing up at step: %d'%ii)
            break
        
        # Evalute the function
        if myfunc is not None:
            output.append(myfunc(mykdv))

        # Output data
        if (mykdv.t%ntout) < mykdv.dt_s:
            #print ii,nn, mykdv.t
            B[nn,:] = mykdv.B[:]
            tout[nn] = mykdv.t

            # Calculate the velocity
            # No point outputting this as can be calculated on the fly
            if full_output:
            #    u,w = mykdv.calc_velocity(nonlinear=True)
               density[nn, :, :] = mykdv.calc_density(nonlinear=True)

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

    coords = {'x':mykdv.x, 'z': np.arange(0, mykdv.Nz)}
    attrs = {'long_name':'Z',\
            'units':'m'}
    dims = ('x','z')
    Zda = xray.DataArray(mykdv.Z.T,
            dims = dims,\
            coords = coords,\
            attrs = attrs,\
        )

    # ds.merge( xray.Dataset({'B_t':Bda}), inplace=True )
    ds['B_t'] = Bda
    # ds['Z'] = Zd a

    if full_output:
        coords = {'time':tout, 'x': mykdv.x, 'z':np.arange(0, mykdv.Nz)}
        attrs = {'long_name':'Density anomaly',\
                'units':'kgm-3'}
        dims = ('time','x','z')
        Rhoda = xray.DataArray(density,
                dims = dims,\
                coords = coords,\
                attrs = attrs,\
            )

        ds['Rho'] = Rhoda
        print('Double setting density.')
        ds['Rho'].values=density # This shouldn't be necessary but it is. xarray bug?

    

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
        print(outfile)
        ds.to_netcdf(outfile)

    if myfunc is None:
        return mykdv, Bda, density
    else:
        return mykdv, Bda, density, output


