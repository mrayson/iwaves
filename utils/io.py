##################
# IO routines
##################
import xarray as xray
from iwaves.kdv.kdv import KdV

def from_netcdf(kdvfile):
    """
    Load a KdV object from a pre-computed netcdf file
    """
    
    # Load the data file (hope it's in the right format)
    kdvx = xray.open_dataset(kdvfile)
    
    # These are the attributes that KdV requires
    #attrs = ['L_d','Nx','Lw','a0','Cmax','nu_H','x0']
    attrs = ['Nx',\
                'L_d',\
                'a0',\
                'Lw',\
                'x0',\
                'mode',\
                'Cmax',\
                'nu_H',\
                'dx_s',\
                'dz_s',\
                'dt_s',\
                'c1',\
                'mu',\
                'epsilon',\
                'r01',\
                'r10',\
                't',\
                #'ekdv',
        ]



    kwargs = {}
    for aa in attrs:
        kwargs.update({aa:getattr(kdvx,aa)})

    z = kdvx.z.values
    rhoz = kdvx.rhoz.values

    # Let the class compute everything else...
    kdv = KdV(rhoz, z, **kwargs)

    # Set the amplitude function
    kdv.B[:] = kdvx.B.values
    
    # Return the time-variable amplitude array as well
    if 'B_t' in kdvx.variables.keys():
        B_t  = kdvx['B_t']
    else:
        B_t = None
    
    return kdv, B_t

    





