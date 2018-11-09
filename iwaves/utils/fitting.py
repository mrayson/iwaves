"""
Various wave profile fitting routines live here...
"""
import numpy as np
from scipy.interpolate import PchipInterpolator
import scipy.linalg as la
from scipy.optimize import newton_krylov

from iwaves import IWaveModes
from iwaves.utils.density import FitDensity


GRAV = 9.81
RHO0 = 1024.

def calc_buoyancy_h99(B, phi_1, c1, N2, dN2_dz, r10, T10, nonlinear=True):
    """
    Use the Holloway et al 99 version of the eqn's
    """
    #dN2_dz = np.gradient(N2, -np.abs(dz_s))
    
    # Linear term
    b = B[:,np.newaxis] * phi_1 * N2
    
    # nonlinear terms
    if nonlinear:
        alpha = -2*c1*r10
        b -= alpha/(2*c1)*B[:,np.newaxis]*phi_1*N2
        b -= 0.5*dN2_dz*B[:,np.newaxis]**2. * phi_1**2.
        b += c1*B[:,np.newaxis]**2. *N2 * T10
        
    return b


def fit_bmodes_linear(rho, rhoz, z,  zmin, modes,\
        Nz=100, density_func='single_tanh', full_output=True):
    """
    Compute the linear modal amplitude to the mode numbers in the list

    Inputs:
    ---
        rho - matrix[nz, nt], density data
        rhoz - vector[nz], background density profile from bottom to top (desceding)
        z - vector[nz], depth from bottom to top, negative values (ascending)
        modes - list[nmodes], mode numbers in python index i.e. 0=1

    Returns
    ---
        A_t - matrix[nmodes, nt], modal amplitude
        phi - matrix[nmodes, nz], modal structure functions
        rhofit - matrix[nz, nt], best fit density profile

    """

    nz, nt = rho.shape
    nmodes = len(modes)

    # Compute buoyancy from density and backgroud density
    rhopr = rho.T - rhoz[np.newaxis,...]
    b = GRAV*rhopr/RHO0

    # Compute the modal structures
    L = np.zeros((nz,nmodes))
    phi_n = []

    # Calculate dz
    Z = np.linspace(zmin, 0, Nz)
    dz = np.mean(np.diff(Z))

    for ii, mode in enumerate(modes):
        # Use the mode class to create the profile
        iw = IWaveModes(rhoz, z,\
                density_class=FitDensity, density_func=density_func)
        phi, c1, he, znew = iw(zmin, dz, mode)

        if full_output:
            if ii==0:
                Nz = iw.Z.size
                Lout = np.zeros((Nz, nmodes))
            Lout[:,ii] = phi * iw.N2
            phi_n.append(phi)

        ## Interpolate the modal shape and N2 onto the measured depth locations
        F = PchipInterpolator(iw.Z[::-1], iw.phi[::-1])
        my_phi = F(z)

        F = PchipInterpolator(iw.Z[::-1], iw.N2[::-1])
        my_N2 = F(z)

        L[:,ii] = my_phi*my_N2
        #phi_n.append(my_phi)


    ## Fit Ax=b
    A_t,_,_,_ = la.lstsq(L , b.T)

    # Reconstruct the density field
    bfit_n = L[:,np.newaxis,:]*A_t.T[np.newaxis,...]
    bfit = bfit_n.sum(axis=-1) # sum the modes

    rhoprfit = bfit.T*RHO0/GRAV
    rhofit = rhoprfit + rhoz[np.newaxis,:]

    if full_output:
        bfit_n = Lout[:,np.newaxis,:]*A_t.T[np.newaxis,...]
        bfit = bfit_n.sum(axis=-1) # sum the modes
        rhoprfit = bfit.T*RHO0/GRAV
        rhofit_full = rhoprfit + iw.rhoZ[np.newaxis,:]
        return A_t, np.array(phi_n).T, rhofit, rhofit_full, iw
    else:
        return A_t, np.array(phi_n).T, rhofit

def fit_bmodes_linear_w_iw(rho, z,  zmin, modes, iw, \
        Nz=100, full_output=True):
    """
    Compute the linear modal amplitude to the mode numbers in the list

    Inputs:
    ---
        rho - matrix[nz, nt], density data
        z - vector[nz], depth from bottom to top, negative values (ascending)
        modes - list[nmodes], mode numbers in python index i.e. 0=1

    Returns
    ---
        A_t - matrix[nmodes, nt], modal amplitude
        phi - matrix[nmodes, nz], modal structure functions
        rhofit - matrix[nz, nt], best fit density profile

    """

    nz, nt = rho.shape
    nmodes = len(modes)

    # Calculate dz
    Z = np.linspace(zmin, 0, Nz)
    dz = np.mean(np.diff(Z))

    # Interpolate the background density from the iwave class
    phi, cn, he, znew = iw(zmin, dz, 0) # need to call the class once
    F = PchipInterpolator(iw.Z[::-1], iw.rhoZ[::-1])
    rhoz = F(z)

    # Compute buoyancy from density and backgroud density
    rhopr = rho.T - rhoz[np.newaxis,...]
    b = GRAV*rhopr/RHO0

    # Compute the modal structures
    L = np.zeros((nz,nmodes))
    phi_n = []

    c1 = []
    r10 = []

    for ii, mode in enumerate(modes):
        # Use the mode class to create the profile
        phi, cn, he, znew = iw(zmin, dz, mode)
        rn0, _, _, _ = iw.calc_nonlin_params()

        c1.append(cn)
        r10.append(rn0)

        if full_output:
            if ii==0:
                Nz = iw.Z.size
                Lout = np.zeros((Nz, nmodes))
            Lout[:,ii] = phi * iw.N2
            phi_n.append(phi)

        ## Interpolate the modal shape and N2 onto the measured depth locations
        F = PchipInterpolator(iw.Z[::-1], iw.phi[::-1])
        my_phi = F(z)

        F = PchipInterpolator(iw.Z[::-1], iw.N2[::-1])
        my_N2 = F(z)

        L[:,ii] = my_phi*my_N2
        #phi_n.append(my_phi)


    ## Fit Ax=b
    A_t,_,_,_ = la.lstsq(L , b.T)

    # Reconstruct the density field
    bfit_n = L[:,np.newaxis,:]*A_t.T[np.newaxis,...]
    bfit = bfit_n.sum(axis=-1) # sum the modes

    rhoprfit = bfit.T*RHO0/GRAV
    rhofit = rhoprfit + rhoz[np.newaxis,:]

    if full_output:
        bfit_n = Lout[:,np.newaxis,:]*A_t.T[np.newaxis,...]
        bfit = bfit_n.sum(axis=-1) # sum the modes
        rhoprfit = bfit.T*RHO0/GRAV
        rhofit_full = rhoprfit + iw.rhoZ[np.newaxis,:]
        return A_t, np.array(phi_n).T, rhofit, rhofit_full, iw, r10, c1
    else:
        return A_t, np.array(phi_n).T, rhofit, r10, c1



def fit_bmodes_nonlinear(rho, rhoz, z, mode, dz=2.5, density_func='single_tanh'):
    """
    Compute the nonlinear modal amplitude to the mode numbers in the list
    

    Inputs:
    ---
        rho - matrix[nz, nt], density data
        rhoz - vector[nz], background density profile from bottom to top (desceding)
        z - vector[nz], depth from bottom to top, negative values (ascending)
        modes - list[nmodes], mode numbers in python index i.e. 0=1

    Returns
    ---
        A_t - matrix[nmodes, nt], modal amplitude
        rhofit - matrix[nz, nt], best fit density profile
        iw - internal wave mode class

    """
    nz, nt = rho.shape
    nmodes = len(modes)

    # Compute buoyancy from density and backgroud density
    rhopr = rho.T - rhoz[np.newaxis,...]
    b = GRAV*rhopr/RHO0


    # Use the mode class to create the profile
    iw = IWaveModes(rhoz, z,\
        density_class=FitDensity, density_func=density_func)
    phi, c1, he, znew = iw(z.min(), dz, mode)

    r10, _, _, _ = iw.calc_nonlin_params()
    _, _, _, T10, _, _, _ = iw.calc_nonlin_structure()
    
    # Interpolate the structure etc
    F = PchipInterpolator(iw.Z[::-1], iw.phi[::-1])
    my_phi = F(z)

    F = PchipInterpolator(iw.Z[::-1], iw.N2[::-1])
    my_N2 = F(z)

    # Interpolate the higher order structure function
    F = PchipInterpolator(iw.Z[::-1], T10[::-1])
    my_T10 = F(Z)
    
    # Calculate the gradient of N2 and then interpolate it...
    dN2_dz = np.gradient(iw.N2, -iw.dz)
    F = PchipInterpolator(iw.Z[::-1], dN2_dz[::-1])
    my_dN2 = F(Z)
    
    ## Fit alpha as well
    #alpha = -0.008*0
    #my_T10 = my_T10s + alpha*my_phi
    
    def minfun(x0):
        
        btest = calc_buoyancy_h99(x0, my_phi, iw.c1, my_N2,\
                my_dN2, r10, my_T10, nonlinear=False)
    
        err = np.sum( (b-btest)**2., axis=1)
        #print err.max(), alpha
        return err
    
    # Use the Newton-Krylov solver that works on large nonlinear problems
    A_nl = newton_krylov(minfun, np.zeros((nt,)), f_tol=3.1e-5,\
            method='gmres', rdiff=1e-6, iter=100)
    
    # Reconstruct the nonlinear density field
    bfit_nl = calc_buoyancy_h99(A_nl, my_phi, iw.c1, my_N2,\
                my_dN2, r10, my_T10, nonlinear=False)
    
    rhoprfit = bfit_nl*RHO0/GRAV
    rhofitnl = rhoprfit + rhobar[np.newaxis,:]

    return A_nl, rhofitnl, iw
 

