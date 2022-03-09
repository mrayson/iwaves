"""
Test the eigenvalue solver
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg, interpolate, sparse

from iwaves.utils import imodes as iwaves
from time import time
import pdb

def iwave_modes(N2, dz, k=None):
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
    A[0,0] = -1.
    A[0,1] = 0.
    A[-1,-1] = -1.
    A[-1,-2] = 0.

    # Construct the RHS matrix i.e. put N^2 along diagonals
    B = np.diag(N2,0)

    # Solve... (use scipy not numpy)
    w, phi = linalg.eig(A, b=B, check_finite=False)
    #w, phi = linalg.eigh(A, b=B, check_finite=False)

    c = 1. / np.power(w, 0.5) # since term is ... + N^2/c^2 \phi

    # Sort by the eigenvalues
    idx = np.argsort(c)[::-1] # descending order

    # Calculate the actual phase speed
    cn = np.real( c[idx] )

    return phi[:,idx], cn



def iwave_modes_sparse(N2, dz, k=None, Asparse=None, return_A=False, v0=None):
    """
    Calculates the eigenvalues and eigenfunctions to the internal wave eigenvalue problem:
    
    $$
    \left[ \frac{d^2}{dz^2} - \frac{1}{c_0} \bar{\rho}_z \right] \phi = 0
    $$
    
    with boundary conditions 
    """

    nz = N2.shape[0] # Remove the surface values
    if k is None:
        k = nz-2

    if Asparse is None:
        dz2 = 1/dz**2

        # Construct the LHS matrix, A
        A = np.vstack([-1*dz2*np.ones((nz,)),\
                2*dz2*np.ones((nz,)),\
                -1*dz2*np.ones((nz,)),\
        ])


        # BC's
        eps = 1e-10
        #A[0,0] = -1.
        #A[0,1] = 0.
        #A[-1,-1] = -1.
        #A[-1,-2] = 0.
        A[1,0] = -1.
        A[2,0] = 0.
        A[1,-1] = -1.
        A[0,-1] = 0.

        Asparse = sparse.spdiags(A,[-1,0,1],nz,nz, format='csc')

    # Construct the RHS matrix i.e. put N^2 along diagonals
    #B = np.diag(N2,0)
    B = sparse.spdiags(N2,[0],nz,nz, format='csc')
    Binv = sparse.spdiags(1/N2,[0],nz,nz, format='csc')
    
    if v0 is not None:
        w0 = 1/v0[0]**2.
    else:
        w0=None
    #w, phi = sparse.linalg.eigsh(Asparse, M=B, Minv=Binv, which='SM', k=k, v0=None)
    w, phi = sparse.linalg.eigsh(Asparse, M=B, sigma=1., k=k)
    #w, phi = sparse.linalg.eigsh(Asparse, M=B, which='LM',  k=k)

    # Solve... (use scipy not numpy)
    #w, phi = linalg.eig(A, b=B)

    c = 1. / np.power(w, 0.5) # since term is ... + N^2/c^2 \phi

    # Sort by the eigenvalues
    idx = np.argsort(c)[::-1] # descending order

    # Calculate the actual phase speed
    cn = np.real( c[idx] )


    if return_A:
        return np.real(phi), np.real(cn), Asparse
    else:
        return np.real(phi), np.real(cn)




def iwave_modes_tri(N2, dz, k=None):
    """
    !!! DOES NOT WORK!!!
    Calculates the eigenvalues and eigenfunctions to the internal wave eigenvalue problem:
    
    $$
    \left[ \frac{d^2}{dz^2} - \frac{1}{c_0} \bar{\rho}_z \right] \phi = 0
    $$
    
    with boundary conditions 
    """

    nz = N2.shape[0] # Remove the surface values
    if k is None:
        k = nz-2

    dz2 = 1/dz**2

    # Construct the LHS matrix, A
    Ao = -1*dz2*np.ones((nz-1,))
    Am =  2*dz2*np.ones((nz,))

    # BC's
    Am[0] = -1.
    Ao[0] = 0.
    Am[-1] = -1.
    Ao[-1] = 0.

    # Now convert from a generalized eigenvalue problem to 
    #       A.v = lambda.B.v
    # a standard problem 
    #       A.v = lambda.v
    # By multiply the LHS by inverse of B
    #       (B^-1.A).v = lambda.v
    # B^-1 = 1/N2 since B is diagonal
    Am /= N2

    w, phi = linalg.eigh_tridiagonal(Am, Ao)
    

    ## Main diagonal
    #dd = 2*dz2*np.ones((nz,))

    #dd /= N2

    #dd[0] = -1
    #dd[-1] = -1

    ## Off diagonal
    #ee = -1*dz2*np.ones((nz-1,))
    #ee /= N2[0:-1]

    #ee[0] = 0
    #ee[-1] = 0


    ## Solve... (use scipy not numpy)
    #w, phi = linalg.eigh_tridiagonal(dd, ee )

    #####

    c = 1. / np.power(w, 0.5) # since term is ... + N^2/c^2 \phi

    # Sort by the eigenvalues
    idx = np.argsort(c)[::-1] # descending order

    ## Calculate the actual phase speed
    cn = np.real( c[idx] )

    idxgood = ~np.isnan(cn)
    phisort = phi[:,idx]

    return np.real(phisort[:,idxgood]), np.real(cn[idxgood])




def iwave_modes_uneven(N2, z, k=None):
    """
    Calculates the eigenvalues and eigenfunctions to the internal wave eigenvalue problem:
    
    $$
    \left[ \frac{d^2}{dz^2} - \frac{1}{c_0} \bar{\rho}_z \right] \phi = 0
    $$
    
    with boundary conditions 
    """

    nz = N2.shape[0] 
    if k is None:
        k = nz-2

    dz = np.zeros((nz,))
    zm = np.zeros((nz,))
    dzm = np.zeros((nz,))

    dz[0:-1] = z[0:-1] - z[1:]
    zm[0:-1] = z[0:-1] - 0.5*dz[0:-1]

    dzm[1:-1] = zm[0:-2] - zm[1:-1]
    dzm[0] = dzm[1]
    dzm[-1] = dzm[-2]

    # Solve as a matrix
    #A = np.zeros((nz,nz))
    #for i in range(1,nz-1):
    #    A[i,i] = 1/ (dz[i-1]*dzm[i]) + 1/(dz[i]*dzm[i])
    #    A[i,i-1] = -1/(dz[i-1]*dzm[i])
    #    A[i,i+1] = -1/(dz[i]*dzm[i])

    # Solve as a banded matrix
    A = np.zeros((nz,3))
    for i in range(1,nz-1):
        A[i,0] = 1/ (dz[i-1]*dzm[i]) + 1/(dz[i]*dzm[i])
        A[i,1] = -1/(dz[i-1]*dzm[i])
        A[i,2] = -1/(dz[i]*dzm[i])



    # BC's
    eps = 1e-10
    #A[0,0] = -1.
    #A[0,1] = 0.
    #A[-1,-1] = -1.
    #A[-1,-2] = 0.
    A[0,0] = -1.
    A[0,2] = 0.
    A[-1,0] = -1.
    A[-1,1] = 0.



    Asparse = sparse.spdiags(A.T,[0,-1,1],nz,nz)

    # Construct the RHS matrix i.e. put N^2 along diagonals
    #B = np.diag(N2,0)
    B = sparse.spdiags(N2,[0],nz,nz)

    # Solve... (use scipy not numpy)
    #w, phi = linalg.eig(A, b=B)
    #w, phi = linalg.eig_banded(A, b=B)
    w, phi = sparse.linalg.eigs(Asparse, M=B, which='SM', k=k)

    ## Solve as a banded matrix
    #A = np.zeros((3,nz))
    #for i in range(1,nz-1):
    #    A[1,i] = 1/ (dz[i-1]*dzm[i]) + 1/(dz[i]*dzm[i])
    #    A[0,i-1] = -1/(dz[i-1]*dzm[i])
    #    A[2,i+1] = -1/(dz[i]*dzm[i])

    ### BC's
    ##eps = 1e-10
    ##A[0,0] = -1.
    ##A[0,1] = 0.
    ##A[-1,-1] = -1.
    ##A[-1,-2] = 0.
    #A[1,0] = -1
    #A[1,-1] = -1



    c = 1. / np.power(w, 0.5) # since term is ... + N^2/c^2 \phi

    # Sort by the eigenvalues
    idx = np.argsort(c)[::-1] # descending order

    # Calculate the actual phase speed
    cn = np.real( c[idx] )

    phiall = phi[:,idx]

    # Normalize so the max(phi)=1
    for ii in range(k):
        phi_1 = phiall[:,ii]
        phi_1 = phi_1 / np.abs(phi_1).max()
        phi_1 *= np.sign(phi_1.sum())
        phiall[:,ii] = phi_1

    return phiall, cn

d = 500
Nz = 50
N0 = 0.01

RHO0 = 1024.
GRAV = 9.81

# Create the density initial conditions
z = np.linspace(0, d, Nz)

dz = np.abs(z[1]-z[0])

# Idealized density profoile
# drho, dp, Li, rho0
#rhoz = ideal_rho(z, drho, dp, Li) + sig0 # Summer

N = N0+0.000001*z
N2 = N*N
drho_dz = -RHO0/GRAV * N2

#N2mld = Nmld*Nmld
#drho_dzmld = -RHO0/GRAV * N2mld

# These are inputs into the eigenvalue solver
rhoz = RHO0-1000. + z*drho_dz

# Initialise the class
#IW = iwaves.IWaveModes(rhoz, -z[::-1])
#
mode = 0
#phi, cn,_,Z= IW(-500, 10., mode)
tic = time()
for ii in range(500):
    phi, cn = iwave_modes(N2, dz)
print('Elapsed time dense method = {}'.format(time()-tic))



## Test the unven spaced algorithm
#sout = np.zeros(Z.shape[0]-1,)
#sout[0] = 1.
#for ii in range(1,sout.shape[0]):
#    sout[ii] = sout[ii-1]*1.0
#
#sout /= np.sum(sout)
#dz = sout*d
#znew = np.zeros(Z.shape)
#znew[1:] = np.cumsum(-dz)
#F = interpolate.interp1d(Z, IW.N2)
#N2new = F(znew)
#
#phiall, cnall = iwave_modes_uneven(N2new, znew)

tic = time()
cnall = None
for ii in range(500):
    phiall, cnall = iwave_modes_sparse(N2, dz, v0=cnall,k=6)
print('Elapsed time sparse method = {}'.format(time()-tic))

plt.figure()
plt.plot(phi[:,mode], -z,lw=3)
plt.plot(phiall[:,mode],-z, color='r')
plt.text(0.1, 0.1, 'c_%d = %3.2f m/s\nc_%d = %3.2f m/s'%(mode+1, cn[mode],mode+1,cnall[mode]), \
        transform=plt.gca().transAxes)
plt.show()
