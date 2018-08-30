"""
Test the eigenvalue solver
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg, interpolate, sparse

import iwaves
import pdb

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
N = 0.01

RHO0 = 1024.
GRAV = 9.81

# Create the density initial conditions
z = np.linspace(0, d, Nz)

dz = np.abs(z[1]-z[0])

# Idealized density profoile
# drho, dp, Li, rho0
#rhoz = ideal_rho(z, drho, dp, Li) + sig0 # Summer

N2 = N*N
drho_dz = -RHO0/GRAV * N2

#N2mld = Nmld*Nmld
#drho_dzmld = -RHO0/GRAV * N2mld

# These are inputs into the eigenvalue solver
rhoz = RHO0-1000. + z*drho_dz

# Initialise the class
IW = iwaves.IWaveModes(rhoz, z)

mode = 0
phi, cn,_,Z= IW(-500, 10., mode)

# Test the unven spaced algorithm
sout = np.zeros(Z.shape[0]-1,)
sout[0] = 1.
for ii in range(1,sout.shape[0]):
    sout[ii] = sout[ii-1]*1.0

sout /= np.sum(sout)
dz = sout*d
znew = np.zeros(Z.shape)
znew[1:] = np.cumsum(-dz)
F = interpolate.interp1d(Z, IW.N2)
N2 = F(znew)

phiall, cnall = iwave_modes_uneven(N2, znew)

plt.figure()
plt.plot(phi, Z,lw=3)
plt.plot(phiall[:,mode],znew)
plt.text(0.1, 0.1, 'c_%d = %3.2f m/s\nc_%d = %3.2f m/s'%(mode+1, cn,mode+1,cnall[mode]), \
        transform=plt.gca().transAxes)
plt.show()
