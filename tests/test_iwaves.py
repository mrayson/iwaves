"""
Test the eigenvalue solver
"""

import numpy as np
import matplotlib.pyplot as plt

import iwaves


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
phi, cn, Z = IW(500, 10., mode)

plt.figure()
plt.plot(phi, Z)
plt.text(0.1, 0.1, 'c_%d = %3.2f m/s'%(mode+1, cn), transform=plt.gca().transAxes)
plt.show()
