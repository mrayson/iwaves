"""
Test T-G equation solver
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.sparse import linalg
from scipy.optimize import least_squares, newton_krylov, fsolve
from scipy.integrate import solve_bvp, odeint
from scipy.interpolate import interp1d

from iwaves.utils.isw import iwave_modes

import pdb

def calc_TG(N2, U, c, z):
    """
    Calculate the LHS to the T-G equation
    """
    # Interpolate onto the mid-point (i = 2 --> N-1)
    N2mid = 0.5*(N2[1:] + N2[:-1])
    N2mid = 0.5*(N2[1:] + N2[:-1])
    nz = Bmid.shape[0] 
    dz2 = 1/dz**2
    
    #### Construct the LHS matrix, A 
    ## (use a dense matrix for now)
    #A = np.diag(dz2*np.ones((nz-1)),-1) +\
    #    np.diag(-2*dz2*np.ones((nz,)) +\
    #    N2mid/c**2*np.ones((nz,)) ,0) + \
    #    np.diag(dz2*np.ones((nz-1)),1)

    ### BC's
    #A[0,0] = -3*dz2 + N2mid[0]/c**2 # Dirichlet BC
    #A[-1,-1] = -3*dz2 + N2mid[-1]/c**2 # Dirichlet BC

    ## Apply normalization condition at kmax
    #if kmax is not None:
    #    A[kmax, kmax] = 0.
    #
    ##soln = linalg.solve(A, Bmid)

    ## Solve with optimization routines
    #def minfun(x0):
    #    return A.dot(x0) - Bmid

    #soln, status = leastsq(minfun, np.zeros_like(Bmid), xtol=1e-12)

    ## Interpolate back onto the grid points
    #phi = np.zeros((nz+1,))
    #soln_mid = 0.5*(soln[1:] + soln[:-1])
    ## Zeros are BCs
    #phi[1:-1] = soln_mid

    #return phi

#####
d = 500
Nz = 250
N = 0.01
S = 0.001

RHO0 = 1024.
GRAV = 9.81

# Create the density initial conditions
z = np.linspace(0, d, Nz)

dz = np.abs(z[1]-z[0])

# Idealized density profoile
# drho, dp, Li, rho0
#rhoz = ideal_rho(z, drho, dp, Li) + sig0 # Summer

N2 = N*N*np.ones_like(z)
U = S*z
Uzz = U*0.

# Start of the TG solver (N2, U, c, z)
def create_tg_lhs(N2, U, c, z):

    # size of the matrix problem
    K = N2.size - 2

    N2mid = N2[1:-1] # i in [2, K-1]

    # Get the grid spacing variables
    dz = z[1:] - z[0:-1] # size = [K-1]

    dz_i = 0.5*(dz[1:] + dz[0:-1]) # \delta z_i, size = [k-1]
    dz_im = dz[0:-1] # delta z_{i-1/2}, size = [k-2]
    dz_ip = dz[1:] # delta z_{i+1/2}, size = [k-2]

    # Calculate the velocity at the i +/- 1/2
    U_i_pm = 0.5*(U[1:] + U[0:-1]) # size [k-1]

    # Construct the discrete operator columns
    a_i = (c - U_i_pm[0:-1])**2./(dz_i*dz_im)

    c_i = (c - U_i_pm[1:])**2./(dz_i*dz_ip)

    b_i = c_i - a_i + N2mid

    # Boundary conditions
    b_i[0] = b_i[0] - a_i[0]
    a_i[0] = 0

    b_i[-1] = b_i[-1] - c_i[-1]
    c_i[0] = 0

    # Create the LHS matrix
    L = sparse.spdiags(np.array([a_i, b_i, c_i]), [-1,0,1], K,K)
    return L.tocsr() 

def tg_eqn(phi, N2, U, c, dz):
    phi[0] = 0
    phi[-1] = 0
    dphi = np.gradient(phi,dz)
    cff = (c - U)**2
    dUzzphi = np.gradient(dphi*cff)

    return dUzzphi + N2*phi


def guess_RHS(x0):
    c = x0[0]
    #phi = x0[1:]
    #L = create_tg_lhs(N2, U, c, z)
    #K = L.shape[0]
    #RHS = np.zeros((K,))
    #phi = linalg.spsolve(L, RHS)
    def solve_tg(phi):
        return tg_eqn(phi,N2, U, c, dz)


    # Use optimization routine instead
    guess = np.zeros((Nz,))
    guess[Nz//2] = 1.
    guess[Nz//4] = 0.5
    guess[3*Nz//4] = 0.5
    phi = newton_krylov(solve_tg, guess )
    #xout = np.zeros((K+1,))
    #xout[1:] = L.dot(phi)
    #xout = L.dot(phi)
    xout = solve_tg(phi)
    return np.abs(xout).max()

#soln = least_squares(guess_RHS, np.array([1.1]), bounds=(1e-6,10.), verbose=1, xtol=1e-11)

def solve_tg(N2, U, Uzz, dz):
    """
    Use the scipy integration function (Uses Newton method)
    """
    nx = N2.shape[0]
    x = np.linspace(0, nx*dz, nx)
    #B = (c - U)**2.
    Fn = interp1d(x, N2, kind=2)
    Fu = interp1d(x, U, kind=2)
    Fuzz = interp1d(x, Uzz, kind=2)
    def fun(x, y, p):
        uc2 = (Fu(x) - p[0])**2
        #A = Fuzz(x)/(Fu(x) - p[0]) - Fn(x)/uc2
        A = -Fn(x)/uc2
        return np.vstack([y[1],  A * y[0] ])

    def bc(ya, yb, p):
        # BC: y(0) = y(H) = 0 and y'(0) = 0 
        #return np.array([ ya[0], yb[0], ya[1] ])
        # BC: y(0) = y(H) = 0
        return np.array([ ya[0], yb[0], yb[1] ])


    y = np.zeros((2,x.size))

    # Find the initial phase speed without shear
    mode = 0
    phi, cn = iwave_modes(N2, dz)

    y[0,:] = phi[mode,:]

    res_a = solve_bvp(fun, bc, x, y, p=[cn[mode]], tol=1e-10)

    return res_a.sol(x)[0,:]


Fn = interp1d(z, N2, kind=2, fill_value='extrapolate')
def odefun(phi, y, c):
    phi1, phi2 = phi
    dphi1 = phi2

    #print y, phi
    dphi2 = -Fn(y)/c**2. * phi1

    return [dphi1, dphi2]

Fu = interp1d(z, U, kind=2, fill_value='extrapolate')
#def Fu(y):
#   return 0
Fuzz = interp1d(z, Uzz, kind=2, fill_value='extrapolate')
def odefun2(phi, y, c):

    phi1, phi2 = phi
    dphi1 = phi2
    uc2 = (c - Fu(y))**2
    #A = Fuzz(x)/(Fu(x) - p[0]) - Fn(x)/uc2
    dphi2 = -Fn(y)/uc2 * phi1

    return [dphi1, dphi2]


# Guess initial conditions
phi_1 = 0.
mode = 0
phin, cn = iwave_modes(N2, dz)

cguess = cn[mode]
#
#soln = odeint(odefun, [phi_1, phi_2], z, args=(1.59,))

def objective(args):
    u2_0, cn = args
    dspan = np.linspace(0, d)
    U = odeint(odefun2, [phi_1, u2_0], dspan, args=(cn,))
    u1 = U[:,0]
    u2 = U[:,1]
    # Ensure the top bc is zero
    return u1[-1]#, u2[-1]

soln = least_squares(objective, [1.05, cguess], xtol=1e-12, \
        bounds=((0,cguess-cguess*0.55), (2., cguess+cguess*0.55)))

phi = odeint(odefun2, [phi_1, soln['x'][0]], z, args=(soln['x'][1],))
idx = np.where(np.abs(phi)==np.abs(phi).max())[0][0]
phi /= phi[idx]

print soln

plt.plot(phi[:,0], z)
plt.plot(phin[:,mode], z)
plt.show()
