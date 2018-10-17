"""
Internal wave functions
"""

import numpy as np
from scipy import linalg, sparse
from scipy.interpolate import interp1d
from scipy.integrate import solve_bvp
from scipy.optimize import least_squares, leastsq

import pdb

GRAV = 9.81
RHO0 = 1020.

###########
# Wave shape
###########
def gaussian(x, a_0, L_w):
    sigma = L_w/4
    return -a_0 * np.exp( - (x/sigma)**2. )

def sine(x, a_0, L_w, x0=0.):
    
    k = 2*np.pi/L_w
    eta = -a_0/2 - a_0/2 * np.sin(k*x + k*x0 + np.pi/2)
    eta[x>x0+L_w/2] = 0.
    eta[x<x0-L_w/2] = 0.

    return eta

def wave_eta(x, a_0, c_n, L_w, wavefunc=gaussian, **kwargs):
    """
    Initial gaussian wave
    """
    #return -a_0 *c_n* np.exp( - (x/L_w)**2. )
    return wavefunc(x, a_0, L_w, **kwargs)

def wave_init(x, rhoz, dz, d, a_0, L_w, mode=0, wavefunc=gaussian, **kwargs):
    """
    Initialise a wavefield
    """
    
    phi, cn, drho_dz = iwave_modes(rhoz, dz, d)
    
    #drho_dz = np.gradient(rhoz, -dz)
    
    eta = wave_eta(x, a_0, np.real(cn[mode]), L_w, wavefunc=wavefunc, **kwargs)
    
    phi_n = phi[:,mode].squeeze()
    phi_n /= np.abs(phi_n).max()
    phi_n *= np.sign(phi_n[1])
    
    rho_pr = eta*drho_dz[:,np.newaxis]*phi_n[:,np.newaxis]
    
    return rhoz[:,np.newaxis] - rho_pr, phi_n

def wave_init_phi(x, rhoz, drho_dz, phi_n, cn, z, d, a_0, L_w, mode=0):
    """
    Proper way to initialize the wavefield
    """
    
    #phi, dphi, cn = iwave_modes(rhoz, dz, d)
    Z = z[...,np.newaxis]
    #drho_dz = np.gradient(rhoz, -dz)
    
    eta = wave_eta(x, a_0, cn, L_w)
    
    #phi_n = phi[:,mode].squeeze()
    phi = phi_n / np.abs(phi_n).max()
    phi *= np.sign(phi_n.sum())
    
    #rho_pr = eta*drho_dz[:,np.newaxis]*phi[:,np.newaxis]
    eta_pr = eta*phi[:,np.newaxis]
    
    #print z.shape, rhoz.shape
    # Interpolation function
    Frho = interp1d(z, rhoz, axis=0)
    
    eta = z[:,np.newaxis] - eta_pr
    
    eta[eta>0.] = 0.
    eta[eta<-d] = -d
    
    # Find rho by interpolating eta
    return Frho(eta), phi
    #return rhoz[:,np.newaxis] - rho_pr, phi

#####
# Nondimensional parameters (Lamb definitions)
#####
def calc_alpha(phi, c, N2, dz):
    """
    Holloway et al 1999 nonlinearity parameter
    """
    phi_z = np.gradient(phi,-np.abs(dz))
    num = 3*c*np.trapz( phi_z**3., dx=dz)
    den = 2*np.trapz( phi_z**2., dx=dz)

    return num/den

def calc_r20(phi, c, N2, dz):
    phi_z = np.gradient(phi,-np.abs(dz))
    S_20 = calc_S20(phi, c, N2, dz)

    num = c*np.trapz( phi*S_20, dx=dz)
    den = 2*np.trapz( phi_z**2., dx=dz)
    return num/den

def calc_alpha_wshear(phi, c, U, dz):
    """
    alpha with shear (see Stastna and Lamb 2002)
    """
    # Stastna and Lamb defn
    E = c/(c-U) * phi
    E_z = np.gradient(E, -np.abs(dz))
    
    num = 3*np.trapz((c-U)**2. * E_z**3., dx = np.abs(dz))
    den = 2*np.trapz((c-U) * E_z**2., dx = np.abs(dz)) 

    return num/den

def calc_alpha_wshear_liu(phi, c, U, dz):
    """
    alpha with shear (see Liu et al 1988)
    """
    # Liu et al definition
    phi_z = np.gradient(phi, -np.abs(dz))
    Uz = np.gradient(U, -np.abs(dz))

    num = 3*c*np.trapz( 1/(c-U) * (phi_z - Uz/(U-c)*phi)**3., dx = np.abs(dz))
    den = 2*np.trapz( 1/(c-U) * (phi_z - Uz/(U-c)*phi)**2., dx = np.abs(dz))

    return num/den

def calc_alpha_wshear_old(phi, c, U, dz):
    """
    alpha with shear (see Grimshaw 2004)
    """
    phi_z = np.gradient(phi,-np.abs(dz))
    num = 3*np.trapz((c-U)**2. * phi_z**3., dx=np.abs(dz))
    den = 2*np.trapz( (c-U) * phi_z**2., dx=np.abs(dz))

    return num/den

def calc_beta_wshear(phi, c, U, dz):
    phi_z = np.gradient(phi, -np.abs(dz))
    num = np.trapz( (c-U)**2. * phi**2., dx=np.abs(dz))
    den = 2*np.trapz( (c-U) * phi_z**2., dx=np.abs(dz))

    return num/den

def calc_r10(phi, c, N2, dz):
    """
    alpha in most other papers
    """
    phi_z = np.gradient(phi,-np.abs(dz))
    num = -3*np.trapz( phi_z**3., dx=np.abs(dz))
    den = 4*np.trapz( phi_z**2., dx=np.abs(dz))

    #N2_z = np.gradient(N2,-dz)
    #S10 = N2_z/c**3*phi**2
    #num = c*np.trapz(phi*S10, dx=dz)
    #den = 2*np.trapz(phi_z**2., dx=dz)
    return num/den

def calc_r01(phi, c, dz):
    phi_z = np.gradient(phi, -np.abs(dz))
    num = -c*np.trapz( phi**2., dx=np.abs(dz))
    den = 2*np.trapz( phi_z**2., dx=np.abs(dz))
    #num = -c*np.sum( phi**2. * dz)
    #den = 2*np.sum( phi_z**2. * dz)

    return num/den

def calc_phi10(phi, c, N2, dz):

    kmax = np.argwhere(phi==phi.max())[0][0]
    phi10rhs = calc_phi10_rhs(phi, c, N2, dz)
    phi10 = solve_phi_bvp(phi10rhs, N2, c, dz, kmax=kmax)

    ## Normalize
    #alpha = -phi10[kmax]
    #phi10 += alpha*phi

    return phi10

def calc_phi01(phi, c, N2, dz):

    kmax = np.argwhere(phi==phi.max())[0][0]
    phi01rhs = calc_phi01_rhs(phi, c, N2, dz)
    phi01 = solve_phi_bvp(phi01rhs, N2, c, dz, kmax=kmax)

    # Normalize
    #alpha = -phi01[kmax]
    #phi01 += alpha*phi

    return phi01

def calc_phi20(phi, c, N2, dz):

    kmax = np.argwhere(phi==phi.max())[0][0]
    phi20rhs = calc_phi20_rhs(phi, c, N2, dz)
    phi20 = solve_phi_bvp(phi20rhs, N2, c, dz, kmax=kmax)

    ## Normalize
    #alpha = -phi20[kmax]
    #phi20 += alpha*phi

    return phi20

def calc_phi01_rhs(phi, c, N2, dz):
    r_01 = calc_r01(phi, c, dz)        

    RHS = -2*r_01*N2/c**3.*phi - phi

    return RHS

def calc_phi10_rhs(phi, c, N2, dz):
    r_10 = calc_r10(phi, c, N2, dz)

    dN2_dz = np.gradient(N2,-np.abs(dz))
    c3i = 1/c**3.

    RHS = -2*r_10*N2*c3i * phi
    RHS += c3i*dN2_dz*phi**2.

    return RHS

def calc_phi20_rhs(phi, c, N2, dz):
    r_20 = calc_r20(phi, c, N2, dz)
    S_20 = calc_S20(phi, c, N2, dz)

    return -r_20*2*N2/c**3. * phi + S_20

def calc_D01(phi, c, N2, dz):
    """
    Calculates the first order nonlinear term for buoyancy
    """
    
    r_01 = calc_r01(phi, c, dz) 
    
    phi01 = calc_phi01(phi, c, N2, dz)
    
    D01 = N2/c*phi01 + r_01*N2/c**2.*phi 

    return D01#/N2

def calc_D10(phi, c, N2, dz):
    """
    Calculates the first order nonlinear term for buoyancy
    """
    
    dN2_dz = np.gradient(N2,-np.abs(dz))
    r_10 = calc_r10(phi, c, N2, dz) 
    
    phi10 = calc_phi10(phi, c, N2, dz)
    
    D10 = N2/c*phi10 + r_10*N2/c**2.*phi - 1/(2*c**2)*dN2_dz*phi**2.

    return D10 #/N2

def calc_D20(phi, c, N2, dz):
    """
    Calculates the second order nonlinear term for buoyancy
    """
    r_20 = calc_r20(phi, c, N2, dz) 
    
    phi20 = calc_phi20(phi, c, N2, dz)

    T_20 = calc_T20(phi, c, N2, dz)
    
    D20 = N2/c*phi20 + r_20*N2/c**2.*phi + T_20

    return D20
 

def calc_S20(phi, c, N2, dz):
    """
    Calculates the second order term
    """
    
    dN2_dz = np.gradient(N2, -dz)
    d2N2_dz2 = np.gradient(dN2_dz, -dz)

    phi_z = np.gradient(phi, -dz)

    r_10 = calc_r10(phi, c, N2, dz) 
    
    phi10 = calc_phi10(phi, c, N2, dz)
    
    S20 = 2/c**3.*dN2_dz * phi * phi10\
        - 1/(2*c**4) * d2N2_dz2 * phi**3 \
        + 3*r_10/c**4 * dN2_dz * phi**2 \
        - r_10*N2/c**4 * phi * phi_z \
        -8/3. * r_10 * N2/c**3 * phi10 \
        -4*r_10**2*N2/c**4 * phi
        #- r_10*N2/c**4 * phi * phi_z \ # Note sure about this term???

    return S20#/N2

def calc_T20(phi, c, N2, dz):
    """
    Calculates the second order term
    """
    
    dN2_dz = np.gradient(N2, -dz)
    d2N2_dz2 = np.gradient(dN2_dz, -dz)

    phi_z = np.gradient(phi, -dz)

    r10 = calc_r10(phi, c, N2, dz) 
    
    phi10 = calc_phi10(phi, c, N2, dz)

    return -dN2_dz/c**2.*phi*phi10\
        -r10*dN2_dz/c**3.*phi**2.\
        +1/6.*d2N2_dz2/c**3.*phi**3.\
        +r10/3.*N2/c**3. * phi * phi_z\
        +4/3.*r10*N2/c**2*phi10\
        +4/3.*r10**2.*N2/c**3.*phi
 

def calc_T10(phi, c, N2, dz):
    """
    Calculates the Grimshaw and co nonlinear correction term
    """
    
    dN2_dz = np.gradient(N2,-np.abs(dz))
    alpha = calc_alpha(phi, c, N2, dz) 
    
    RHS = alpha*N2/c**4. * phi
    RHS += dN2_dz/c**3. *phi**2.


    kmax = np.argwhere(phi==phi.max())[0][0]
    T10 = solve_phi_bvp(RHS, N2, c, dz, kmax=kmax)

    # Normalize
    #alpha = -T10[kmax]
    #T10 += alpha*phi
    
    # normalize ??
    #T10 = T10 * phi/np.mean(phi)
    #T10 = T10 * np.mean(phi)/phi
    
    # Normalize so the max(phi)=1
    #T10 = T10 / np.abs(T10).max()
    #T10 *= np.sign(T10.sum())
    
    #T10 *= (1-phi)
    
    return T10


#####
# Nondimensional parameters
#####
def wave_delta_star(phi, dz):
    """
    Nonlinearity parameter - Liu, 1988
    """
    
    dphi = np.gradient(phi, dz)
    
    num = np.trapz( np.power(dphi,3.), dx=dz)
    den = np.trapz( np.power(dphi,2.), dx=dz)

    return 1.5*num/den
 
def wave_delta(phi, dz, a0):
    """
    Nonlinearity parameter
    """
    dphi = np.gradient(phi, dz)
    
    num = np.trapz( np.power(dphi,3.), dx=dz)
    den = np.trapz( np.power(dphi,2.), dx=dz)

    return -a0 * num/den

def wave_epsilon_star(phi, dz, C0):
    """
    Wave dispersion (nonhydrostasy) parameter - Liu 1988
    """
    
    dphi = np.gradient(phi, dz)
    
    num = np.trapz( phi**2., dx=dz)
    den = np.trapz( dphi**2., dx=dz)
    
    return  C0/2 * num / den


def wave_epsilon(phi, dz, Lw):
    """
    Wave dispersion (nonhydrostasy) parameter
    """
    
    dphi = np.gradient(phi, dz)
    
    num = np.trapz( phi**2., dx=dz)
    den = np.trapz( dphi**2., dx=dz)
    
    return np.sqrt( 3.0/Lw**2 * num / den)

def wave_he(phi, dz):
    """
    Equivalent layer height
    """
    
    dphi = np.gradient(phi, dz)
    
    num = np.trapz( phi**2., dx=dz)
    den = np.trapz( dphi**2., dx=dz)
    
    return np.sqrt(3.0 * num / den)

#####
# Boundary value problem solvers
#####
def solve_phi_bvp_newton(B, N2, c, dz):
    """
    Use the scipy integration function (Uses Newton method)
    """
    nx = N2.shape[0]
    x = np.linspace(0, nx*dz, nx)
    k = -N2/c**2.
    Fk = interp1d(x, k, kind=2)
    Fb = interp1d(x, B, kind=2)
    def fun(x, y, p):
        return np.vstack([y[1], Fk(x) * y[0] + Fb(x)])

    def bc(ya, yb, p):
        # BC: y(0) = y(H) = 0 and y'(0) = 0 
        #return np.array([ ya[0], yb[0], ya[1] ])
        # BC: y(0) = y(H) = 0
        return np.array([ ya[0], yb[0], yb[1] ])


    y = np.zeros((2,x.size))

    res_a = solve_bvp(fun, bc, x, y, p=[0], tol=1e-10)

    return res_a.sol(x)[0,:]


def solve_phi_bvp(B, N2, c, dz, kmax=None):
    """
    Finite-difference  solver for the higher-order vertical structure
    functions

    """
    # Interpolate onto the mid-point
    Bmid = 0.5*(B[1:] + B[:-1])
    N2mid = 0.5*(N2[1:] + N2[:-1])
    nz = Bmid.shape[0] 
    dz2 = 1/dz**2
    
    ### Construct the LHS matrix, A 
    # (use a dense matrix for now)
    A = np.diag(dz2*np.ones((nz-1)),-1) +\
        np.diag(-2*dz2*np.ones((nz,)) +\
        N2mid/c**2*np.ones((nz,)) ,0) + \
        np.diag(dz2*np.ones((nz-1)),1)

    ## BC's
    A[0,0] = -3*dz2 + N2mid[0]/c**2 # Dirichlet BC
    A[-1,-1] = -3*dz2 + N2mid[-1]/c**2 # Dirichlet BC

    # Apply normalization condition at kmax
    if kmax is not None:
        A[kmax, kmax] = 0.
    
    #soln = linalg.solve(A, Bmid)

    # Solve with optimization routines
    def minfun(x0):
        return A.dot(x0) - Bmid

    soln, covc, info, mesg, status = leastsq(minfun, np.zeros_like(Bmid),\
        xtol=1e-12, full_output=True)

    #if status != 2:
    #    print mesg
    #    # These are bad eggs... zero them as its also a solution??
    #    soln *= 0.

    # Interpolate back onto the grid points
    phi = np.zeros((nz+1,))
    soln_mid = 0.5*(soln[1:] + soln[:-1])
    # Zeros are BCs
    phi[1:-1] = soln_mid

    return phi


def _oldsolve_phi_bvp_fd(B, N2, c, dz):
    """
    Finite-difference  solver for the higher-order vertical structure
    functions

    !!!!DOES NOT WORK PROPERLY!!!!
    """

    nz = B.shape[0] 
    dz2 = 1/dz**2
    
    ### Construct the LHS matrix, A 
    # (use a dense matrix for now)
    A = np.diag(dz2*np.ones((nz-1)),-1) +\
        np.diag(-2*dz2*np.ones((nz,)) +\
        N2/c**2*np.ones((nz,)) ,0) + \
        np.diag(dz2*np.ones((nz-1)),1)

    ## BC's
    #eps = 1e-10
    #A[0,0] = -1
    #A[0,1] = eps
    #A[-1,-1] = -1
    #A[-1,-2] = eps
    
    #B[0] = eps
    #B[-1] = eps

    return linalg.solve(A, B)

    #nz = B.shape[0] - 2
    #dz2 = 1/dz**2

    #### Construct the LHS matrix, A 
    ## (use a dense matrix for now)
    #A = np.diag(dz2*np.ones((nz-1)),-1) +\
    #    np.diag(-2*dz2*np.ones((nz,)) +\
    #    N2[1:-1]/c**2*np.ones((nz,)) ,0) + \
    #    np.diag(dz2*np.ones((nz-1)),1)

    ## BC's
    ##eps = 1e-10
    ##A[0,0] = eps
    ##A[0,1] = eps
    ##A[-1,-1] = eps
    ##A[-1,-2] = eps

    #soln = np.zeros((nz+2,))
    #soln[1:-1] = linalg.solve(A, B[1:-1])
    #return soln
    
    #### Use the direct banded solver
    #A = np.zeros((3,nz))
    #A[0,:] = dz2*np.ones(nz)
    #A[1,:] = -2*dz2*np.ones(nz)
    #A[1,:] += N2[1:-1]/c**2*np.ones(nz)
    #A[2,:] = dz2*np.ones(nz)
    #
    ##A[0,0]=eps
    ##A[1,0]=eps
    ##A[2,-1] = eps
    ##A[1,-1] = eps
    #
    #soln = np.zeros((nz+2,))
    #soln[1:-1] = linalg.solve_banded((1,1), A, B[1:-1])
    #return soln
    


#####
# Eigenvalue solver functions
#####
def iwave_modes_banded(N2, dz, k=None):
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
    A = np.vstack([-1*dz2*np.ones((nz,)),\
        2*dz2*np.ones((nz,)),\
        -1*dz2*np.ones((nz,)),\
    ])

    # BC's
    #A[0,0] = -1.
    #A[0,1] = 0.
    #A[-1,-1] = -1.
    #A[-1,-2] = 0.
    A[1,0] = -1.
    A[2,0] = 0.
    A[1,-1] = -1.
    A[0,-1] = 0.

    # Now convert from a generalized eigenvalue problem to 
    #       A.v = lambda.B.v
    # a standard problem 
    #       A.v = lambda.v
    # By multiply the LHS by inverse of B
    #       (B^-1.A).v = lambda.v
    # B^-1 = 1/N2 since B is diagonal
    A[0,:] /= N2
    A[1,:] /= N2
    A[2,:] /= N2

    w, phi = linalg.eig_banded(A)
    
    pdb.set_trace()

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


def iwave_modes_sparse(N2, dz, k=None, Asparse=None, return_A=False):
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
    w, phi = sparse.linalg.eigsh(Asparse, M=B, Minv=Binv, which='SM', k=k)

    # Solve... (use scipy not numpy)
    #w, phi = linalg.eig(A, b=B)

    c = 1. / np.power(w, 0.5) # since term is ... + N^2/c^2 \phi

    # Sort by the eigenvalues
    #idx = np.argsort(c)[::-1] # descending order

    ## Calculate the actual phase speed
    #cn = np.real( c[idx] )

    if return_A:
        return np.real(phi), np.real(c), Asparse
    else:
        return np.real(phi), np.real(c)


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
    w, phi = linalg.eig(A, b=B)

    c = 1. / np.power(w, 0.5) # since term is ... + N^2/c^2 \phi

    # Sort by the eigenvalues
    idx = np.argsort(c)[::-1] # descending order

    # Calculate the actual phase speed
    cn = np.real( c[idx] )

    return phi[:,idx], cn

def iwave_modes_uneven(N2, z):
    """
    Calculates the eigenvalues and eigenfunctions to the internal wave eigenvalue problem:
    
    $$
    \left[ \frac{d^2}{dz^2} - \frac{1}{c_0} \bar{\rho}_z \right] \phi = 0
    $$
    
    with boundary conditions 
    """

    nz = N2.shape[0] 

    dz = np.zeros((nz,))
    zm = np.zeros((nz,))
    dzm = np.zeros((nz,))

    dz[0:-1] = z[0:-1] - z[1:]
    zm[0:-1] = z[0:-1] - 0.5*dz[0:-1]

    dzm[1:-1] = zm[0:-2] - zm[1:-1]
    dzm[0] = dzm[1]
    dzm[-1] = dzm[-2]

    A = np.zeros((nz,nz))
    for i in range(1,nz-1):
        A[i,i] = 1/ (dz[i-1]*dzm[i]) + 1/(dz[i]*dzm[i])
        A[i,i-1] = -1/(dz[i-1]*dzm[i])
        A[i,i+1] = -1/(dz[i]*dzm[i])

    # BC's
    eps = 1e-10
    A[0,0] = -1.
    A[0,1] = 0.
    A[-1,-1] = -1.
    A[-1,-2] = 0.

    # Construct the RHS matrix i.e. put N^2 along diagonals
    B = np.diag(N2,0)

    # Solve... (use scipy not numpy)
    w, phi = linalg.eig(A, b=B)

    c = 1. / np.power(w, 0.5) # since term is ... + N^2/c^2 \phi

    # Sort by the eigenvalues
    idx = np.argsort(c)[::-1] # descending order

    # Calculate the actual phase speed
    cn = np.real( c[idx] )

    phiall = phi[:,idx]

    # Normalize so the max(phi)=1
    for ii in range(nz):
        phi_1 = phiall[:,ii]
        phi_1 = phi_1 / np.abs(phi_1).max()
        phi_1 *= np.sign(phi_1.sum())
        phiall[:,ii] = phi_1

    return phiall, cn



def iwave_modes_nondim(N2, dz, d, nondim=True):
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
    eps = 1e-10
    A[0,0] = -1.
    A[0,1] = 0.
    A[-1,-1] = -1.
    A[-1,-2] = 0.

    # Construct the RHS matrix i.e. put N^2 along diagonals
    B = np.diag(N2,0)

    # Solve... (use scipy not numpy)
    w, phi = linalg.eig(A, b=B)

    c = 1. / np.power(w, 0.5) # since term is ... + N^2/c^2 \phi

    # Sort by the eigenvalues
    idx = np.argsort(c)[::-1] # descending order

    # Calculate the actual phase speed
    cn = np.real( c[idx] )

    return phi[:,idx], cn



