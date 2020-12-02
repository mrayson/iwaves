"""
Base class for solving KdV 
"""

import numpy as np
from scipy import sparse 
from scipy import linalg as la


class KdVCore(object):
    
    # KdV parameters
    c = 1.
    alpha = 0.
    beta = 0.

    # Grid/discretisation parameters
    N = 10 
    dx = 50.
    dt = 1.
    t = 0.

    spongedist = 0.
    spongetime = 1e6

    nonhydrostatic = 1.
    nonlinear = 1.

    # Implicit /Explicit time-stepping parameters

    ## MCN - AX2+
    #c_im = 1/8.
    #b_ex = 3/8.

    # AM2 - AX2
    c_im = 1/2.
    b_ex = 1/2.

    ### AI2 - AB3
    #c_im = 3/2.
    #b_ex = 5/6.

    ###
    # These are valid
    # BDF2 - BX2
    #c_im = 0.
    #b_ex = 0.

    ## BDF2 - BX2*
    #c_im = 0.
    #b_ex = 1/2.

    ## BI2 - BX3*
    #c_im = 1/3.
    #b_ex = 2/3.

    def __init__(self, **kwargs):
        """
        Numerical KdV solution
        """
        self.__dict__.update(**kwargs)

        # Set the boundary conditions and the state-vectors
        self.bcs = [0.,0.,0.]

        N = self.N
        self.B_n_m2 = np.zeros((N,))
        self.B_n_m1 = np.zeros((N,))
        self.B = np.zeros((N,))
        self.B_n_p1 = np.zeros((N,))

        # Set the parameters to be vectors
        if isinstance(self.alpha,float):
            self.alpha = self.alpha*np.ones((self.N,))
        else:
            self.alpha = self.alpha

        if isinstance(self.c,float):
            self.c = self.c*np.ones((self.N,))
        else:
            self.c = self.c

        if isinstance(self.beta,float):
            self.beta = self.beta*np.ones((self.N,))
        else:
            self.beta = self.beta

        # This effectively turns on/off dispersion and nonlinear steepening
        self.beta *= self.nonhydrostatic
        self.alpha *= self.nonlinear




        # Construct the RHS linear operator (matrix)
        diags = self.build_linear_diags()
        self.L_rhs = sparse.spdiags(diags, [-3,-2,-1,0,1,2,3], self.N, self.N)

        diags = self.build_dispersion_diags()
        self.L_rhs += sparse.spdiags(diags, [-3,-2,-1,0,1,2,3], self.N, self.N)
        

        # Construct the LHS
        self.L_lhs, diags = self.build_lhs_matrix()

    def solve_step(self, bc_left=0):
        """
        Solve the current step
        """
        status = 0
        self.t += self.dt

        ### Construct the RHS vector
        # Implicit terms
        #cff1 = 0. # Fully implicit
        #cff2 = 0.
        cff1 = 0.5*(1. - 2.*self.c_im)*self.dt
        cff2 = 0.5*self.c_im*self.dt
        RHS = cff1*self.L_rhs.dot(self.B) +\
                cff2*self.L_rhs.dot(self.B_n_m1)

        # Nonlinear (explicit) terms
        cff3 = self.dt*(3 + self.b_ex)*0.5
        cff4 = -self.dt*(1+2*self.b_ex)*0.5
        cff5 = self.dt*(self.b_ex)*0.5
        
        RHS += cff3*self.calc_nonlinear_rhs(self.B)
        RHS += cff4*self.calc_nonlinear_rhs(self.B_n_m1)
        RHS += cff5*self.calc_nonlinear_rhs(self.B_n_m2)

        # Other terms from the time-derivative
        RHS += self.B

        # Add the BCs to the RHS
        cff0 = 0.5*(1 + self.c_im)*self.dt
        RHS[0] += cff0 * self.c[0]/(2*self.dx)*bc_left
        RHS[0] += cff1 * self.c[0]/(2*self.dx)*self.bcs[1]
        RHS[0] += cff2 * self.c[0]/(2*self.dx)*self.bcs[2]

        # Use the direct banded matrix solver (faster)
        self.B_n_p1[:] = la.solve_banded( (3,3), self.L_lhs.data[::-1,:], RHS)

        # Check solutions
        if np.any( np.isnan(self.B_n_p1)):
            return -1

        # Update the terms last
        self.B_n_m2[:] = self.B_n_m1
        self.B_n_m1[:] = self.B
        self.B[:] = self.B_n_p1

        ## Update the boundary terms in these equations
        self.bcs[2] = self.bcs[1]
        self.bcs[1] = self.bcs[0]
        self.bcs[0] = bc_left

        return status

    def build_lhs_matrix(self):
        """
        Build the LHS sparse matrix

        This has the form:
            [1 - 0.5(1+a)*dt*L ] * u_n_p1
        """
        j=3
        diags1 = self.build_linear_diags()
        diags1 += self.build_dispersion_diags()

        # Ones down primary diagonal
        diags2 = np.zeros_like(diags1)
        diags2[j,:] = 1.

        cff = self.dt*(1+self.c_im)*0.5        
        diags =  diags2 - cff*diags1
        
        # Build the sparse matrix
        M = sparse.spdiags(diags, [-3,-2,-1,0,1,2,3], self.N, self.N)

        return M, diags

    def build_linear_diags(self):
        """
        Build the diagonal terms for the linear (implicit) terms 
        """
        N = self.N
        dx = self.dx
        j = 3 # Index of the mid-point

        diags = np.zeros((7, self.N))

        # Advection term
        cff1 = -self.c/(2*dx)

        diags[j-1, :] += -1*cff1
        diags[j+1, :] += 1*cff1

        # Sponge term
        x = np.arange(0,N*dx,dx)
        rdist = x[-1] - x # Distance from right boundary
        spongefac = -np.exp(-6*rdist/self.spongedist)/self.spongetime
        diags[j,:] += spongefac 

        return diags

    def build_dispersion_diags(self):
        """
        Build diagonals for the dispersion term
        """
        N = self.N
        j = 3 # Index of the mid-point
        diags = np.zeros((7, self.N))

        dx3 = np.power(self.dx, 3.)
        cff = -self.beta/(2*dx3)

        #diags[j-2,:] = np.arange(1,N+1)
        #diags[j-1,:] = np.arange(1,N+1)
        #diags[j+1,:] = np.arange(1,N+1)
        #diags[j+2,:] = np.arange(1,N+1)

        #diags[j,0:2] = 11
        #diags[j+1,1:3] = 12
        #diags[j+2,2:4] = 13
        #diags[j+3,3:5]= 14

        diags[j-2,:] += -1*cff
        diags[j-1,:] += 2*cff
        diags[j+1,:] += -2*cff
        diags[j+2,:] += 1*cff

        ## Left boundary - use forward differencing
        #diags[j-1,0] = 0
        #diags[j,0:2] = -2*cff
        #diags[j+1,1:3] = 6*cff
        #diags[j+2,2:4] = -6*cff
        #diags[j+3,3:5] = 2*cff

        # Zero first two points
        diags[j-1,0] = 0
        diags[j,0:2] = 0 
        diags[j+1,1:3] = 0 
        diags[j+2,2:4] = 0 
        diags[j+3,3:5] = 0 

        return diags

    def calc_nonlinear_rhs(self, A):
        """
        Calculate the nonlinear steepening term vectors
        """
        cff = -1/(4*self.dx)   
        N = self.N
        rhs = np.zeros((N,))
        alpha = self.alpha

        # Central difference for the interior points
        rhs[1:N-1] = cff*alpha[1:-1] * (A[2:]*A[2:] - A[0:-2]*A[0:-2])

        # Boundary term forward difference
        rhs[0] = 0.5*cff*alpha[0]*(A[1]*A[1] - A[0]*A[0])

        return rhs








     


