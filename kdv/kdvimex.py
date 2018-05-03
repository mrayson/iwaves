"""
Implicit-explicit KdV solver

(more stable dissipation term)

Classes:
    KdVImEx : Implicit explicit time stepping
    KdVTheta : Theta method time stepping
"""

import iwaves.utils.isw as iwaves
import kdv
import numpy as np
from scipy import sparse 
from scipy.sparse import linalg
from scipy import linalg as la

import pdb

class KdVImEx(kdv.KdV):

    alpha_1 = 1.
    alpha_0 = -1.
    alpha_m1 = 0.

    # Implicit /Explicit parameters

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


    def __init__(self, rhoz, z, wavefunc=iwaves.sine, **kwargs):
        """
        Numerical KdV solution
        """
        kdv.KdV.__init__(self, rhoz, z, wavefunc=wavefunc, **kwargs)

        # Construct the RHS linear operator (matrix)
        self.L_rhs, diags = self.build_linear_matrix()

        # Construct the LHS
        #L_lhs,diags = self.build_lhs_matrix()
        #self.L_lhs = L_lhs.tocsc() #spsolve requires csc

        self.L_lhs,diags = self.build_lhs_matrix()


    def solve_step(self):
        """
        Solve the current step
        """
        status = 0
        self.t += self.dt_s

        ### Construct the RHS vector

        # Implicit terms
        cff1 = 0.5*(1. - 2.*self.c_im)*self.dt_s
        cff2 = 0.5*self.c_im*self.dt_s
        RHS = cff1*self.L_rhs.dot(self.B) +\
                cff2*self.L_rhs.dot(self.B_n_m1)

        # Explicit terms (nonlinear terms)
        if self.nonlinear:
            M_n = self.build_nonlinear_matrix(self.B)
            M_n_m1 = self.build_nonlinear_matrix(self.B_n_m1)
            M_n_m2 = self.build_nonlinear_matrix(self.B_n_m2)

            cff3 = self.dt_s*(3 + self.b_ex)*0.5
            cff4 = -self.dt_s*(1+2*self.b_ex)*0.5
            cff5 = self.dt_s*(self.b_ex)*0.5
           
            RHS += cff3*M_n.dot(self.B)
            RHS += cff4*M_n_m1.dot(self.B_n_m1)
            RHS += cff5*M_n_m2.dot(self.B_n_m2)

        # Other terms from the time-derivative
        RHS -= self.alpha_0*self.B
        RHS -= self.alpha_m1*self.B_n_m1

        ##
        # Solve for B_n_p1
        # Use the sparse matrix solver
        #self.B_n_p1[:] = linalg.spsolve(self.L_lhs.tocsc(), RHS)

        # Use the direct banded matrix solver (faster)
        self.B_n_p1[:] = la.solve_banded( (2,2), self.L_lhs.data[::-1,:], RHS)

        # Check solutions
        if np.any( np.isnan(self.B_n_p1)):
            return -1

        # Update the terms last
        self.B_n_m2[:] = self.B_n_m1
        self.B_n_m1[:] = self.B
        self.B[:] = self.B_n_p1

        return status


    def build_lhs_matrix(self):
        """
        Build the LHS sparse matrix
        """
        #M,diags = self.build_linear_matrix()
        #diags *= self.dt_s*(1+self.c_im)*0.5
        #diags[2,:] = self.alpha_1 - diags[2,:]

        M,diags1 = self.build_linear_matrix()

        # Ones down primary diagonal
        diags2 = np.zeros_like(diags1)
        diags2[2,:] = 1.

        cff = self.dt_s*(1+self.c_im)*0.5        
        diags =  diags2 - cff*diags1

        # Build the sparse matrix
        M = sparse.spdiags(diags, [-2,-1,0,1,2], self.Nx, self.Nx)

        return M, diags

    def build_nonlinear_diags(self, An):
        """
        Build the nonlinear steepening term
        """
        diags = np.zeros((5,self.Nx))
        
        # Add the nonlinear terms
        cff2 = 2*self.epsilon*self.r10*self.c1
        cff3 = 0.5*cff2/self.dx_s
        cff3 *= 0.5
        if self.nonlinear:
            diags[1,:] = diags[1,:] - cff3*An
            diags[3,:] = diags[3,:] + cff3*An

        # extended KdV
        if self.ekdv:
            cff4 = 3*self.epsilon**2*self.r20*self.c1**2
            cff5 = 0.5*cff4/self.dx_s
            An2 = 0.25*np.power(An, 2.)
            diags[1,:] = diags[1,:] - cff5*An2
            diags[3,:] = diags[3,:] + cff5*An2
            
        # Bottom friction parameterization (Holloway et al, 1997)
        if self.k_chezy > 0:
            cff = -self.k_chezy*self.c1 / self.H**2.
            diags[2,:] += cff * np.abs(An)

        return diags

    def build_nonlinear_matrix(self, An):
        """
        Build the nonlinear steepening term
        """
        diags = self.build_nonlinear_diags(An)
        # Build the sparse matrix
        M = sparse.spdiags(diags, [-2,-1,0,1,2], self.Nx, self.Nx)

        return M

    def build_linear_matrix(self):
        """
        Build the linear matrices
        """

        diags = np.zeros((5,self.Nx))

        # pressure terms
        diags[1,:] -= (-0.5*self.c1/self.dx_s) * np.ones((self.Nx,)) #i-1
        diags[3,:] -= (+0.5*self.c1/self.dx_s) * np.ones((self.Nx,)) #i+1

        # Constants
        cff1 = 1*self.mu*self.r01
        #cff1 = 0
        dx3 = 1./np.power(self.dx_s,3.)
        
        # Dispersion term (2nd order)
        if self.nonhydrostatic:
            diags[0,:] += -0.5*cff1*dx3 * np.ones((self.Nx,))
            diags[1,:] += (+cff1*dx3) * np.ones((self.Nx,))
            diags[3,:] += (-cff1*dx3) * np.ones((self.Nx,))
            diags[4,:] += 0.5*cff1*dx3 * np.ones((self.Nx,))

        # Dispersion term (4th order)
        #diags[0,:] += -1/8.*cff1*dx3 * np.ones((self.Nx,))
        #diags[1,:] += -1*cff1*dx3 * np.ones((self.Nx,))
        #diags[2,:] += 13/8.*cff1*dx3 * np.ones((self.Nx,))
        #diags[4,:] += -13/8.*cff1*dx3 * np.ones((self.Nx,))
        #diags[5,:] += +1*cff1*dx3 * np.ones((self.Nx,))
        #diags[6,:] += +1/8.*cff1*dx3 * np.ones((self.Nx,))

        ## Add Laplacian diffusion operator
        #nu_H = 1e1
        nu_H = self.nu_H
        dx2 = 1./np.power(self.dx_s,2.)
        # 2nd order
        diags[1,:] += 0.5*nu_H*dx2 * np.ones((self.Nx,))
        diags[2,:] -= 1*(nu_H*dx2) * np.ones((self.Nx,))
        diags[3,:] += 0.5*nu_H*dx2* np.ones((self.Nx,))

        ## 4th order
        #c1 = -1/12.
        #c2 = 16/12.
        #c3 = -30/12.
        #c4 = 16/12.
        #c5 = -1/12.
        #diags[0,:] += c1*nu_H*dx2 * np.ones((self.Nx,))
        #diags[1,:] += c2*nu_H*dx2 * np.ones((self.Nx,))
        #diags[2,:] += c3*nu_H*dx2 * np.ones((self.Nx,))
        #diags[3,:] += c4*nu_H*dx2* np.ones((self.Nx,))
        #diags[4,:] += c5*nu_H*dx2 * np.ones((self.Nx,))

        # Build the sparse matrix
        M = sparse.spdiags(diags, [-2,-1,0,1,2], self.Nx, self.Nx)

        return M, diags

class KdVTheta(KdVImEx):

    # Typical SUNTANS value
    theta = 0.55

    # Leapfrog
    beta0 = 1.
    betam1 = 0.
    betam2 = 0.

    def __init__(self, rhoz, z, wavefunc=iwaves.sine, **kwargs):
        """
        Numerical KdV solution using the "theta method"

        Consistent with SUNTANS time stepping
        """
        KdVImEx.__init__(self, rhoz, z, wavefunc=wavefunc, **kwargs)

    def solve_step(self):
        """
        Solve the current step
        """
        status = 0

        ### Construct the RHS vector

        # Implicit terms
        cff2 = (1-self.theta)*self.dt_s
        RHS = cff2*self.L_rhs.dot(self.B)

        # Explicit terms (nonlinear terms)
        if self.nonlinear:
            M_n = self.build_nonlinear_matrix(self.B)
            cff3 = self.beta0*self.dt_s
            RHS += cff3*M_n.dot(self.B)

        # Other terms from the time-derivative
        RHS += self.B

        ##
        # Solve for B_n_p1
        self.B_n_p1[:] = linalg.spsolve(self.L_lhs, RHS)

        # Check solutions
        if np.any( np.isnan(self.B_n_p1)):
            return -1

        # Update the terms last
        self.B_n_m2[:] = self.B_n_m1
        self.B_n_m1[:] = self.B
        self.B[:] = self.B_n_p1

        return status

    def build_lhs_matrix(self):
        """
        Build the LHS sparse matrix
        """
        M,diags1 = self.build_linear_matrix()

        # Ones down primary diagonal
        diags2 = np.zeros_like(diags1)
        diags2[2,:] = 1.

        cff = self.dt_s*self.theta        
        diags =  diags2 - cff*diags1

        # Build the sparse matrix
        M = sparse.spdiags(diags, [-2,-1,0,1,2], self.Nx, self.Nx)

        return M, diags


