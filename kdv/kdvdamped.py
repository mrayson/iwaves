"""
Damped KdV equation
"""
import iwaves.utils.isw as iwaves
import kdvimex as kdv
import numpy as np
from scipy import sparse 
from scipy.sparse import linalg
from scipy import linalg as la

import pdb



class KdVDamp(kdv.KdVImEx):

    Ricr = 1.0
    tau_d = None
    k_diss = 1.

    def __init__(self, rhoz, z, wavefunc=iwaves.sine, **kwargs):
        """
        Numerical KdV solution
        """
        kdv.KdVImEx.__init__(self, rhoz, z, wavefunc=wavefunc, **kwargs)

        # Need to find the critical amplitude
        self.A_cr = self.calc_Acr()

    def calc_Acr(self):
        phi_z = np.gradient(self.phi_1, -self.z)
        phi_zz = np.gradient(phi_z, -self.z)
        A_ri = self.N2 / (self.c1**2. * self.Ricr * phi_zz**2.)

        # We want the minimum stable amplitude i.e. the most unstable
        A_cr = np.sqrt(np.min(np.abs(A_ri)))

        return A_cr


    def calc_taud(self, An):
        """
        Calcs the dissipation time scale
        """

        N2phi = np.max(self.N2*self.phi_1)
        den = np.abs(An*N2phi)

        idx = np.abs(den<1e-7)
        den[idx] = 1.

        #den = ( self.N2[...,np.newaxis]*\
        #        np.abs(An[np.newaxis,...]) * self.phi_1[...,np.newaxis] )

        ## Just set the timescale large when divide by zeros are encountered

        tau_d =  self.c1 /den

        tau_d[idx] = 1e7

        return tau_d

        #return np.min(tau_d,axis=0)
        

    def build_nonlinear_matrix(self, An):
        """
        Build the nonlinear steepening term
        """
        diags = self.build_nonlinear_diags(An)

        # Ri-based damping scheme
        # Compute the terms in the scheme
        dA = np.abs(An) - self.A_cr
        dA[dA<0] = 0.
        tau_d = self.calc_taud(An)
        diags[2,:] += -self.k_diss * dA / tau_d

        # Build the sparse matrix
        M = sparse.spdiags(diags, [-2,-1,0,1,2], self.Nx, self.Nx)

        return M




