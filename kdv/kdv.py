# coding: utf-8

# # Numerical KdV Solver

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse, linalg
from scipy.interpolate import interp1d
import xarray as xray

import matplotlib.pyplot as plt

from iwaves.utils.isw import *

import pdb 


###################
# Constants
RHO0=1000
GRAV=9.8

class KdV(object):
   
    ###################
    # Default inputs
    ###################
    Nx = 12000

    # Domain length
    L_d = 1.5e5

    x = None

    # Depth
    #H = 300.
    
    # Nondimensional velocity scale
    U = 1.0

    # Initial wave amplitude
    a0 = 20.
    x0 = None

    # Initial wave length
    Lw = 5000.

    # Initial wave eigenmode
    mode=0
    
    # Courant number
    Cmax = 0.01
    dt = None
    
    # Nondimensionalize the dimensions
    nondim = False
    
    # Deactivate nonlinear term
    nonlinear = True

    # Turn on/off dispersion term
    nonhydrostatic = True

    # extended KdV solver
    ekdv = True

    # Horizontal eddy viscosity
    nu_H = 0.0

    # time counter
    t = 0.

    # Higher order correction factors
    alpha_10 = 0. # -0.008
    alpha_20 = 0. # 3e-5

    # Nonlinear scaling factor for r10 (for testing)
    nonlin_scale = 1.
    
    
    def __init__(self, rhoz, z, wavefunc=sine, **kwargs):
        """
        Numerical KdV solution
        """
        self.__dict__.update(**kwargs)
                
        # These need to be copied...
        self.rhoz = 1*rhoz
        self.z = 1*z

        ####
        # Initialise the domain

        self.H = np.abs(self.z).max()
        
        self.Nz = rhoz.shape[0]
        self.dz = np.abs(self.z[1]-self.z[0])

	if self.x is None:
	    self.x = np.linspace(-self.L_d, self.L_d, self.Nx)
	else:
	    self.Nx = self.x.shape[0]

        #self.x = np.linspace(0, self.L_d, self.Nx)
        self.dx = np.abs(self.x[1]-self.x[0])
        
        ####
        # Calculate the non-dimensional coordinates and parameters
        #self.L = 1000*self.dx # Use the grid spacing as the choice of scaling parameter
        self.L = self.Lw
        if self.nondim:
            self.mu = (self.H/self.L)**2.
            self.epsilon = self.a0/self.H
        else:
            self.mu = 1.
            self.epsilon = 1.
        
        # All non-dimensional coordinates have a subscript "_s"
        if self.nondim:
            self.dx_s = self.dx/self.L
            self.dz_s = self.dz/self.H
            
            self.rhoz /= RHO0
            
        else:
            self.dx_s = self.dx
            self.dz_s = self.dz
        
        # Time is later...

	# Calculate the eigenfunctions/values
	self.phi_1, self.c1 = self.calc_linearstructure()
        
        # Find the location of max (phi) - used to normalize high order terms
        self.kmax = np.argwhere(np.abs(self.phi_1) == np.abs(self.phi_1).max())[0,0]

        
	self.r01, self.r10, self.r20, self.T10 = self.calc_coeffs()

        ####
        # Nondimensional time step
        if self.dt is None:
            self.dt_s = self.Cmax * self.dx / self.c1
            if self.nondim:
                self.dt_s = self.dt_s*self.c1/self.Lw
        else:
            self.dt_s = 1*self.dt

        #print self.dx/self.c1, 

	####
	# Calculate the nonlinear correction terms
	self.phi01, self.phi10, self.phi20 = self.calc_nonlinstructure()

        self.D01, self.D10, self.D20 = self.calc_buoyancy_coeffs()

        #########
        # Initialise the wave function B(x,t)
        #wavefunc = iwaves.sine # Hardwire for now...
        if self.x0 is None:
            self.x0 = self.Lw/2

        self.B_n_m2, self.B_n_m1, self.B, self.B_n_p1 = self.init_wave(wavefunc)
    
    def calc_linearstructure(self):
        ####
        # Calculate the linear vertical structure functions
        self.N2 = self.calc_N2()
        if self.nondim:
            self.N2 = self.N2*self.H**2/self.U**2
            
        phi, cn = iwave_modes(self.N2, self.dz_s)

        # Extract the mode of interest
        phi_1 = phi[:,self.mode]
        c1 = cn[self.mode]

        # Normalize so the max(phi)=1
        phi_1 = phi_1 / np.abs(phi_1).max()
        phi_1 *= np.sign(phi_1.sum())

	return phi_1, c1


    def init_wave(self, wavefunc):
        
        A = wave_eta(self.x, self.a0, self.c1, self.Lw,\
                wavefunc=wavefunc, x0=self.x0)
        
        if self.nondim:
            A /= self.a0
            
        B = A  #/ self.c1 # ?? Not sure about this (I don't think it should be scaled)
        B_n_m1 = B*1. # n-1 time step
        B_n_m2 = B*1. # n-2 time step
        B_n_p1 = np.zeros_like(self.x) # n+1 time step
        
        return B_n_m2, B_n_m1, B, B_n_p1, 
        
    def solve_step(self):
        """
        Solve the KdV for one time step
        """    
        status =0
        self.t += self.dt_s

        M = self.build_matrix_sparse(self.B)

        # Solve the next step

        # Second-order time stepping
        self.B_n_p1[:] = self.B_n_m1 + 2*self.dt_s * M.dot(self.B)

        # First-order time stepping
        #self.B_n_p1[:] = self.B + self.dt_s * M.dot(self.B)

        # Check solutions
        if np.any( np.isnan(self.B_n_p1)):
            return -1

        # Ensure the boundaries match the interior values i.e. dB/dx = 0 at BCs
        #self.B_n_p1[0] = self.B_n_p1[1]
        #self.B_n_p1[-1] = self.B_n_p1[-2]

        # Update the terms last
        self.B_n_m2[:] = self.B_n_m1
        self.B_n_m1[:] = self.B
        self.B[:] = self.B_n_p1

        return status

    def calc_coeffs(self):
	# Compute nonlinear and dispersion constants
        r01 = calc_r01(self.phi_1, self.c1, self.dz_s)
        r10 = calc_r10(self.phi_1, self.c1, self.N2, self.dz_s)
        #r10 = alpha(self.phi_1, self.c1, self.N2, self.dz_s)
        r20 = calc_r20(self.phi_1, self.c1, self.N2, self.dz_s)

        # Holloway 99 nonlinear correction
        T10 = calc_T10(self.phi_1, self.c1, self.N2, self.dz_s)

	return r01, self.nonlin_scale*r10, r20, T10

    def calc_nonlinstructure(self):
        # Structure function for higher powers of epsilon & mu
        phi01 = calc_phi01(self.phi_1, self.c1, self.N2, self.dz_s)

        phi10 = calc_phi10(self.phi_1, self.c1, self.N2, self.dz_s)

        #if self.ekdv:
        phi20 = calc_phi20(self.phi_1, self.c1, self.N2, self.dz_s)

        return phi01, phi10, phi20
 
    def calc_buoyancy_coeffs(self):

        D01 = calc_D01(self.phi_1, self.c1, self.N2, self.dz_s)
        D10 = calc_D10(self.phi_1, self.c1, self.N2, self.dz_s)
        D20 = calc_D20(self.phi_1, self.c1, self.N2, self.dz_s)

        return D01, D10, D20

    def build_matrix_sparse(self, An):
        """
        Build the LHS sparse matrix 
        """ 

        diags = np.zeros((5,self.Nx))

        # Constants
        cff1 = 1*self.mu*self.r01
        #cff1 = 0
        dx3 = 1./np.power(self.dx_s,3.)

        # Equations in Lamb & Yan
        # pressure terms
        diags[1,:] -= (-0.5*self.c1/self.dx_s) * np.ones((self.Nx,)) # i-1
        diags[3,:] -= (+0.5*self.c1/self.dx_s) * np.ones((self.Nx,)) # i+1

        # Dispersion term (2nd order)
        if self.nonhydrostatic:
            diags[0,:] += -0.5*cff1*dx3 * np.ones((self.Nx,)) # i-2
            diags[1,:] += (+cff1*dx3) * np.ones((self.Nx,)) # i-1
            diags[3,:] += (-cff1*dx3) * np.ones((self.Nx,)) # i+1
            diags[4,:] += 0.5*cff1*dx3 * np.ones((self.Nx,)) # i+2

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
        #diags[1,:] += nu_H*dx2 * np.ones((self.Nx,))
        #diags[2,:] -= 2*(nu_H*dx2) * np.ones((self.Nx,))
        #diags[3,:] += nu_H*dx2* np.ones((self.Nx,))

        # 4th order
        c1 = -1/12.
        c2 = 16/12.
        c3 = -30/12.
        c4 = 16/12.
        c5 = -1/12.
        diags[0,:] += c1*nu_H*dx2 * np.ones((self.Nx,))
        diags[1,:] += c2*nu_H*dx2 * np.ones((self.Nx,))
        diags[2,:] += c3*nu_H*dx2 * np.ones((self.Nx,))
        diags[3,:] += c4*nu_H*dx2* np.ones((self.Nx,))
        diags[4,:] += c5*nu_H*dx2 * np.ones((self.Nx,))


        #print diags.max(axis=1)
        #print cff1, cff2, dxs, cff1/dxs**3.

        # Add the nonlinear terms
        cff2 = 2*self.epsilon*self.r10*self.c1 # Written like this in the paper
        #cff2 = 2*self.epsilon*self.r10
        cff3 = 0.5*cff2/self.dx_s
        cff3 *= 0.5# factor 0.5 is because I am taking average
        if self.nonlinear:
            diags[1,:] = diags[1,:] - cff3*An # i-1
            diags[3,:] = diags[3,:] + cff3*An # i+1
            
            #diags[1,1:] = diags[1,1:] - cff3*An[0:-1] # i-1
            #diags[3,0:-1] = diags[3,0:-1] + cff3*An[1:] # i+1
            #diags[1,0:-1] = diags[1,0:-1] - cff3*An[1:] # i-1
            #diags[3,1:] = diags[3,1:] + cff3*An[0:-1] # i+1
            #diags[1,0:-1] = diags[1,1:] - cff3*An[0:-1] # i-1
            #diags[3,1:] = diags[3,0:-1] + cff3*An[1:] # i+1

            #diags[1,0:-2] = diags[1,0:-2] + cff3*An[1:-1]
            #diags[1,1:-1] = diags[1,1:-1] - cff3*An[0:-2]

            #diags[0,0:-2] = diags[0,0:-2] + cff3*An[1:-1] # i+1
            #diags[0,1:-1] = diags[0,1:-1] - cff3*An[0:-2] # i-1
            
            
        # extended KdV
        if self.ekdv:
            cff4 = 3*self.epsilon**2*self.r20*self.c1**2
            cff5 = 0.5*cff4/self.dx_s
            An2 = 0.25*np.power(An, 2.) # factor 0.5 is because I am taking average
            diags[1,:] = diags[1,:] - cff5*An2
            diags[3,:] = diags[3,:] + cff5*An2
            #diags[1,1:] = diags[1,1:] - cff5*An2[0:-1] # i-1
            #diags[3,0:-1] = diags[3,0:-1] + cff5*An2[1:] # i+1
            
        # Zero ends...
        #diags[:,0] =0.
        #diags[:,-1] =0.

        # Build the sparse matrix
        M = sparse.spdiags(diags, [-2,-1,0,1,2], self.Nx, self.Nx)

        # Convert to CSR format as this is more efficient (it isn't...)
        #M = M.tocsr()

        return M
    
    def calc_Bxx(self):
        B = self.B # local pointer
        B_xx = np.zeros_like(B)
        B_xx[1:-1] = (B[0:-2] - 2*B[1:-1] + B[2:])/self.dx_s**2
        return B_xx
        
    def calc_streamfunction(self, nonlinear=True):
        """
        Calculate the stream function up to the first power of epsilon and mu
        """
        B = self.B
        
        #phi01, phi10 = self.calc_nonlinstructure()

        # Calculate the second-derivative
        B_xx = self.calc_Bxx()

        A = B[:,np.newaxis] * self.c1
        A_xx = B_xx * self.c1


        # Linear streamfunction
        psi = A*self.phi_1 #*self.c1

        # First-order nonlinear terms
        if nonlinear:
            psi += self.epsilon * A**2. * self.phi10 
            psi += self.mu * A_xx[:,np.newaxis] * self.phi01 

            if self.ekdv:
                psi += self.epsilon * A**3. * self.phi20
        
        if self.nondim:
            psi = psi/(self.epsilon*self.U*self.H)
            
        return psi
    
   
    def calc_velocity(self, nonlinear=True):
        """
        Return the velocity components 
        
        u = d \psi /dz
        w = -d \psi /dx
        """
        psi = self.calc_streamfunction(nonlinear=nonlinear)
        ws, us = np.gradient(psi)
        
        return -us/self.dz_s, -ws/self.dx_s
    
    def calc_buoyancy(self, nonlinear=True):
        """
        Calculate the buoyancy perturbation: b = g*rho'
        """
        B = self.B
        B_xx = self.calc_Bxx()

        # Use the dimensional N2
        N2 = 1*self.N2
        if self.nondim:
            N2 *=self.U**2/self.H**2.
        
        A = B[:,np.newaxis] * self.c1
        A_xx = B_xx * self.c1
        
        # Linear component, See lamb & yan Eq. (3.16) (no c_n)
        b = A*self.phi_1*self.N2/self.c1
        
        ## Nonlinear components
        if nonlinear:
            b += self.epsilon*A**2.*self.D10
        
            b += self.mu*A_xx[:,np.newaxis]*self.D01

            if self.ekdv:
                b += self.epsilon*A**3.*self.D20
        
        if self.nondim:
            b *= self.H/(self.epsilon*self.U**2.)
            
        return b
    
    def calc_buoyancy_h99(self, nonlinear=True):
        """
        Use the Holloway et al 99 version of the eqn's
        """
        dN2_dz = np.gradient(self.N2, -np.abs(self.dz_s))
        
        # Linear term
        b = self.B[:,np.newaxis] * self.phi_1 * self.N2
        
        #alpha = self.r10/(2*self.c1) ??
        alpha = -2*self.c1*self.r10
        
        # nonlinear terms
        if nonlinear:
            b -= alpha/(2*self.c1)*self.B[:,np.newaxis]*self.phi_1*self.N2
            b -= 0.5*dN2_dz*self.B[:,np.newaxis]**2. * self.phi_1**2.
            b += self.c1*self.B[:,np.newaxis]**2. *self.N2 * self.T10
            
        return b
            
        
    def calc_density(self, nonlinear=True, method='l96'):
        """
        Returns density
        
        Method: 
            'h99' holloway 1999
            'l96' lamb 1996
            'exact' interpolate density from the perturbation height
        """
        if method == 'exact':
            eta_pr = self.B[:,np.newaxis]*self.phi_1 # Need to add the nonlinear components
    
            # Interpolation function
            Frho = interp1d(self.z, self.rhoz, axis=0, fill_value='extrapolate')
    
            eta = self.z[np.newaxis,:] - eta_pr
    
            #eta[eta>0.] = 0.
            #eta[eta<-d] = -d
    
            # Find rho by interpolating eta
            rho = Frho(eta) - RHO0
            return rho

        if method == 'h99':
            b = self.calc_buoyancy_h99(nonlinear=nonlinear)
        elif method == 'l96':
            b = self.calc_buoyancy(nonlinear=nonlinear)


        #rho1 =  RHO0*(( b/GRAV + self.rhoz[np.newaxis,:]/RHO0 - 1))
        rho = b*RHO0/GRAV + self.rhoz[np.newaxis,:] - RHO0
        return rho
        #return RHO0*(b/GRAV) + self.rhoz[np.newaxis,:] - RHO0
        #return (b/GRAV + self.rhoz[np.newaxis,:]) - RHO0
    
    def calc_N2(self):
        """
        Calculate the buoyancy frequency
        """
        drho_dz = np.gradient(self.rhoz, -np.abs(self.dz))
        N2 = -GRAV*drho_dz
        if not self.nondim:
            N2/=RHO0

        return N2
        
    #####
    # Printing routines
    def print_params(self):
        """
        Print parameters of interests
        """
        printstr = 'Parameters:\n c1 = %3.6f\n epsilon = %3.6f\n'%            (self.c1, self.epsilon)
        printstr += ' mu = %3.6f\n r01 = %3.6f\n r10 = %3.6f\n'%            (self.mu, self.r01, self.r10)

        printstr += ' r20 = %3.7f\n'%(self.r20)

        print printstr

    ######
    # IO methods
    def to_Dataset(self):
        """
        Convert to an xray dataset object
        """
        ######
        # Amplitude function
        coords = {'x':self.x}
        attrs = {'long_name':'Wave amplitude',\
                'units':'m'}
        dims = ('x')
                
        B = xray.DataArray(self.B,
            dims = dims,\
            coords = coords,\
            attrs = attrs,\
        )

        #######
        # density profile
        coords = {'z':self.z}
        attrs = {'long_name':'Water density',\
                'units':'kg m-3'}
        dims = ('z')
                
        rhoz = xray.DataArray(self.rhoz,
            dims = dims,\
            coords = coords,\
            attrs = attrs,\
        )

        #########
        # Dictionary of attributes
        # List of attributes
        saveattrs = ['Nx',\
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

        attrs = {}
        for aa in saveattrs:
            attrs.update({aa:getattr(self, aa)})

        attrs.update({'Description':'1D KdV Solution'})

        return xray.Dataset({'B':B,'rhoz':rhoz}, attrs=attrs)


