"""
Variable coefficient code
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy import sparse 

from .kdvimex import  KdVImEx as KdV
from iwaves.utils import isw 
from iwaves.utils.tools import grad_z

def calc_alpha(phi, c, dz):
    phi_z = np.gradient(phi,-dz)
    num = 3*c*np.trapz( phi_z**3., dx=dz)
    den = 2*np.trapz( phi_z**2., dx=dz)

    return num/den

def calc_beta(phi, c, dz):
    phi_z = np.gradient(phi, dz)
    num = c*np.trapz( phi**2., dx=dz)
    den = 2*np.trapz( phi_z**2., dx=dz)

    return num/den

def calc_Qamp(phi, phi0, c, c0, dz, dz0):
    phi0_z = np.gradient(phi0, dz0)
    phi_z = np.gradient(phi, dz)
    num = c0**3. * np.trapz( phi0_z**2., dx=dz0)
    den = c**3. * np.trapz( phi_z**2., dx=dz)
    #num = c0**3. * np.sum( phi0_z**2. * dz0)
    #den = c**3. * np.sum( phi_z**2. * dz)

    return num/den


GRAV=9.81
RHO0=1000.

class vKdV(KdV):
    """
    Variable-coefficient (depth-dependent) KdV solver
    """

    def __init__(self, rhoz, z, h, x, mode, wavefunc=isw.sine, **kwargs):
	
        ekdv=False # Hard wire this for now 

	Nz = z.shape[0]
	Nx = x.shape[0]

        self.Nz = Nz
        self.Nx = Nx
	## Create a 2D array of vertical coordinates
	self.Z = -np.linspace(0,1,Nz)[:, np.newaxis] * h[np.newaxis,:]

	self.dZ = h/Nz

	dx = np.diff(x).mean()
	self.X = x[np.newaxis,...] * np.ones((Nz,1))

	# Interpolate the density profile onto all points
	Fi = interp1d(z, rhoz, axis=0)
	self.rhoZ = Fi(self.Z)

	drho_dz = grad_z(self.rhoZ, self.Z,  axis=0)
	self.N2 = -GRAV*drho_dz/RHO0

	# Initialise arrays
	self.Phi = np.zeros((Nz, Nx))
	self.Cn = np.zeros((Nx,))
	self.Alpha = np.zeros((Nx,))
	self.Beta = np.zeros((Nx,))
	Q = np.zeros((Nx,))

	# Loop through and compute the eigenfunctions etc at each point
        print 'Calculating eigenfunctions...'
	for ii in range(Nx):
	    point = Nx/100
	    if(ii % (5 * point) == 0):
		print '%3.1f %% complete...'%(float(ii)/Nx*100)

	    #phi, cn = iwave_modes_sparse(N2[:,ii], dZ[ii], h[ii])
	    #phi, cn = isw.iwave_modes(self.N2[:,ii], self.dZ[ii], h[ii])
	    phi, cn = isw.iwave_modes(self.N2[:,ii], self.dZ[ii])

	    # Extract the mode of interest
	    phi_1 = phi[:,mode]
	    c1 = cn[mode]
	    
	    # Normalize so the max(phi)=1
	    phi_1 = phi_1 / np.abs(phi_1).max()
	    phi_1 *= np.sign(phi_1.sum())

	    self.Cn[ii] = c1
	    self.Phi[:,ii] = phi_1

	    self.Alpha[ii] = calc_alpha(phi_1, c1, self.dZ[ii])
	    self.Beta[ii] = calc_beta(phi_1, c1, self.dZ[ii])
	    Q[ii] = calc_Qamp(phi_1, self.Phi[:,0],\
	    	c1, self.Cn[0], self.dZ[ii], self.dZ[0])

	# Calculate the Q-term in the equation here
	Q_x = np.gradient(Q, dx)
	self.Qterm = self.Cn/(2.*Q) * Q_x

	# Now initialise the class
	KdV.__init__(self, rhoz, z, wavefunc=wavefunc, x=x, mode=mode, ekdv=ekdv, **kwargs)

	# 
	self.dt_s = np.min(self.dt_s)

        if self.ekdv:
            raise Exception, 'Extended-KdV not currently supported for spatially-varying model.'

    def build_linear_matrix(self):
        """
        Build the linear matrices

	Overloaded function to include:
	    - spatially variable coefficients
	    - topographic amplification term
        """
	#self.r10 = -self.Alpha
	#self.r01 = -self.Beta
        M,diags = KdV.build_linear_matrix(self)

	# Add on the Q-term
	diags[2,:] += self.Qterm

        # Build the sparse matrix
        M = sparse.spdiags(diags, [-2,-1,0,1,2], self.Nx, self.Nx)

        return M, diags

    def calc_linearstructure(self):
    	return self.Phi, self.Cn

    def calc_nonlinstructure(self):
        # Structure function for higher powers of epsilon & mu

	phi01 = np.zeros((self.Nz, self.Nx))
	phi10 = np.zeros((self.Nz, self.Nx))

        print 'Calculating nonlinear structure functions...'
	for ii in range(self.Nx):
	    point = self.Nx/100
	    if(ii % (5 * point) == 0):
		print '%3.1f %% complete...'%(float(ii)/self.Nx*100)

	    rhs01 = isw.calc_phi01_rhs(self.Phi[:,ii], \
	    	self.c1[ii], self.N2[:,ii], self.dZ[ii])
	    phi01[:,ii] = isw.solve_phi_bvp(rhs01,\
	    	self.N2[:,ii], self.c1[ii], self.dZ[ii])
	    rhs10 = isw.calc_phi10_rhs(self.Phi[:,ii],\
	    	self.c1[ii], self.N2[:,ii], self.dZ[ii])
	    phi10[:,ii] = isw.solve_phi_bvp(rhs10, \
	    	self.N2[:,ii], self.c1[ii], self.dZ[ii])

        return phi01, phi10, None

    def calc_coeffs(self):
    	return -self.Beta, -self.Alpha, None, None

    def calc_buoyancy_coeffs(self):

	D01 = np.zeros((self.Nz, self.Nx))
	D10 = np.zeros((self.Nz, self.Nx))

        for ii in range(self.Nx):
            D01[:,ii] = isw.calc_D01(self.Phi[:,ii], self.c1[ii],\
                self.N2[:,ii], self.dZ[ii])
            D10[:,ii] = isw.calc_D10(self.Phi[:,ii], self.c1[ii],\
                self.N2[:,ii], self.dZ[ii])

        return D01, D10, None

    def calc_streamfunction(self, nonlinear=True):
        """
        Calculate the stream function up to the first power of epsilon and mu
        """
        B = self.B
        
        #phi01, phi10 = self.calc_nonlinstructure()

        # Calculate the second-derivative
        B_xx = self.calc_Bxx()

        # Linear streamfunction
        psi = B[np.newaxis, :]*self.phi_1 #*self.c1

        # First-order nonlinear terms
        if nonlinear:
            psi += B[np.newaxis,:]**2. * self.phi10 #* self.c1**2.
            psi += B_xx[np.newaxis,:] * self.phi01 #* self.c1
        
            
        return psi

    def calc_buoyancy(self, nonlinear=True):
        """
        Calculate the buoyancy perturbation: b = g*rho'
        """
        B = self.B
        B_xx = self.calc_Bxx()

        # Use the dimensional N2
        N2 = 1*self.N2
        
        A = B /self.c1
        A_xx = B_xx / self.c1
        
        # Linear component
        b = A[np.newaxis,:]*self.phi_1*N2/self.c1[np.newaxis,:]
        
        ## Nonlinear components
        if nonlinear:
            #D10 = calc_D10(self.phi_1, self.c1, self.N2, self.dz_s)
            b += self.epsilon*A[np.newaxis,:]**2.*self.D10
        
            #D01 = calc_D01(self.phi_1, self.c1, self.N2, self.dz_s)
            b += self.mu*A_xx[np.newaxis,:]*self.D01
        
        return b

    def calc_density(self, nonlinear=True, method='l96'):
        """
        Returns density
        
        Method: 
            'h99' holloway 1999
            'l96' lamb 1996
        """
        b = self.calc_buoyancy(nonlinear=nonlinear)
        return RHO0*(b/GRAV) + self.rhoZ - RHO0

    def calc_velocity(self, nonlinear=True):
        """
        Return the velocity components 
        
        u = d \psi /dz
        w = -d \psi /dx
        """
        psi = self.calc_streamfunction(nonlinear=nonlinear)
        us, ws = np.gradient(psi)
        
        return -us/self.dZ[np.newaxis,...], -ws/self.dx_s
 
 
