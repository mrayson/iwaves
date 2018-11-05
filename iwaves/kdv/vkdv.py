"""
Variable coefficient code
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy import sparse 

from .kdvimex import  KdVImEx as KdV
from iwaves.utils import isw 
from iwaves.utils.tools import grad_z

import xarray as xray

import pdb

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

    return np.sqrt(num/den)


GRAV=9.81
RHO0=1000.

class vKdV(KdV):
    """
    Variable-coefficient (depth-dependent) KdV solver
    """

    def __init__(self, rhoz, z, h, x, mode,\
        Nsubset=1,\
        fweight=1.,\
        rhoZ=None,
        Cn=None,
        Phi=None,
        Alpha=None,
        Beta=None,
        Qterm=None,
        phi01=None,
        phi10=None,
        D01=None,
        D10=None,
        wavefunc=isw.sine, **kwargs):
	
        # Initialise properties
        # (This is ugly but **kwargs are reserved for the superclass)

        ekdv=False # Hard wire this for now 

        self.mode = mode
        self.x = x
        self.h = h
        self.Nsubset = Nsubset
        self.fweight = fweight

        # These variables can be input so they don't need to be computed
        self.rhoZ = rhoZ
        self.Cn = Cn
        self.Phi = Phi
        self.Alpha = Alpha
        self.Beta = Beta
        self.Qterm = Qterm
        self.phi01 = phi01
        self.phi10 = phi10
        self.D01 = D01
        self.D10 = D10

        Nz = z.shape[0]
        Nx = x.shape[0]

        self.Nz = Nz
        self.Nx = Nx

        ## Create a 2D array of vertical coordinates
        self.Z = -np.linspace(0,1,Nz)[:, np.newaxis] * h[np.newaxis,:]

        #self.dZ = h/(Nz-1)
        self.dZ = np.abs(self.Z[1,:]-self.Z[0,:])

        self.dx = np.diff(x).mean()
        self.X = x[np.newaxis,...] * np.ones((Nz,1))

        # Interpolate the density profile onto all points
        if self.rhoZ is None:
            Fi = interp1d(z, rhoz, axis=0, fill_value='extrapolate')
            self.rhoZ = Fi(self.Z)


        # Only calculate the parameters if they aren't specified
        self.N2 = self.calc_N2()
        if self.Phi is None:
            self.Phi, self.Cn, self.Alpha, self.Beta, self.Qterm =\
                self.calc_vkdv_params(Nz, Nx)

        # Now initialise the class
        KdV.__init__(self, rhoz, z, wavefunc=wavefunc, x=x, mode=mode, ekdv=ekdv, **kwargs)

        self.c1 = self.Cn

        # Change these to be consistent with the Lamb discretization
        #self.r10 = self.Alpha/(-2*self.c1)
        #self.r01 = -self.Beta

        # 
        self.dt_s = np.min(self.dt_s)

        if self.ekdv:
            raise Exception('Extended-KdV not currently supported for spatially-varying model.')

    def calc_N2(self):
        drho_dz = grad_z(self.rhoZ, self.Z,  axis=0)
        return -GRAV*drho_dz/RHO0

    def calc_vkdv_params(self, Nz, Nx):
        # Initialise arrays
        Phi = np.zeros((Nz, Nx))
        Cn = np.zeros((Nx,))
        Alpha = np.zeros((Nx,))
        Beta = np.zeros((Nx,))
        Q = np.zeros((Nx,))

        # Loop through and compute the eigenfunctions etc at each point
        print('Calculating eigenfunctions...')
        phi0, cn0 = isw.iwave_modes(self.N2[:,0], self.dZ[0])
        phi0 = phi0[:,self.mode]
        phi0 = phi0 / np.abs(phi0).max()
        phi0 *= np.sign(phi0.sum())
 
        for ii in range(0, Nx, self.Nsubset):
            point = Nx/100
            if(ii % (5 * point) == 0):
                print('%3.1f %% complete...'%(float(ii)/Nx*100))

            #phi, cn = iwave_modes_sparse(N2[:,ii], dZ[ii], h[ii])
            #phi, cn = isw.iwave_modes(self.N2[:,ii], self.dZ[ii], h[ii])
            phi, cn = isw.iwave_modes(self.N2[:,ii], self.dZ[ii])

            # Extract the mode of interest
            phi_1 = phi[:,self.mode]
            c1 = cn[self.mode]
            
            # Normalize so the max(phi)=1
            phi_1 = phi_1 / np.abs(phi_1).max()
            phi_1 *= np.sign(phi_1.sum())

            # Work out if we need to flip the sign (only really matters for higher modes)
            dphi0 = phi0[1]-phi0[0]
            dphi1 = phi_1[1]-phi_1[0]
            if np.sign(dphi1) != np.sign(dphi0):
                phi_1 *= -1

            Cn[ii] = c1
            Phi[:,ii] = phi_1


        # Interpolate all of the variables back onto the regular grid
        x = self.x
        idx = list(range(0,Nx,self.Nsubset))
        interpm = 'cubic'
        F = interp1d(x[idx],Cn[idx], kind=interpm, fill_value='extrapolate')
        Cn = F(x)

        F = interp1d(x[idx],Phi[:,idx], kind=interpm, axis=1, fill_value='extrapolate')
        Phi = F(x)

        for ii in range(self.Nx):
            phi_1 = Phi[:,ii]
            Alpha[ii] = calc_alpha(phi_1, c1, self.dZ[ii])
            Beta[ii] = calc_beta(phi_1, c1, self.dZ[ii])
            Q[ii] = calc_Qamp(phi_1, Phi[:,0],\
                Cn[ii], Cn[0], self.dZ[ii], self.dZ[0])

        # Weight the nonlinear terms
        Alpha *= self.fweight

        # Calculate the Q-term in the equation here
        Q_x = np.gradient(Q, self.dx)
        Qterm = Cn/(2.*Q) * Q_x

        return Phi, Cn ,Alpha, Beta, Qterm


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

        # Adjust for the Neumann boundary conditions
        self.insert_bcs(diags)

        # Build the sparse matrix
        M = sparse.spdiags(diags, [-2,-1,0,1,2], self.Nx, self.Nx)

        return M, diags

    def calc_linearstructure(self):
    	return self.Phi, self.Cn

    def calc_nonlinstructure(self):
        # Structure function for higher powers of epsilon & mu

        if self.phi01 is not None and self.phi10 is not None:
             return self.phi01, self.phi10, None

        phi01 = np.zeros((self.Nz, self.Nx))
        phi10 = np.zeros((self.Nz, self.Nx))

        print('Calculating nonlinear structure functions...')
        for ii in range(0, self.Nx, self.Nsubset):
            point = self.Nx/100
            if(ii % (5 * point) == 0):
                print('%3.1f %% complete...'%(float(ii)/self.Nx*100))

            rhs01 = isw.calc_phi01_rhs(self.Phi[:,ii], \
                self.c1[ii], self.N2[:,ii], self.dZ[ii])
            phi01[:,ii] = isw.solve_phi_bvp(rhs01,\
                self.N2[:,ii], self.c1[ii], self.dZ[ii])
            rhs10 = isw.calc_phi10_rhs(self.Phi[:,ii],\
                self.c1[ii], self.N2[:,ii], self.dZ[ii])
            phi10[:,ii] = isw.solve_phi_bvp(rhs10, \
                self.N2[:,ii], self.c1[ii], self.dZ[ii])

            # Interpolate all of the variables back onto the regular grid
            idx = list(range(0,self.Nx,self.Nsubset))
            interpm = 'cubic'

            F = interp1d(self.X[0,idx], phi01[:,idx], kind=interpm,\
                    axis=1, fill_value='extrapolate')
            phi01 = F(self.X[0,:])

            F = interp1d(self.X[0,idx], phi10[:,idx], kind=interpm,\
                    axis=1, fill_value='extrapolate')
            phi10 = F(self.X[0,:])


        return phi01, phi10, None

    def calc_coeffs(self):
    	return -self.Beta, self.Alpha/(-2*self.Cn), None, None

    def calc_buoyancy_coeffs(self):

        if self.D01 is not None and self.D10 is not None:
            return self.D01, self.D10, None

        D01 = np.zeros((self.Nz, self.Nx))
        D10 = np.zeros((self.Nz, self.Nx))

        print('Calculating buoyancy coefficients...')
        for ii in range(0,self.Nx, self.Nsubset):
            D01[:,ii] = isw.calc_D01(self.Phi[:,ii], self.c1[ii],\
                self.N2[:,ii], self.dZ[ii])
            D10[:,ii] = isw.calc_D10(self.Phi[:,ii], self.c1[ii],\
                self.N2[:,ii], self.dZ[ii])

        # Interpolate all of the variables back onto the regular grid
        idx = list(range(0,self.Nx,self.Nsubset))
        interpm = 'cubic'

        F = interp1d(self.X[0,idx], D01[:,idx], kind=interpm,\
                axis=1, fill_value='extrapolate')
        D01 = F(self.X[0,:])

        F = interp1d(self.X[0,idx], D10[:,idx], kind=interpm,\
                axis=1, fill_value='extrapolate')
        D10 = F(self.X[0,:])

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
        psi = B[np.newaxis, :]*self.phi_1 * self.c1

        ## First-order nonlinear terms
        if nonlinear:
            psi += B[np.newaxis,:]**2. * self.phi10 * self.c1**2.
            psi += B_xx[np.newaxis,:] * self.phi01 * self.c1
        
            
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
        
        return b.T # Need to return the dimensions

    def calc_density(self, nonlinear=True, method='l96'):
        """
        Returns density
        
        Method: 
            'h99' holloway 1999
            'l96' lamb 1996
        """
        b = self.calc_buoyancy(nonlinear=nonlinear)
        return RHO0*(b/GRAV) + self.rhoZ.T - RHO0

    def calc_velocity(self, nonlinear=True):
        """
        Return the velocity components 
        
        u = d \psi /dz
        w = -d \psi /dx
        """
        psi = self.calc_streamfunction(nonlinear=nonlinear)
        #us, ws = np.gradient(psi)
        #return -us/self.dZ[np.newaxis,...], -ws/self.dx_s
        u = grad_z(psi, self.Z, axis=0)
        w = -1* grad_z(psi, self.X, axis=1)
        return u.T, w.T
 
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

        attrs = {'long_name':'Depth',\
                'units':'m'}
                
        h = xray.DataArray(self.h,
            dims = dims,\
            coords = coords,\
            attrs = attrs,\
        )

 

        attrs = {'long_name':'Nonlinearity',\
                'units':'m-1'}
                
        Alpha = xray.DataArray(self.Alpha,
            dims = dims,\
            coords = coords,\
            attrs = attrs,\
        )

        attrs = {'long_name':'Dispersion',\
                'units':'m-1'}
                
        Beta = xray.DataArray(self.Beta,\
            dims = dims,\
            coords = coords,\
            attrs = attrs,\
        )

        attrs = {'long_name':'Phase Speed',\
                'units':'m s-1'}
                
        Cn = xray.DataArray(self.Cn,
            dims = dims,\
            coords = coords,\
            attrs = attrs,\
        )

        attrs = {'long_name':'Topographic amplification term',\
                'units':'xx'}
                
        Qterm = xray.DataArray(self.Qterm,
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

        ######
        # 2D Functions
        coords = {'x':self.x, 'z':self.Z[:,0]}
        dims = ('z','x')
        attrs = {'long_name':'X-coordinate',\
                'units':''}
                
        X = xray.DataArray(self.X,
            dims = dims,\
            coords = coords,\
            attrs = attrs,\
        )
        attrs = {'long_name':'Z-coordinate',\
                'units':''}
                
        Z = xray.DataArray(self.Z,
            dims = dims,\
            coords = coords,\
            attrs = attrs,\
        )

        attrs = {'long_name':'Density',\
                'units':'kg m-3'}
                
        rhoZ = xray.DataArray(self.rhoZ,
            dims = dims,\
            coords = coords,\
            attrs = attrs,\
        )

        attrs = {'long_name':'Eigenfunction',\
                'units':''}
                
        Phi = xray.DataArray(self.Phi,
            dims = dims,\
            coords = coords,\
            attrs = attrs,\
        )

        phi01 = xray.DataArray(self.phi01,
            dims = dims,\
            coords = coords,\
            attrs = attrs,\
        )

        phi10 = xray.DataArray(self.phi10,
            dims = dims,\
            coords = coords,\
            attrs = attrs,\
        )

        D01 = xray.DataArray(self.D01,
            dims = dims,\
            coords = coords,\
            attrs = attrs,\
        )

        D10 = xray.DataArray(self.D10,
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
                'spongedist',\
                #'c1',\
                #'mu',\
                #'epsilon',\
                #'r01',\
                #'r10',\
                't',\
                #'ekdv',
        ]

        attrs = {}
        for aa in saveattrs:
            attrs.update({aa:getattr(self, aa)})

        attrs.update({'Description':'1D variable-coefficient KdV Solution'})

        return xray.Dataset({'B':B,\
                        'Alpha':Alpha,\
                        'Beta':Beta,\
                        'Qterm':Qterm,\
                        'h':h,\
                        'Cn':Cn,\
                        'X':X,\
                        'Z':Z,\
                        'rhoZ':rhoZ,\
                        'Phi':Phi,\
                        'phi01':phi01,\
                        'phi10':phi10,\
                        'D01':D01,\
                        'D10':D10,\
                        #'rhoz':rhoz,\
                        }, attrs=attrs)



