"""
Variable coefficient code
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy import sparse 

from .kdvimex import  KdVImEx as KdV
from iwaves.utils import isw 
from iwaves.utils.tools import grad_z, quadinterp

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

def calc_Qamp_new(phi, c, dz):
    """Small 2001 definition
    Normalizing is not necessary as it cancels out"""
    phi_z = np.gradient(phi, dz)
    return c**3. * np.trapz( phi_z**2., dx=dz)

def calc_Qamp(phi, phi0, c, c0, dz, dz0):
    """Holloway 1997 definition"""
    phi0_z = np.gradient(phi0, dz0)
    phi_z = np.gradient(phi, dz)
    den = c0**3. * np.trapz( phi0_z**2., dx=dz0)
    num = c**3. * np.trapz( phi_z**2., dx=dz)

    return num/den

GRAV=9.81
RHO0=1000.

class vKdV(KdV):
    """
    Variable-coefficient (depth-dependent) KdV solver
    """
    verbose = True

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
        phi20=None,
        D01=None,
        D10=None,
        D20=None,
        r20=None,
        wavefunc=isw.sine, **kwargs):
	
        # Initialise properties
        # (This is ugly but **kwargs are reserved for the superclass)

        self.__dict__.update(**kwargs)

        #ekdv=False # Hard wire this for now 

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
        self.phi20 = phi20
        self.D01 = D01
        self.D10 = D10
        self.D20 = D20
        self.r20 = r20
        self.r20 = r20

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
            self.Phi, self.Cn, self.Alpha, self.Beta, self.Qterm, self.r20 =\
                self.calc_vkdv_params(Nz, Nx)

        # Now initialise the class
        KdV.__init__(self, rhoz, z, wavefunc=wavefunc, x=x, mode=mode, \
            **kwargs)

        self.c1 = self.Cn

        # Change these to be consistent with the Lamb discretization
        #self.r10 = self.Alpha/(-2*self.c1)
        #self.r01 = -self.Beta

        # 
        self.dt_s = np.min(self.dt_s)

        #if self.ekdv:
        #    raise Exception('Extended-KdV not currently supported for spatially-varying model.')

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
        r20 = np.zeros((Nx,))

        # Loop through and compute the eigenfunctions etc at each point
        if self.verbose:
            print('Calculating eigenfunctions...')
        phi0, cn0 = isw.iwave_modes(self.N2[:,0], self.dZ[0])
        phi0 = phi0[:,self.mode]
        phi0 = phi0 / np.abs(phi0).max()
        phi0 *= np.sign(phi0.sum())
 
        for ii in range(0, Nx, self.Nsubset):
            point = Nx//100
            if(ii % (5 * point) == 0):
                if self.verbose:
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

            if self.ekdv:
                # Compute cubic nonlinearity on the subsetted grid then interpolate
                r20[ii] = isw.calc_r20(phi_1, c1, self.N2[:,ii], self.dZ[ii])


        # Interpolate all of the variables back onto the regular grid
        x = self.x
        idx = list(range(0,Nx,self.Nsubset))
        interpm = 'cubic'
        F = interp1d(x[idx],Cn[idx], kind=interpm, fill_value='extrapolate')
        Cn = F(x)

        F = interp1d(x[idx],Phi[:,idx], kind=interpm, axis=1, fill_value='extrapolate')
        Phi = F(x)

        if self.ekdv:
            F = interp1d(x[idx],r20[idx], kind=interpm, fill_value='extrapolate')
            r20 = F(x)


        for ii in range(self.Nx):
            phi_1 = Phi[:,ii]
            Alpha[ii] = calc_alpha(phi_1, c1, self.dZ[ii])
            Beta[ii] = calc_beta(phi_1, c1, self.dZ[ii])
            #Q[ii] = calc_Qamp(phi_1, Cn[ii], self.dZ[ii])
            Q[ii] = calc_Qamp(phi_1, Phi[:,0],\
                Cn[ii], Cn[0], self.dZ[ii], self.dZ[0])

        # Zero beta near the boundary

        # Weight the nonlinear terms
        #Alpha *= self.fweight
        #Beta *= self.fweight

        ## Calculate the Q-term in the equation here
        #Q_x = np.gradient(Q, self.dx)
        #Qterm = Cn/(2.*Q) * Q_x

        return Phi, Cn ,Alpha, Beta, Q, r20


    def build_linear_diags(self):
        """
        Build the linear matrices

        Overloaded function to include:
	    - spatially variable coefficients
	    - topographic amplification term
        """
        #self.r10 = -self.Alpha
        #self.r01 = -self.Beta
        diags = KdV.build_linear_diags(self)

        # Add on the Q-term
        #diags[2,:] += self.Qterm
        self.add_topo_effects(diags)
        
        # Adjust for the Dirichlet boundary conditions
        #self.insert_bcs(diags)

        ## Build the sparse matrix
        #M = sparse.spdiags(diags, [-2,-1,0,1,2], self.Nx, self.Nx)

        return diags

    def add_topo_effects(self, diags):
        """
        Add the topographic effect terms to the LHS FD matrix
        """
        #cff = self.Cn / (2*self.Qterm)
        #dx2 = 1/(2*self.dx_s)
        #dQdx = np.ones_like(self.Qterm)
        #dQdx[1:-1] = (self.Qterm[2::] - self.Qterm[0:-2])*dx2

        #diags[2,:] += cff*dQdx
        Q_x = np.gradient(self.Qterm, self.dx_s)
        Qterm = self.Cn/(2.*self.Qterm) * Q_x
        diags[2,:] += Qterm

        #diags[1,:] += cff*self.Qterm*dx2
        #diags[3,:] -= cff*self.Qterm*dx2

    def add_bcs_rhs(self, RHS, cff, A_l):
        # Add boundary condition terms to the RHS
        r01 = -self.Beta[0]
        c = self.Cn[0]

        dx_i = 1/(2*self.dx_s)
        dx3_i = 1./np.power(self.dx_s,3.)

        #A_ll = A_l-cff*c*(self.B_n_m1[1]-A_l)*dx_i
        #A_0 = A_l
        # Quadratic interpolation
        dx=self.dx_s
        A_0 = quadinterp(dx,0,2*dx,3*dx,A_l,RHS[2],RHS[3])

        # Left Dirichlet (linear terms)
        
        # Left Dirichlet (linear terms)
        RHS[0] += cff*(c*A_0*dx_i)
        if self.nonhydrostatic:
            RHS[0] += cff*r01*1.0*A_0*dx3_i
            RHS[0] -= cff*r01*0.5*A_l*dx3_i
            RHS[1] -= cff*r01*0.5*A_0*dx3_i

       
        # Topographic amplification term...not neccessary as it's on the main diag


    def calc_linearstructure(self):
    	return self.Phi, self.Cn

    def calc_nonlinstructure(self):
        # Structure function for higher powers of epsilon & mu

        if self.phi01 is not None and self.phi10 is not None:
             return self.phi01, self.phi10, self.phi20

        phi01 = np.zeros((self.Nz, self.Nx))
        phi10 = np.zeros((self.Nz, self.Nx))
        phi20 = np.zeros((self.Nz, self.Nx))

        if self.verbose:
            print('Calculating nonlinear structure functions...')
        for ii in range(0, self.Nx, self.Nsubset):
            point = self.Nx//100
            if(ii % (5 * point) == 0):
                if self.verbose:
                    print('%3.1f %% complete...'%(float(ii)/self.Nx*100))

            rhs01 = isw.calc_phi01_rhs(self.Phi[:,ii], \
                self.c1[ii], self.N2[:,ii], self.dZ[ii])
            phi01[:,ii] = isw.solve_phi_bvp(rhs01,\
                self.N2[:,ii], self.c1[ii], self.dZ[ii])
            rhs10 = isw.calc_phi10_rhs(self.Phi[:,ii],\
                self.c1[ii], self.N2[:,ii], self.dZ[ii])
            phi10[:,ii] = isw.solve_phi_bvp(rhs10, \
                self.N2[:,ii], self.c1[ii], self.dZ[ii])

            if self.ekdv:
                rhs20 = isw.calc_phi20_rhs(self.Phi[:,ii],\
                    self.c1[ii], self.N2[:,ii], self.dZ[ii])
                phi20[:,ii] = isw.solve_phi_bvp(rhs20, \
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

            if self.ekdv:
                F = interp1d(self.X[0,idx], phi20[:,idx], kind=interpm,\
                        axis=1, fill_value='extrapolate')
                phi20 = F(self.X[0,:])

        return phi01, phi10, phi20

    def calc_coeffs(self):
    	return -self.Beta, self.Alpha/(-2*self.Cn), self.r20, None

    def calc_buoyancy_coeffs(self):

        if self.D01 is not None and self.D10 is not None:
            return self.D01, self.D10, self.D20

        D01 = np.zeros((self.Nz, self.Nx))
        D10 = np.zeros((self.Nz, self.Nx))
        D20 = np.zeros((self.Nz, self.Nx))

        if self.verbose:
            print('Calculating buoyancy coefficients...')
        for ii in range(0,self.Nx, self.Nsubset):
            D01[:,ii] = isw.calc_D01(self.Phi[:,ii], self.c1[ii],\
                self.N2[:,ii], self.dZ[ii])
            D10[:,ii] = isw.calc_D10(self.Phi[:,ii], self.c1[ii],\
                self.N2[:,ii], self.dZ[ii])
            if self.ekdv:
                D20[:,ii] = isw.calc_D20(self.Phi[:,ii], self.c1[ii],\
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

        if self.ekdv:
            F = interp1d(self.X[0,idx], D20[:,idx], kind=interpm,\
                    axis=1, fill_value='extrapolate')
            D20 = F(self.X[0,:])



        return D01, D10, D20

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

            if self.ekdv:
                psi += self.epsilon * B[np.newaxis,...]**3. * self.phi20 * self.c1**3.

        return psi

    def calc_buoyancy_l96(self, nonlinear=True):
        """
        Calculate the buoyancy perturbation: b = g*rho'
        """
        B = self.B
        B_xx = self.calc_Bxx()

        # Use the dimensional N2
        N2 = self.N2
        
        A = B * self.c1
        A_xx = B_xx * self.c1
        
        # Linear component
        b = A[np.newaxis,:]*self.phi_1*N2/self.c1[np.newaxis,:]
        
        ## Nonlinear components
        if nonlinear:
            #D10 = calc_D10(self.phi_1, self.c1, self.N2, self.dz_s)
            b += self.epsilon*A[np.newaxis,:]**2.*self.D10
        
            #D01 = calc_D01(self.phi_1, self.c1, self.N2, self.dz_s)
            b += self.mu*A_xx[np.newaxis,:]*self.D01

            if self.ekdv:
                b += self.epsilon*A[np.newaxis,...]**3.*self.D20
        
        return b.T # Need to return the dimensions

    def calc_buoyancy_h99(self, nonlinear=True):
        """
        Use the Holloway et al 99 version of the eqn's
        """
        dN2_dz = grad_z(self.N2, self.Z, axis=0)
        
        # Linear term
        b = self.B[np.newaxis,:] * self.phi_1 * self.N2
        
        #alpha = self.r10/(2*self.c1) ??
        #alpha = -2*self.c1*self.r10
        alpha = self.Alpha
        
        # nonlinear terms
        if nonlinear:
            b -= alpha/(2*self.c1)*self.B[np.newaxis,:]*self.phi_1*self.N2
            b -= 0.5*dN2_dz*self.B[np.newaxis,:]**2. * self.phi_1**2.
            # Cubic nonlinearity
            #b += self.c1*self.B[:,np.newaxis]**2. *self.N2 * self.T10
            
        return b.T
 

    def calc_density(self, nonlinear=True, method='l96'):
        """
        Returns density
        
        Method: 
            'h99' holloway 1999
            'l96' lamb 1996
        """
        b = self.calc_buoyancy_l96(nonlinear=nonlinear)
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

        attrs = {'long_name':'Cubic Nonlinearity',\
                'units':'m-3 s'}
                
        r20 = xray.DataArray(self.r20,
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

        phi20 = xray.DataArray(self.phi20,
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

        D20 = xray.DataArray(self.D20,
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
                'ekdv',\
                'nonlinear',\
                'nonhydrostatic',\
        ]

        attrs = {}
        for aa in saveattrs:
            myatt = getattr(self,aa)
            if isinstance(myatt,bool):
                attrs.update({aa: int(myatt)})
            else:
                attrs.update({aa:myatt})

        attrs.update({'Description':'1D variable-coefficient KdV Solution'})

        return xray.Dataset({'B':B,\
                        'Alpha':Alpha,\
                        'Beta':Beta,\
                        'Qterm':Qterm,\
                        'r20':r20,\
                        'h':h,\
                        'Cn':Cn,\
                        'X':X,\
                        'Z':Z,\
                        'rhoZ':rhoZ,\
                        'Phi':Phi,\
                        'phi01':phi01,\
                        'phi10':phi10,\
                        'phi20':phi20,\
                        'D01':D01,\
                        'D10':D10,\
                        'D20':D20,\
                        #'rhoz':rhoz,\
                        }, attrs=attrs)



