

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy import sparse 
from scipy.sparse import linalg

import matplotlib.pyplot as plt
import kdvlamb as kdv
#from kdvimex import kdv, iwaves # Imports the Lamb functions
#from kdvgrimshaw import  KdV
from kdvimex import  KdVImEx as KdV
from isw import lamb_tanh_rho, sine

import pdb

####
# Functions for 2D array
def grad_z(y, z, axis=0):
    """
    Compute the vertical gradient

    "z" can be an array same size as y, or vector along the first axis of "y"

    Takes the derivative along the dimension specified by axis(=0)
    """
    Nz = z.shape[0]

    # Reshape the y variable
    y = y.swapaxes(0, axis)
    assert y.shape[0] == Nz

    z = z.swapaxes(0, axis)
    assert z.shape == (Nz,) or z.shape == y.shape

    dy_dz = np.zeros_like(y)
    
    # Second-order accurate for mid-points
    ymid = 0.5*(y[1:,...]+y[0:-1,...])

    zmid = 0.5*(z[1:,...]+z[0:-1,...])

    dzmid  = zmid[1:,...] - zmid[0:-1,...] 

    dy_dz[1:-1, ...] = (ymid[1:,...] - ymid[0:-1,...])/\
            dzmid[:,...]

    # First-order accurate for top and bottom cells
    dy_dz[0,...] = (y[1,...] - y[0,...])/dzmid[0,...]
    dy_dz[-1,...] = (y[-1,...] - y[-2,...])/dzmid[-1,...]

    return dy_dz.swapaxes(axis, 0)

def iwave_modes_sparse(N2, dz, d):
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
    # Dense
    #A = np.diag(-1*dz2*np.ones((nz-1)),-1) + \
    #    np.diag(2*dz2*np.ones((nz,)),0) + \
    #    np.diag(-1*dz2*np.ones((nz-1)),1)
    # Sparse
    diags = np.ones((3,nz))
    diags[0,:] *= -1*dz2
    diags[1,:] *= 2*dz2
    diags[2,:] *= -1*dz2
    
    # BCs
    #diags[0,0] = 0.#??
    diags[1,0] = -1.
    diags[2,0] = 0.
    diags[1,-1] = -1.
    diags[0,-1] = 0.
    #diags[2,-1] = 0.#??
    A = sparse.spdiags(diags, [-1,0,1], nz, nz)

    ## BC's
    #eps = 1e-10
    #A[0,0] = -1.
    #A[0,1] = 0.
    #A[-1,-1] = -1.
    #A[-1,-2] = 0.

    # Construct the RHS matrix i.e. put N^2 along diagonals
    #B = np.diag(N2,0)
    B = sparse.spdiags(N2,[0], nz, nz)

    # Solve... (use scipy not numpy)
    w, phi = linalg.eigs(A, M=B, k=nz-2 )	

    c = 1. / np.power(w, 0.5) # since term is ... + N^2/c^2 \phi

    # Sort by the eigenvalues
    idx = np.argsort(c)[::-1] # descending order

    # Calculate the actual phase speed
    cn = np.real( c[idx] )

    return np.real(phi[:,idx]), cn


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


###########
# Stratification and wave functions
def ideal_rho(z, drho, dp, L):
    return drho/2 - drho/2*np.tanh(dp + dp*z/L )
    #return drho/2 * (1 - np.tanh(dp + dp*z/L ) )


GRAV=9.81
RHO0=1000.

class KdVGrimshaw(KdV):
    
    def __init__(self, rhoz, z, h, x, wavefunc=sine, **kwargs):
	
	Nz = z.shape[0]
	Nx = x.shape[0]
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
	    phi, cn = kdv.iwave_modes(self.N2[:,ii], self.dZ[ii], h[ii])

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
	KdV.__init__(self, rhoz, z, wavefunc=wavefunc, x=x, **kwargs)

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

	phi01 = np.zeros((Nz, Nx))
	phi10 = np.zeros((Nz, Nx))

        print 'Calculating nonlinear structure functions...'
	for ii in range(self.Nx):
	    point = self.Nx/100
	    if(ii % (5 * point) == 0):
		print '%3.1f %% complete...'%(float(ii)/self.Nx*100)

	    rhs01 = kdv.calc_phi01_rhs(self.Phi[:,ii], \
	    	self.c1[ii], self.N2[:,ii], self.dZ[ii])
	    phi01[:,ii] = kdv.solve_phi_bvp(rhs01,\
	    	self.N2[:,ii], self.c1[ii], self.dZ[ii])
	    rhs10 = kdv.calc_phi10_rhs(self.Phi[:,ii],\
	    	self.c1[ii], self.N2[:,ii], self.dZ[ii])
	    phi10[:,ii] = kdv.solve_phi_bvp(rhs10, \
	    	self.N2[:,ii], self.c1[ii], self.dZ[ii])

        return phi01, phi10

    def calc_coeffs(self):
    	return -self.Beta, -self.Alpha, None

    def calc_buoyancy_coeffs(self):

	D01 = np.zeros((Nz, Nx))
	D10 = np.zeros((Nz, Nx))

        for ii in range(self.Nx):
            D01[:,ii] = kdv.calc_D01(self.Phi[:,ii], self.c1[ii],\
                self.N2[:,ii], self.dZ[ii])
            D10[:,ii] = kdv.calc_D10(self.Phi[:,ii], self.c1[ii],\
                self.N2[:,ii], self.dZ[ii])

        return D01, D10

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
 
 

#
## Compute nonlinear and dispersion constants
#self.r01 = calc_r01(self.phi_1, self.c1, self.dz_s)
#self.r10 = calc_r10(self.phi_1, self.c1, self.N2, self.dz_s)

# In[3]:

##########
# Inputs
H = 250
Nz = 50
#Nx = 16667 # 18 and 9 m
Nx = 6000
z = np.linspace(0, -H, Nz)
#rhoz = ideal_rho(z, 3.1, 3.2, 160.) + 20.6 # Winter
#rhoz = ideal_rho(z, 3.5, 1.8, 150.) + 20.4 # Summer
#rhoz = lamb_rho(z)

# Horizontal domain
L_d = 1.5e5
x = np.linspace(0, L_d, Nx)

# Create some topography
slope = 5e-4
h = H - slope*x

mode = 0
a0 = 40.
Lw = 40000.

#tstart = '20080401.000000'
#tstart = '20080314.000000'
tstart = '20080821.000000'

nonlinear = True
nonhydrostatic= True

#Cmax=0.04 # Explicit solver
Cmax = 0.8 # IMEX solver
dt = None
runtime = 86400. * 0.75
#runtime = 1.
#runtime = 5360*5
#runtime = 8000*0.1
###########

#########
# Generate a density profile

# Load the parameters
strat_param_file = 'DATA/bestfit_rho_params.csv'
params = pd.read_csv(strat_param_file,index_col='time', parse_dates=True)
pp = params[tstart[0:8]:'20100101']

rhoz = lamb_tanh_rho(z, pp['rho0'][0], pp['dp'][0], pp['z1'][0], pp['h1'][0])


# In[4]:

###
# Intitialise the class

mykdv = KdVGrimshaw(rhoz, z, \
	h, x=x,\
	mode=mode,\
	a0=a0,\
	Lw=Lw,\
	x0=Lw,\
	Cmax=Cmax,\
        nonlinear=nonlinear,\
        nonhydrostatic=nonhydrostatic,\
        #timesolver=timesolver,\
        dt=dt)
#
#mykdv.print_params()
#
#print 'Lamb values:\n r01 = -2.91e3\n r10 = 8.31e-3\n r20 = 2.35e-5'
##########
# run the model
nsteps = int(runtime//mykdv.dt_s)

print nsteps, mykdv.dt_s

print 'Solving KdV equations for %d steps...'%nsteps
for ii in range(nsteps):
    point = nsteps/100
    if(ii % (5 * point) == 0):
        print '%3.1f %% complete...'%(float(ii)/nsteps*100)
    if mykdv.solve_step() != 0:
        print 'Blowing up at step: %d'%ii
        break


psi = mykdv.calc_streamfunction()
u,w = mykdv.calc_velocity()
rho = mykdv.calc_density(nonlinear=True)


clims = [-0.4,0.4]

plt.figure(figsize=(12,12))
plt.subplot(111)
plt.pcolormesh(mykdv.X, mykdv.Z, u, \
    vmin=clims[0], vmax=clims[1], cmap='RdBu')
cb=plt.colorbar(orientation='horizontal')
cb.ax.set_title('u [m/s]')
plt.contour(mykdv.X, mykdv.Z, rho, np.arange(20.,30.,0.25), \
        colors='k', linewidths=0.5)




########
# Plot the amplitude function
plt.figure(figsize=(12,6))
ax1 = plt.subplot(211)
plt.plot(mykdv.x , mykdv.B, color='k')
#plt.plot(mykdv.x - mykdv.c1*runtime, mykdv.B)
#plt.plot(mykdv2.x, mykdv2.B, 'r')
#plt.xlim(-1.1e4, 0.7e4)
plt.ylim(-50, 0)
#plt.xlim(8e4, 10e4)
#plt.xlim(4*Lw, 8*Lw)
plt.ylabel('B(m)')

ax2 = plt.subplot(212)
plt.pcolormesh(mykdv.X, mykdv.Z, psi, cmap='RdBu')
plt.colorbar()

plt.show()

"""

#ax2 = plt.subplot(212,sharex=ax1)
#plt.plot(mykdv.x - xoffset, u[:,0], color='k')
#plt.ylabel('u [m/s]')
#plt.xlim(-1.1e4, 0.7e4)
#plt.ylim(0,0.5)


# In[6]:

#####
# Plot the density profile
plt.figure(figsize=(6,8))
ax=plt.subplot(111)
plt.plot( mykdv.rhoz, mykdv.z)
plt.xlabel(r'$\rho(z)$')

ax2=ax.twiny()
ax2.plot(mykdv.N2 , mykdv.z,'k')
plt.xlabel('$N^2(z)$')


# In[7]:

#######
# Plot the eigenfunction (Fig 4 in LY96)
dphi = np.gradient(mykdv.phi_1, -mykdv.dz_s)

plt.figure(figsize=(6,8))
ax=plt.subplot(111)
plt.plot( mykdv.phi_1, mykdv.z,'0.5')
plt.xlabel(r'$\phi_1(z)$')

ax2=ax.twiny()
ax2.plot(dphi , mykdv.z,'k')
ax2.plot([0,0],[-H,0],'k--')
plt.xlabel('$ d \phi_1 / dz$')


# In[8]:

## Calculate nonlinear structure functions phi01, phi10
phi01, phi10 = mykdv.calc_nonlinstructure()

D01 = kdv.calc_D01(mykdv.phi_1, mykdv.c1, mykdv.N2, mykdv.dz_s)/mykdv.N2
D10 = kdv.calc_D10(mykdv.phi_1, mykdv.c1, mykdv.N2, mykdv.dz_s)/mykdv.N2

plt.figure(figsize=(16,8))
ax = plt.subplot(221)
plt.plot(mykdv.z, phi10,'k--')
plt.plot([-H,0],[0,0],'k:')
plt.ylabel('$\phi^{1,0}$')
plt.ylim(-0.01, 0.002)

ax = plt.subplot(222)
plt.plot(mykdv.z, D10,'k--')
plt.plot([-H,0],[0,0],'k:')
plt.ylabel('$D^{1,0}$')
plt.ylim(-0.02, 0.005)

ax = plt.subplot(223)
plt.plot(mykdv.z, phi01,'k--')
plt.plot([-H,0],[0,0],'k:')
plt.ylabel('$\phi^{0,1}$')
plt.ylim(-8000,6000)

ax = plt.subplot(224)
plt.plot(mykdv.z, D01,'k--')
plt.plot([-H,0],[0,0],'k:')
plt.ylabel('$D^{0,1}$')
plt.ylim(-1.2e4, 4e3)


#plt.show()
# In[9]:

##########
# run the model
nsteps = int(runtime//mykdv.dt_s)

print nsteps, mykdv.dt_s

for ii in range(nsteps):
    point = nsteps/100
    if(ii % (5 * point) == 0):
        print '%3.1f %% complete...'%(float(ii)/nsteps*100)
    if mykdv.solve_step() != 0:
        print 'Blowing up at step: %d'%ii
        break


# In[10]:

# Calculate the velocity and density fields
rho = mykdv.calc_density(nonlinear=True)
u,w = mykdv.calc_velocity(nonlinear=True)
#psi = mykdv.calc_streamfunction()


# In[11]:

xoffset = U0*runtime
########
# Plot the amplitude function
plt.figure(figsize=(12,6))
ax1 = plt.subplot(211)
plt.plot(mykdv.x - xoffset, mykdv.B, color='k')
#plt.plot(mykdv.x - mykdv.c1*runtime, mykdv.B)
#plt.plot(mykdv2.x, mykdv2.B, 'r')
plt.xlim(-1.1e4, 0.7e4)
plt.ylim(-50, 0)
#plt.xlim(8e4, 10e4)
#plt.xlim(4*Lw, 8*Lw)
plt.ylabel('B(m)')


ax2 = plt.subplot(212,sharex=ax1)
plt.plot(mykdv.x - xoffset, u[:,0], color='k')
plt.ylabel('u [m/s]')
plt.xlim(-1.1e4, 0.7e4)
plt.ylim(0,0.5)


# In[12]:

### Plot the density
clims = [-0.4,0.4]

plt.figure(figsize=(12,12))
plt.subplot(211)
plt.pcolormesh(mykdv.x-xoffset, mykdv.z, u.T, 
    vmin=clims[0], vmax=clims[1], cmap='RdBu')
cb=plt.colorbar(orientation='horizontal')
cb.ax.set_title('u [m/s]')
plt.contour(mykdv.x-xoffset, mykdv.z, rho.T, np.arange(20.,30.,0.5),        colors='k', linewidths=0.5)

#plt.xlim(-0.8e4, 0.8e4)
plt.xlim(-1.1e4, 0.7e4)

plt.subplot(212)
plt.pcolormesh(mykdv.x-xoffset, mykdv.z, w.T, 
     cmap='RdBu')
cb=plt.colorbar(orientation='horizontal')
cb.ax.set_title('w [m/s]')
plt.contour(mykdv.x-xoffset, mykdv.z, rho.T, np.arange(20.,30.,0.5),        colors='k', linewidths=0.5)

#plt.xlim(-0.8e4, 0.8e4)
plt.xlim(-1.1e4, 0.7e4)

plt.tight_layout()


# In[ ]:

plt.show()


"""
