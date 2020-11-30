
# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import iwaves.utils.isw as kdv
#from iwaves.kdv.kdv import  KdV
from iwaves.kdv.kdvimex import  KdVImEx as KdV

from iwaves.kdv.vkdv import  vKdV as KdV

import pdb


# In[2]:


###########
# Stratification and wave functions
def ideal_rho(z, drho, dp, L):
    #return drho/2 - drho/2*np.tanh(dp + dp*z/L )
    return drho/2 * (1 - np.tanh(dp + dp*z/L ) )

def lamb_rho(z):
    Z = z+300. # Height above the bottom
    return 1027.31 - 3.3955*np.exp((Z-300)/50.0 )

def lamb_drho(z):
    Z = z+300. # Height above the bottom
    return  - 3.3955/50.*np.exp((Z-300)/50.0 )


def lamb_wave(x, a0,  Lw, x0=0.):
    xl = -20000.
    dr = Lw/2
    dl = Lw/2
    return -a0*0.25*(1+np.tanh( (x-xl)/dl))        *(1-np.tanh( (x)/dr) )

def zeroic(x, a0, Lw, x0=0.):
    return 0*x


def double_tanh(beta, z):
    return beta[0] - beta[1]*(np.tanh((z+beta[2])/beta[3])
        + np.tanh((z+beta[4])/beta[5]))


# In[3]:


##########
# Inputs
H = 250.
Nz = 50
#Nx = 16667 # 18 and 9 m
Nx = 12500
z = np.linspace(0, -H, Nz)
# Density parameters
betas = [1023.7, 1.12, 105, 52, 155, 43] # ~April 5
#betas = [1023.5, 1.22, 67, 55, 157, 52] # ~March 1



mode = 0
a0 = 30.

nondim = False
nonlinear = True
nonhydrostatic= True
ekdv=False
timesolver = 'AB2'
spongedist=1e4
nu_H = 1.0


#Cmax=0.02 # Explicit solver
Cmax = 5 # IMEX solver
dt = 10 # None

nsteps = 25000
#runtime = 86400.
#runtime = 200.
#runtime = 1.
#runtime = 5360*5
#runtime = 8000*0.1
###########


# In[4]:

#rhoz = lamb_rho(z)
rhoz = double_tanh(betas, z)



# In[5]:


#### Intitialise the class
if nondim:
    runtime = runtime/Lw

# vkdv
x = np.linspace(0,1.5e5,Nx//2)
h = H*np.ones_like(x) - 0.002*x

mykdv = KdV(rhoz, z, \
        h, x,\
        Nsubset=20,\
        mode=mode,\
        Cmax=Cmax,\
        Nx=Nx,\
        nondim=nondim,\
        nonlinear=nonlinear,\
        nonhydrostatic=nonhydrostatic,\
        ekdv=ekdv,\
        wavefunc=zeroic,\
        timesolver=timesolver,\
        spongedist=spongedist,\
        nu_H = nu_H,\
        dt=dt)

mykdv.print_params()



# In[7]:


#####
## Plot the density profile
#plt.figure(figsize=(6,8))
#ax=plt.subplot(111)
#plt.plot( mykdv.rhoz, mykdv.z)
#plt.xlabel(r'$\rho(z)$')
#
#ax2=ax.twiny()
#ax2.plot(mykdv.N2 , mykdv.z,'k')
#plt.xlabel('$N^2(z)$')
#
#plt.show()

##########
# run the model
#nsteps = int(runtime//mykdv.dt_s)

print( nsteps, mykdv.dt_s)

omega = 2*np.pi/(12.42*3600)
def bcfunc(a0,t):
    return a0*np.sin(omega*t)



for ii in range(nsteps):
    point = nsteps/100
    if(ii % (5 * point) == 0):
        print('%3.1f %% complete...'%(float(ii)/nsteps*100))
    if mykdv.solve_step(bc_left=bcfunc(a0,mykdv.t)) != 0:
        print('Blowing up at step: %d'%ii)
        break


# In[13]:


# Calculate the velocity and density fields
rho = mykdv.calc_density(nonlinear=True)
u,w = mykdv.calc_velocity(nonlinear=True)
#psi = mykdv.calc_streamfunction()


# In[15]:


## Comparison with Fig 8 in Lamb and Yan, 1996

xoffset =0
########
# Plot the amplitude function
plt.figure(figsize=(12,6))
ax1 = plt.subplot(211)
plt.plot(mykdv.x - xoffset, mykdv.B, color='k')
#plt.plot(mykdv.x - mykdv.c1*runtime, mykdv.B)
#plt.plot(mykdv2.x, mykdv2.B, 'r')
#plt.xlim(-0.8e4, .8e4)
#plt.xticks(np.arange(-8000,9000,1000))
#plt.ylim(-0.01*a0, 0.01*a0)
#plt.xlim(4*Lw, 8*Lw)
plt.ylabel('B(m)')


ax2 = plt.subplot(212,sharex=ax1)
plt.plot(mykdv.x - xoffset, u[:,0], color='k')
plt.ylabel('u [m/s]')
#plt.xlim(-0.8e4, 0.8e4)
#plt.xlim(-1.5e5,-1.499e5)
plt.ylim(-0.5,0.5)

plt.show()

"""
# In[16]:


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
plt.xlim(-1.1e4, 0.8e4)

plt.subplot(212)
plt.pcolormesh(mykdv.x-xoffset, mykdv.z, w.T, 
     cmap='RdBu')
cb=plt.colorbar(orientation='horizontal')
cb.ax.set_title('w [m/s]')
plt.contour(mykdv.x-xoffset, mykdv.z, rho.T, np.arange(20.,30.,0.5),        colors='k', linewidths=0.5)

#plt.xlim(-0.8e4, 0.8e4)
plt.xlim(-1.1e4, 0.8e4)

plt.tight_layout()





"""
