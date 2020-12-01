"""


"""

import numpy as np
import matplotlib.pyplot as plt
from iwaves.kdv.vkdv import  vKdV as KdV

def rho_double_tanh(beta, z):
    """
    Double hyperbolic tangent 
    """
    return beta[0] - beta[1]*(np.tanh((z+beta[2])/beta[3])
        + np.tanh((z+beta[4])/beta[5]))


def depth_tanh(beta, x):
    """
    Hyperbolic tangent shelf break

    H - total depth
    h0 - shelf height
    x0 - shelf break x location
    lt - shelf break width
    """
    
    H, h0, x0, lt = beta

    return H-0.5*h0*(1+np.tanh((x-x0)/(0.5*lt)))
   

##########
# Inputs
a0 = 1.
nsteps = 12000

rho_params =[1023.68,
     1.22,
     156.7,
     53.6,
     73.1,
     40.2] # 1st April 2017

H = 400
h0 = 225
Nz = 100
bathy_params = [H, h0, 90e3, 70e3]       


dt = 10.

N = 6000
dx = 25.
# L_d = N*dx

mode = 0
###

kdvargs = dict(
   N=N,
   dx=dx,
   dt=dt,
   spongedist = 5e3,
   spongetime = 60.,
   Nsubset = 10,
   nonhydrostatic=0.,
   nonlinear=0.,
)

z = np.linspace(-H,0,Nz)
x = np.arange(0, N*dx, dx)
rhoz = rho_double_tanh(rho_params,z)
h = depth_tanh(bathy_params, x) 

#plt.figure()
#plt.subplot(121)
#plt.plot( rhoz, z)
#
#plt.subplot(122)
#plt.plot(x, -h)
#plt.show()

## Initialise the class
mykdv = KdV(rhoz, z, h, x, mode, **kdvargs)

#plt.figure()
#plt.subplot(411)
#plt.plot( mykdv.x, mykdv.alpha)
#plt.subplot(412)
#plt.plot( mykdv.x, mykdv.beta)
#plt.subplot(413)
#plt.plot( mykdv.x, mykdv.c)
#plt.subplot(414)
#plt.plot( mykdv.x, -mykdv.h)
#plt.show()


##print(mykdv.L_rhs.todense()[0:4,0:8])
##print(mykdv.L_lhs.todense())
##print(mykdv.L_rhs.shape, mykdv.B.shape)
##print(mykdv.L_beta.todense()[0:4,0:8])

def bcfunc2(a0,t):
    t0 = 3*3600
    T = 1*3600
    return a0*np.exp(-((t - t0)/T)**2.)

def bcfunc(a0,t):
    T = 6*3600
    omega = 2*np.pi/T
    return a0*np.sin(omega*t)

for ii in range(nsteps):
    if mykdv.solve_step(bc_left=bcfunc(a0,mykdv.t)) != 0:
        print('Blowing up at step: %d'%ii)
        break

plt.figure()
plt.subplot(211)
plt.plot(mykdv.x, mykdv.B_n_p1)
plt.title(nsteps)
plt.plot(mykdv.x, np.abs(a0)*mykdv.Qterm[0]/mykdv.Qterm)
#plt.plot(mykdv.x, np.abs(a0)*mykdv.Qterm)
plt.grid(b=True)
plt.subplot(212)
plt.plot(mykdv.x, -mykdv.h)

plt.show()
#
#

