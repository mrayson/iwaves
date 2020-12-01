
# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from iwaves.kdv.kdvcore import  KdVCore as KdV

# In[2]:

##########
# Inputs
a0 = -25.
nsteps = 19000
N = 2000
#ones = np.ones((N,))
ones = 1.

kdvargs = dict(
   N=N,
   c=1.5*ones,
   alpha=0.01*ones,
   beta=8000.*ones,
   dx=50.,
   dt=10.,
   spongedist = 5e3,
   spongetime = 60.,
)

h = ics.depth_tanh2(bathy_params, x) # CHANGED

# Initialise the class
mykdv = KdV(**kdvargs)
print(mykdv.L_rhs.todense()[0:4,0:8])
#print(mykdv.L_lhs.todense())
#print(mykdv.L_rhs.shape, mykdv.B.shape)
#print(mykdv.L_beta.todense()[0:4,0:8])

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
plt.plot(mykdv.B_n_p1)
plt.title(nsteps)
plt.show()

