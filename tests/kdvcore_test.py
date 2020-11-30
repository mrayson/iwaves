
# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from iwaves.kdv.kdvcore import  KdVCore as KdV

# In[2]:

##########
# Inputs
a0 = -25.
nsteps = 4560

kdvargs = dict(
   N=2000,
   c=1.5,
   alpha=0.01,
   beta=8000.,
   dx=50.,
   dt=10.,
)
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
    if mykdv.solve_step(bc_left=bcfunc2(a0,mykdv.t)) != 0:
        print('Blowing up at step: %d'%ii)
        break


plt.figure()
plt.plot(mykdv.B_n_p1)
plt.title(nsteps)
plt.show()

