"""
Test dedalus eigenvalue solver
"""


#import dedalus.public as de
from dedalus import public as de
import numpy as np
import matplotlib.pyplot as plt
import logging
import time
logger = logging.getLogger(__name__)


# Domain
Nx = 20
x_basis = de.Chebyshev('x', Nx, interval=(-200, 0))
domain = de.Domain([x_basis], np.float64)

# Non-constant (variable) coefficent
ncc = domain.new_field(name='N2')
ncc['g'] = 0.0001 
#ncc.meta['x', 'y']['constant'] = True

# Problem
problem = de.EVP(domain, variables=['u', 'u_x'],eigenvalue='c2')
problem.meta[:]['x']['dirichlet'] = True
problem.parameters['N2'] = ncc
problem.add_equation("N2/c2*u + dx(u_x) = 0")
problem.add_equation("u_x - dx(u) = 0")
problem.add_bc("left(u) = 0")
problem.add_bc("right(u) = 0")

# Solver
solver = problem.build_solver()
t1 = time.time()
solver.solve_dense(solver.pencils[0])
t2 = time.time()
logger.info('Elapsed solve time: %f' %(t2-t1))

# Filter infinite/nan eigenmodes
finite = np.isfinite(solver.eigenvalues)
solver.eigenvalues = solver.eigenvalues[finite]
solver.eigenvectors = solver.eigenvectors[:, finite]

# Sort eigenmodes by eigenvalue
order = np.argsort(solver.eigenvalues)
solver.eigenvalues = solver.eigenvalues[order]
solver.eigenvectors = solver.eigenvectors[:, order]

# Phase speed
cn = 1/np.sqrt(solver.eigenvalues)
print(Nx,cn[0])

# This is how to get the eigvals/vecs on a grid
z = domain.grid(0)
solver.set_state(1) # 0'th mode
phi = solver.state['u']['g'].real

plt.figure()
plt.plot(phi,z,'.')
plt.show()
## Plot error vs exact eigenvalues
#mode_number = 1 + np.arange(len(solver.eigenvalues))
#exact_eigenvalues = mode_number**2 * np.pi**2 / 4
#eval_relative_error = (solver.eigenvalues - exact_eigenvalues) / exact_eigenvalues


