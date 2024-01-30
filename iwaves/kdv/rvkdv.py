import numpy as np
from .vkdv import  vKdV

class rvKdV(vKdV):
    def __init__(self, rhoz, z, h, x, mode, cor_f=0, **kwargs):
        """
        Variable-coefficient KdV with rotation
        """
        self.cor_f = cor_f
        vKdV.__init__(self, rhoz, z, h, x, mode, **kwargs)

    def calc_nonlinear_rhs(self, A):
        """
        Calculate the nonlinear steepening term vectors

        RHS of the rotation equation is:
        $$
        \int_{-\infty}^{x}\frac{f^2}{2c}A\ dx'
        $$
        """
        rhs = vKdV.calc_nonlinear_rhs(self, A)

        cff = 0.5*self.cor_f*self.cor_f / self.c 

        rhs -= np.cumsum(cff * A * self.dx)

        return rhs