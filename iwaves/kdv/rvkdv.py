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
        \frac{f^2}{2c}\left[ v(x,t) -\langle v \rangle \right]
        $$
        where
        $$v(x,t)=\int_{0}^{x} A(x',t)\ dx'$$
        and
        $$
        \langle v \rangle = \frac{1}{L}\int_{0}^{L} v(x,t)\ dx
        $$
        """
        rhs = vKdV.calc_nonlinear_rhs(self, A)

        cff = 0.5*self.cor_f*self.cor_f / self.c 

        #rhs -= np.cumsum(cff * A * self.dx)
        v = np.cumsum(A) * self.dx
        L = self.N *  self.dx
        v_bar = np.sum(v)/L
        rhs -= cff*(v-v_bar)

        return rhs