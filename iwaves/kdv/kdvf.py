"""
Variable coefficient code
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy import sparse 
import scipy.signal

#from .kdvimex import  KdVImEx as KdV
from .kdvcore import  KdVCore as KdV
from iwaves.utils import isw 
from iwaves.utils.tools import grad_z, quadinterp

import xarray as xray

import pdb

class KdVf(KdV):
    """
    KdV with friction
    """
    verbose = True
    print_freq = 5.
    
    def __init__(self, 
            m_star=0, # Bottom drag
            **kwargs):
	
        self.m_star = m_star

        # Initialise properties
        # (This is ugly but **kwargs are reserved for the superclass)

        self.__dict__.update(**kwargs)

        # Now initialise the class
        KdV.__init__(self, **kwargs)

    def calc_nonlinear_rhs(self, A):
        """
        AZ ATTEMPTING A PARENT FUNCTION OVERLOAD TO INCLUDE DRAG
        """
        
        rhs = KdV.calc_nonlinear_rhs(self, A)

        # print('Adding drag')

        h2 = self.beta/self.c
        cff = -self.m_star*self.c / h2
        rhs += cff * np.abs(A)*A

        return rhs
