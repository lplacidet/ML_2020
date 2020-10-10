# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    phi = np.ones((x.shape[0],degree+1))
    
    for i in range(degree+1):
        power_column = np.power(x,i)
        phi[:,i] = power_column
    return phi
