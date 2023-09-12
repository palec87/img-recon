#!/usr/bin/env python

"""
Mostly helper functions to reproduce the paper
"""

import numpy as np
from imrec.phantom_3d import phantom3d
from hadamard.fwht import run_exp


def z2exp(x):
    """Converts complex number to exponential form

    Args:
        x (numeric): complex number

    Returns:
        tuple: modulo and phase of the z
    """
    mod = abs(x)
    phase = np.angle(x)
    return (mod, phase)


def exp2z(x):
    """Converts exponential form to complex number

    Args:
        x (tuple): modulo and phase

    Returns:
        numeric: complex number
    """
    return x[0]*np.exp(1j*x[1])


if __name__ == '__main__':
    print('Testing autocorrelation OPT retrieval')