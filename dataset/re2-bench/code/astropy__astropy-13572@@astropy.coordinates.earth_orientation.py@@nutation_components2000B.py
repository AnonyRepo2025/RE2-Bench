import numpy as np
import erfa
from astropy.time import Time
from .builtin_frames.utils import get_jd12
from .matrix_utilities import rotation_matrix, matrix_product, matrix_transpose

jd1950 = Time('B1950').jd
jd2000 = Time('J2000').jd

def nutation_components2000B(jd):
    """
    Computes nutation components following the IAU 2000B specification

    Parameters
    ----------
    jd : scalar
        Julian date (TT) at which to compute the nutation components

    Returns
    -------
    eps : float
        epsilon in radians
    dpsi : float
        dpsi in radians
    deps : float
        depsilon in raidans
    """
    dpsi, deps, epsa, _, _, _, _, _ = erfa.pn00b(jd, 0)
    return epsa, dpsi, deps
