from functools import reduce
import numpy as np
from astropy import units as u
from astropy.utils import deprecated
from .angles import Angle



def is_O3(matrix, atol=None):
    """Check whether a matrix is in the length-preserving group O(3).

    Parameters
    ----------
    matrix : (..., N, N) array-like
        Must have attribute ``.shape`` and method ``.swapaxes()`` and not error
        when using `~numpy.isclose`.
    atol : float, optional
        The allowed absolute difference.
        If `None` it defaults to 1e-15 or 5 * epsilon of the matrix's dtype, if floating.

        .. versionadded:: 5.3

    Returns
    -------
    is_o3 : bool or array of bool
        If the matrix has more than two axes, the O(3) check is performed on
        slices along the last two axes -- (M, N, N) => (M, ) bool array.

    Notes
    -----
    The orthogonal group O(3) preserves lengths, but is not guaranteed to keep
    orientations. Rotations and reflections are in this group.
    For more information, see https://en.wikipedia.org/wiki/Orthogonal_group
    """
    # matrix is in O(3) (rotations, proper and improper).
    I = np.identity(matrix.shape[-1])
    if atol is None:
        if np.issubdtype(matrix.dtype, np.floating):
            atol = np.finfo(matrix.dtype).eps * 5
        else:
            atol = 1e-15

    is_o3 = np.all(
        np.isclose(matrix @ matrix.swapaxes(-2, -1), I, atol=atol), axis=(-2, -1)
    )

    return is_o3
