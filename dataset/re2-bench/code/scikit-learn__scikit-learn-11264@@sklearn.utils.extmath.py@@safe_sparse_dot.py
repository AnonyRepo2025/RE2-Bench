from __future__ import division
import warnings
import numpy as np
from scipy import linalg, sparse
from . import check_random_state, deprecated
from .fixes import np_version
from .fixes import logsumexp as scipy_logsumexp
from ._logistic_sigmoid import _log_logistic_sigmoid
from ..externals.six.moves import xrange
from .sparsefuncs_fast import csr_row_norms
from .validation import check_array



def safe_sparse_dot(a, b, dense_output=False):
    """Dot product that handle the sparse matrix case correctly

    Uses BLAS GEMM as replacement for numpy.dot where possible
    to avoid unnecessary copies.

    Parameters
    ----------
    a : array or sparse matrix
    b : array or sparse matrix
    dense_output : boolean, default False
        When False, either ``a`` or ``b`` being sparse will yield sparse
        output. When True, output will always be an array.

    Returns
    -------
    dot_product : array or sparse matrix
        sparse if ``a`` or ``b`` is sparse and ``dense_output=False``.
    """
    if sparse.issparse(a) or sparse.issparse(b):
        ret = a * b
        if dense_output and hasattr(ret, "toarray"):
            ret = ret.toarray()
        return ret
    else:
        return np.dot(a, b)
