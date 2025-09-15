import scipy.sparse as sp
import numpy as np
from .sparsefuncs_fast import (
    csr_mean_variance_axis0 as _csr_mean_var_axis0,
    csc_mean_variance_axis0 as _csc_mean_var_axis0,
    incr_mean_variance_axis0 as _incr_mean_var_axis0)



def _minor_reduce(X, ufunc):
    major_index = np.flatnonzero(np.diff(X.indptr))

    # reduceat tries casts X.indptr to intp, which errors
    # if it is int64 on a 32 bit system.
    # Reinitializing prevents this where possible, see #13737
    X = type(X)((X.data, X.indices, X.indptr), shape=X.shape)
    value = ufunc.reduceat(X.data, X.indptr[major_index])
    return major_index, value
