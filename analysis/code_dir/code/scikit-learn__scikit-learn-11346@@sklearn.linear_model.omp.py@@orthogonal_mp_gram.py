import warnings
from math import sqrt
import numpy as np
from scipy import linalg
from scipy.linalg.lapack import get_lapack_funcs
from .base import LinearModel, _pre_fit
from ..base import RegressorMixin
from ..utils import as_float_array, check_array, check_X_y
from ..model_selection import check_cv
from ..externals.joblib import Parallel, delayed
premature = ' Orthogonal matching pursuit ended prematurely due to linear\ndependence in the dictionary. The requested precision might not have been met.\n'

def orthogonal_mp_gram(Gram, Xy, n_nonzero_coefs=None, tol=None, norms_squared=None, copy_Gram=True, copy_Xy=True, return_path=False, return_n_iter=False):
    Gram = check_array(Gram, order='F', copy=copy_Gram)
    Xy = np.asarray(Xy)
    if Xy.ndim > 1 and Xy.shape[1] > 1:
        copy_Gram = True
    if Xy.ndim == 1:
        Xy = Xy[:, np.newaxis]
        if tol is not None:
            norms_squared = [norms_squared]
    if copy_Xy or not Xy.flags.writeable:
        Xy = Xy.copy()
    if n_nonzero_coefs is None and tol is None:
        n_nonzero_coefs = int(0.1 * len(Gram))
    if tol is not None and norms_squared is None:
        raise ValueError('Gram OMP needs the precomputed norms in order to evaluate the error sum of squares.')
    if tol is not None and tol < 0:
        raise ValueError('Epsilon cannot be negative')
    if tol is None and n_nonzero_coefs <= 0:
        raise ValueError('The number of atoms must be positive')
    if tol is None and n_nonzero_coefs > len(Gram):
        raise ValueError('The number of atoms cannot be more than the number of features')
    if return_path:
        coef = np.zeros((len(Gram), Xy.shape[1], len(Gram)))
    else:
        coef = np.zeros((len(Gram), Xy.shape[1]))
    n_iters = []
    for k in range(Xy.shape[1]):
        out = _gram_omp(Gram, Xy[:, k], n_nonzero_coefs, norms_squared[k] if tol is not None else None, tol, copy_Gram=copy_Gram, copy_Xy=False, return_path=return_path)
        if return_path:
            _, idx, coefs, n_iter = out
            coef = coef[:, :, :len(idx)]
            for n_active, x in enumerate(coefs.T):
                coef[idx[:n_active + 1], k, n_active] = x[:n_active + 1]
        else:
            x, idx, n_iter = out
            coef[idx, k] = x
        n_iters.append(n_iter)
    if Xy.shape[1] == 1:
        n_iters = n_iters[0]
    if return_n_iter:
        return (np.squeeze(coef), n_iters)
    else:
        return np.squeeze(coef)