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

premature = """ Orthogonal matching pursuit ended prematurely due to linear
dependence in the dictionary. The requested precision might not have been met.
"""

def _gram_omp(Gram, Xy, n_nonzero_coefs, tol_0=None, tol=None,
              copy_Gram=True, copy_Xy=True, return_path=False):
    """Orthogonal Matching Pursuit step on a precomputed Gram matrix.

    This function uses the Cholesky decomposition method.

    Parameters
    ----------
    Gram : array, shape (n_features, n_features)
        Gram matrix of the input data matrix

    Xy : array, shape (n_features,)
        Input targets

    n_nonzero_coefs : int
        Targeted number of non-zero elements

    tol_0 : float
        Squared norm of y, required if tol is not None.

    tol : float
        Targeted squared error, if not None overrides n_nonzero_coefs.

    copy_Gram : bool, optional
        Whether the gram matrix must be copied by the algorithm. A false
        value is only helpful if it is already Fortran-ordered, otherwise a
        copy is made anyway.

    copy_Xy : bool, optional
        Whether the covariance vector Xy must be copied by the algorithm.
        If False, it may be overwritten.

    return_path : bool, optional. Default: False
        Whether to return every value of the nonzero coefficients along the
        forward path. Useful for cross-validation.

    Returns
    -------
    gamma : array, shape (n_nonzero_coefs,)
        Non-zero elements of the solution

    idx : array, shape (n_nonzero_coefs,)
        Indices of the positions of the elements in gamma within the solution
        vector

    coefs : array, shape (n_features, n_nonzero_coefs)
        The first k values of column k correspond to the coefficient value
        for the active features at that step. The lower left triangle contains
        garbage. Only returned if ``return_path=True``.

    n_active : int
        Number of active features at convergence.
    """
    Gram = Gram.copy('F') if copy_Gram else np.asfortranarray(Gram)

    if copy_Xy or not Xy.flags.writeable:
        Xy = Xy.copy()

    min_float = np.finfo(Gram.dtype).eps
    nrm2, swap = linalg.get_blas_funcs(('nrm2', 'swap'), (Gram,))
    potrs, = get_lapack_funcs(('potrs',), (Gram,))

    indices = np.arange(len(Gram))  # keeping track of swapping
    alpha = Xy
    tol_curr = tol_0
    delta = 0
    gamma = np.empty(0)
    n_active = 0

    max_features = len(Gram) if tol is not None else n_nonzero_coefs

    L = np.empty((max_features, max_features), dtype=Gram.dtype)

    L[0, 0] = 1.
    if return_path:
        coefs = np.empty_like(L)

    while True:
        lam = np.argmax(np.abs(alpha))
        if lam < n_active or alpha[lam] ** 2 < min_float:
            # selected same atom twice, or inner product too small
            warnings.warn(premature, RuntimeWarning, stacklevel=3)
            break
        if n_active > 0:
            L[n_active, :n_active] = Gram[lam, :n_active]
            linalg.solve_triangular(L[:n_active, :n_active],
                                    L[n_active, :n_active],
                                    trans=0, lower=1,
                                    overwrite_b=True,
                                    check_finite=False)
            v = nrm2(L[n_active, :n_active]) ** 2
            Lkk = Gram[lam, lam] - v
            if Lkk <= min_float:  # selected atoms are dependent
                warnings.warn(premature, RuntimeWarning, stacklevel=3)
                break
            L[n_active, n_active] = sqrt(Lkk)
        else:
            L[0, 0] = sqrt(Gram[lam, lam])

        Gram[n_active], Gram[lam] = swap(Gram[n_active], Gram[lam])
        Gram.T[n_active], Gram.T[lam] = swap(Gram.T[n_active], Gram.T[lam])
        indices[n_active], indices[lam] = indices[lam], indices[n_active]
        Xy[n_active], Xy[lam] = Xy[lam], Xy[n_active]
        n_active += 1
        # solves LL'x = X'y as a composition of two triangular systems
        gamma, _ = potrs(L[:n_active, :n_active], Xy[:n_active], lower=True,
                         overwrite_b=False)
        if return_path:
            coefs[:n_active, n_active - 1] = gamma
        beta = np.dot(Gram[:, :n_active], gamma)
        alpha = Xy - beta
        if tol is not None:
            tol_curr += delta
            delta = np.inner(gamma, beta[:n_active])
            tol_curr -= delta
            if abs(tol_curr) <= tol:
                break
        elif n_active == max_features:
            break

    if return_path:
        return gamma, indices[:n_active], coefs[:, :n_active], n_active
    else:
        return gamma, indices[:n_active], n_active
