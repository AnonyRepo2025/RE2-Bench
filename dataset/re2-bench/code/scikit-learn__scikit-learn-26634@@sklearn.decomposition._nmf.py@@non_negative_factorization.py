import itertools
import time
import warnings
from abc import ABC
from math import sqrt
from numbers import Integral, Real
import numpy as np
import scipy.sparse as sp
from scipy import linalg
from .._config import config_context
from ..base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    _fit_context,
)
from ..exceptions import ConvergenceWarning
from ..utils import check_array, check_random_state, gen_batches, metadata_routing
from ..utils._param_validation import (
    Hidden,
    Interval,
    StrOptions,
    validate_params,
)
from ..utils.extmath import randomized_svd, safe_sparse_dot, squared_norm
from ..utils.validation import (
    check_is_fitted,
    check_non_negative,
)
from ._cdnmf_fast import _update_cdnmf_fast

EPSILON = np.finfo(np.float32).eps

def non_negative_factorization(
    X,
    W=None,
    H=None,
    n_components="warn",
    *,
    init=None,
    update_H=True,
    solver="cd",
    beta_loss="frobenius",
    tol=1e-4,
    max_iter=200,
    alpha_W=0.0,
    alpha_H="same",
    l1_ratio=0.0,
    random_state=None,
    verbose=0,
    shuffle=False,
):
    """Compute Non-negative Matrix Factorization (NMF).

    Find two non-negative matrices (W, H) whose product approximates the non-
    negative matrix X. This factorization can be used for example for
    dimensionality reduction, source separation or topic extraction.

    The objective function is:

        .. math::

            L(W, H) &= 0.5 * ||X - WH||_{loss}^2

            &+ alpha\\_W * l1\\_ratio * n\\_features * ||vec(W)||_1

            &+ alpha\\_H * l1\\_ratio * n\\_samples * ||vec(H)||_1

            &+ 0.5 * alpha\\_W * (1 - l1\\_ratio) * n\\_features * ||W||_{Fro}^2

            &+ 0.5 * alpha\\_H * (1 - l1\\_ratio) * n\\_samples * ||H||_{Fro}^2

    Where:

    :math:`||A||_{Fro}^2 = \\sum_{i,j} A_{ij}^2` (Frobenius norm)

    :math:`||vec(A)||_1 = \\sum_{i,j} abs(A_{ij})` (Elementwise L1 norm)

    The generic norm :math:`||X - WH||_{loss}^2` may represent
    the Frobenius norm or another supported beta-divergence loss.
    The choice between options is controlled by the `beta_loss` parameter.

    The regularization terms are scaled by `n_features` for `W` and by `n_samples` for
    `H` to keep their impact balanced with respect to one another and to the data fit
    term as independent as possible of the size `n_samples` of the training set.

    The objective function is minimized with an alternating minimization of W
    and H. If H is given and update_H=False, it solves for W only.

    Note that the transformed data is named W and the components matrix is named H. In
    the NMF literature, the naming convention is usually the opposite since the data
    matrix X is transposed.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Constant matrix.

    W : array-like of shape (n_samples, n_components), default=None
        If `init='custom'`, it is used as initial guess for the solution.
        If `update_H=False`, it is initialised as an array of zeros, unless
        `solver='mu'`, then it is filled with values calculated by
        `np.sqrt(X.mean() / self._n_components)`.
        If `None`, uses the initialisation method specified in `init`.

    H : array-like of shape (n_components, n_features), default=None
        If `init='custom'`, it is used as initial guess for the solution.
        If `update_H=False`, it is used as a constant, to solve for W only.
        If `None`, uses the initialisation method specified in `init`.

    n_components : int or {'auto'} or None, default=None
        Number of components, if n_components is not set all features
        are kept.
        If `n_components='auto'`, the number of components is automatically inferred
        from `W` or `H` shapes.

        .. versionchanged:: 1.4
            Added `'auto'` value.

    init : {'random', 'nndsvd', 'nndsvda', 'nndsvdar', 'custom'}, default=None
        Method used to initialize the procedure.

        Valid options:

        - None: 'nndsvda' if n_components < n_features, otherwise 'random'.
        - 'random': non-negative random matrices, scaled with:
          `sqrt(X.mean() / n_components)`
        - 'nndsvd': Nonnegative Double Singular Value Decomposition (NNDSVD)
          initialization (better for sparseness)
        - 'nndsvda': NNDSVD with zeros filled with the average of X
          (better when sparsity is not desired)
        - 'nndsvdar': NNDSVD with zeros filled with small random values
          (generally faster, less accurate alternative to NNDSVDa
          for when sparsity is not desired)
        - 'custom': If `update_H=True`, use custom matrices W and H which must both
          be provided. If `update_H=False`, then only custom matrix H is used.

        .. versionchanged:: 0.23
            The default value of `init` changed from 'random' to None in 0.23.

        .. versionchanged:: 1.1
            When `init=None` and n_components is less than n_samples and n_features
            defaults to `nndsvda` instead of `nndsvd`.

    update_H : bool, default=True
        Set to True, both W and H will be estimated from initial guesses.
        Set to False, only W will be estimated.

    solver : {'cd', 'mu'}, default='cd'
        Numerical solver to use:

        - 'cd' is a Coordinate Descent solver that uses Fast Hierarchical
          Alternating Least Squares (Fast HALS).
        - 'mu' is a Multiplicative Update solver.

        .. versionadded:: 0.17
           Coordinate Descent solver.

        .. versionadded:: 0.19
           Multiplicative Update solver.

    beta_loss : float or {'frobenius', 'kullback-leibler', \
            'itakura-saito'}, default='frobenius'
        Beta divergence to be minimized, measuring the distance between X
        and the dot product WH. Note that values different from 'frobenius'
        (or 2) and 'kullback-leibler' (or 1) lead to significantly slower
        fits. Note that for beta_loss <= 0 (or 'itakura-saito'), the input
        matrix X cannot contain zeros. Used only in 'mu' solver.

        .. versionadded:: 0.19

    tol : float, default=1e-4
        Tolerance of the stopping condition.

    max_iter : int, default=200
        Maximum number of iterations before timing out.

    alpha_W : float, default=0.0
        Constant that multiplies the regularization terms of `W`. Set it to zero
        (default) to have no regularization on `W`.

        .. versionadded:: 1.0

    alpha_H : float or "same", default="same"
        Constant that multiplies the regularization terms of `H`. Set it to zero to
        have no regularization on `H`. If "same" (default), it takes the same value as
        `alpha_W`.

        .. versionadded:: 1.0

    l1_ratio : float, default=0.0
        The regularization mixing parameter, with 0 <= l1_ratio <= 1.
        For l1_ratio = 0 the penalty is an elementwise L2 penalty
        (aka Frobenius Norm).
        For l1_ratio = 1 it is an elementwise L1 penalty.
        For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.

    random_state : int, RandomState instance or None, default=None
        Used for NMF initialisation (when ``init`` == 'nndsvdar' or
        'random'), and in Coordinate Descent. Pass an int for reproducible
        results across multiple function calls.
        See :term:`Glossary <random_state>`.

    verbose : int, default=0
        The verbosity level.

    shuffle : bool, default=False
        If true, randomize the order of coordinates in the CD solver.

    Returns
    -------
    W : ndarray of shape (n_samples, n_components)
        Solution to the non-negative least squares problem.

    H : ndarray of shape (n_components, n_features)
        Solution to the non-negative least squares problem.

    n_iter : int
        Actual number of iterations.

    References
    ----------
    .. [1] :doi:`"Fast local algorithms for large scale nonnegative matrix and tensor
       factorizations" <10.1587/transfun.E92.A.708>`
       Cichocki, Andrzej, and P. H. A. N. Anh-Huy. IEICE transactions on fundamentals
       of electronics, communications and computer sciences 92.3: 708-721, 2009.

    .. [2] :doi:`"Algorithms for nonnegative matrix factorization with the
       beta-divergence" <10.1162/NECO_a_00168>`
       Fevotte, C., & Idier, J. (2011). Neural Computation, 23(9).

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1,1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
    >>> from sklearn.decomposition import non_negative_factorization
    >>> W, H, n_iter = non_negative_factorization(
    ...     X, n_components=2, init='random', random_state=0)
    """
    est = NMF(
        n_components=n_components,
        init=init,
        solver=solver,
        beta_loss=beta_loss,
        tol=tol,
        max_iter=max_iter,
        random_state=random_state,
        alpha_W=alpha_W,
        alpha_H=alpha_H,
        l1_ratio=l1_ratio,
        verbose=verbose,
        shuffle=shuffle,
    )
    est._validate_params()

    X = check_array(X, accept_sparse=("csr", "csc"), dtype=[np.float64, np.float32])

    with config_context(assume_finite=True):
        W, H, n_iter = est._fit_transform(X, W=W, H=H, update_H=update_H)

    return W, H, n_iter
