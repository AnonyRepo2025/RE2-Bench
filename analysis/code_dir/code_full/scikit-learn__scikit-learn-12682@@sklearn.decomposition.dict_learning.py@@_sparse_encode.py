def _check_positive_coding(method, positive):
    if positive and method in ['omp', 'lars']:
        raise ValueError("Positive constraint not supported for '{}' coding method.".format(method))

def row_norms(X, squared=False):
    if sparse.issparse(X):
        if not isinstance(X, sparse.csr_matrix):
            X = sparse.csr_matrix(X)
        norms = csr_row_norms(X)
    else:
        norms = np.einsum('ij,ij->i', X, X)
    if not squared:
        np.sqrt(norms, norms)
    return norms

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

def check_array(array, accept_sparse=False, accept_large_sparse=True, dtype='numeric', order=None, copy=False, force_all_finite=True, ensure_2d=True, allow_nd=False, ensure_min_samples=1, ensure_min_features=1, warn_on_dtype=None, estimator=None):
    if warn_on_dtype is not None:
        warnings.warn("'warn_on_dtype' is deprecated in version 0.21 and will be removed in 0.23. Don't set `warn_on_dtype` to remove this warning.", DeprecationWarning)
    array_orig = array
    dtype_numeric = isinstance(dtype, str) and dtype == 'numeric'
    dtype_orig = getattr(array, 'dtype', None)
    if not hasattr(dtype_orig, 'kind'):
        dtype_orig = None
    dtypes_orig = None
    if hasattr(array, 'dtypes') and hasattr(array.dtypes, '__array__'):
        dtypes_orig = np.array(array.dtypes)
    if dtype_numeric:
        if dtype_orig is not None and dtype_orig.kind == 'O':
            dtype = np.float64
        else:
            dtype = None
    if isinstance(dtype, (list, tuple)):
        if dtype_orig is not None and dtype_orig in dtype:
            dtype = None
        else:
            dtype = dtype[0]
    if force_all_finite not in (True, False, 'allow-nan'):
        raise ValueError('force_all_finite should be a bool or "allow-nan". Got {!r} instead'.format(force_all_finite))
    if estimator is not None:
        if isinstance(estimator, str):
            estimator_name = estimator
        else:
            estimator_name = estimator.__class__.__name__
    else:
        estimator_name = 'Estimator'
    context = ' by %s' % estimator_name if estimator is not None else ''
    if sp.issparse(array):
        _ensure_no_complex_data(array)
        array = _ensure_sparse_format(array, accept_sparse=accept_sparse, dtype=dtype, copy=copy, force_all_finite=force_all_finite, accept_large_sparse=accept_large_sparse)
    else:
        with warnings.catch_warnings():
            try:
                warnings.simplefilter('error', ComplexWarning)
                array = np.asarray(array, dtype=dtype, order=order)
            except ComplexWarning:
                raise ValueError('Complex data not supported\n{}\n'.format(array))
        _ensure_no_complex_data(array)
        if ensure_2d:
            if array.ndim == 0:
                raise ValueError('Expected 2D array, got scalar array instead:\narray={}.\nReshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.'.format(array))
            if array.ndim == 1:
                raise ValueError('Expected 2D array, got 1D array instead:\narray={}.\nReshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.'.format(array))
        if dtype_numeric and np.issubdtype(array.dtype, np.flexible):
            warnings.warn("Beginning in version 0.22, arrays of bytes/strings will be converted to decimal numbers if dtype='numeric'. It is recommended that you convert the array to a float dtype before using it in scikit-learn, for example by using your_array = your_array.astype(np.float64).", FutureWarning)
        if dtype_numeric and array.dtype.kind == 'O':
            array = array.astype(np.float64)
        if not allow_nd and array.ndim >= 3:
            raise ValueError('Found array with dim %d. %s expected <= 2.' % (array.ndim, estimator_name))
        if force_all_finite:
            _assert_all_finite(array, allow_nan=force_all_finite == 'allow-nan')
    if ensure_min_samples > 0:
        n_samples = _num_samples(array)
        if n_samples < ensure_min_samples:
            raise ValueError('Found array with %d sample(s) (shape=%s) while a minimum of %d is required%s.' % (n_samples, array.shape, ensure_min_samples, context))
    if ensure_min_features > 0 and array.ndim == 2:
        n_features = array.shape[1]
        if n_features < ensure_min_features:
            raise ValueError('Found array with %d feature(s) (shape=%s) while a minimum of %d is required%s.' % (n_features, array.shape, ensure_min_features, context))
    if warn_on_dtype and dtype_orig is not None and (array.dtype != dtype_orig):
        msg = 'Data with input dtype %s was converted to %s%s.' % (dtype_orig, array.dtype, context)
        warnings.warn(msg, DataConversionWarning)
    if copy and np.may_share_memory(array, array_orig):
        array = np.array(array, dtype=dtype, order=order)
    if warn_on_dtype and dtypes_orig is not None and ({array.dtype} != set(dtypes_orig)):
        msg = 'Data with input dtype %s were all converted to %s%s.' % (', '.join(map(str, sorted(set(dtypes_orig)))), array.dtype, context)
        warnings.warn(msg, DataConversionWarning, stacklevel=3)
    return array

def _ensure_no_complex_data(array):
    if hasattr(array, 'dtype') and array.dtype is not None and hasattr(array.dtype, 'kind') and (array.dtype.kind == 'c'):
        raise ValueError('Complex data not supported\n{}\n'.format(array))

def _assert_all_finite(X, allow_nan=False):
    from .extmath import _safe_accumulator_op
    if _get_config()['assume_finite']:
        return
    X = np.asanyarray(X)
    is_float = X.dtype.kind in 'fc'
    if is_float and np.isfinite(_safe_accumulator_op(np.sum, X)):
        pass
    elif is_float:
        msg_err = 'Input contains {} or a value too large for {!r}.'
        if allow_nan and np.isinf(X).any() or (not allow_nan and (not np.isfinite(X).all())):
            type_err = 'infinity' if allow_nan else 'NaN, infinity'
            raise ValueError(msg_err.format(type_err, X.dtype))
    elif X.dtype == np.dtype('object') and (not allow_nan):
        if _object_dtype_isnan(X).any():
            raise ValueError('Input contains NaN')

def get_config():
    return _global_config.copy()

def _safe_accumulator_op(op, x, *args, **kwargs):
    if np.issubdtype(x.dtype, np.floating) and x.dtype.itemsize < 8:
        result = op(x, *args, **kwargs, dtype=np.float64)
    else:
        result = op(x, *args, **kwargs)
    return result

def _num_samples(x):
    if hasattr(x, 'fit') and callable(x.fit):
        raise TypeError('Expected sequence or array-like, got estimator %s' % x)
    if not hasattr(x, '__len__') and (not hasattr(x, 'shape')):
        if hasattr(x, '__array__'):
            x = np.asarray(x)
        else:
            raise TypeError('Expected sequence or array-like, got %s' % type(x))
    if hasattr(x, 'shape'):
        if len(x.shape) == 0:
            raise TypeError('Singleton array %r cannot be considered a valid collection.' % x)
        if isinstance(x.shape[0], numbers.Integral):
            return x.shape[0]
        else:
            return len(x)
    else:
        return len(x)

def _gram_omp(Gram, Xy, n_nonzero_coefs, tol_0=None, tol=None, copy_Gram=True, copy_Xy=True, return_path=False):
    Gram = Gram.copy('F') if copy_Gram else np.asfortranarray(Gram)
    if copy_Xy or not Xy.flags.writeable:
        Xy = Xy.copy()
    min_float = np.finfo(Gram.dtype).eps
    nrm2, swap = linalg.get_blas_funcs(('nrm2', 'swap'), (Gram,))
    potrs, = get_lapack_funcs(('potrs',), (Gram,))
    indices = np.arange(len(Gram))
    alpha = Xy
    tol_curr = tol_0
    delta = 0
    gamma = np.empty(0)
    n_active = 0
    max_features = len(Gram) if tol is not None else n_nonzero_coefs
    L = np.empty((max_features, max_features), dtype=Gram.dtype)
    L[0, 0] = 1.0
    if return_path:
        coefs = np.empty_like(L)
    while True:
        lam = np.argmax(np.abs(alpha))
        if lam < n_active or alpha[lam] ** 2 < min_float:
            warnings.warn(premature, RuntimeWarning, stacklevel=3)
            break
        if n_active > 0:
            L[n_active, :n_active] = Gram[lam, :n_active]
            linalg.solve_triangular(L[:n_active, :n_active], L[n_active, :n_active], trans=0, lower=1, overwrite_b=True, check_finite=False)
            v = nrm2(L[n_active, :n_active]) ** 2
            Lkk = Gram[lam, lam] - v
            if Lkk <= min_float:
                warnings.warn(premature, RuntimeWarning, stacklevel=3)
                break
            L[n_active, n_active] = sqrt(Lkk)
        else:
            L[0, 0] = sqrt(Gram[lam, lam])
        Gram[n_active], Gram[lam] = swap(Gram[n_active], Gram[lam])
        Gram.T[n_active], Gram.T[lam] = swap(Gram.T[n_active], Gram.T[lam])
        indices[n_active], indices[lam] = (indices[lam], indices[n_active])
        Xy[n_active], Xy[lam] = (Xy[lam], Xy[n_active])
        n_active += 1
        gamma, _ = potrs(L[:n_active, :n_active], Xy[:n_active], lower=True, overwrite_b=False)
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
        return (gamma, indices[:n_active], coefs[:, :n_active], n_active)
    else:
        return (gamma, indices[:n_active], n_active)

def __init__(self, alpha=1.0, fit_intercept=True, verbose=False, normalize=True, precompute='auto', max_iter=500, eps=np.finfo(np.float).eps, copy_X=True, fit_path=True, positive=False):
    self.alpha = alpha
    self.fit_intercept = fit_intercept
    self.max_iter = max_iter
    self.verbose = verbose
    self.normalize = normalize
    self.positive = positive
    self.precompute = precompute
    self.copy_X = copy_X
    self.eps = eps
    self.fit_path = fit_path

def fit(self, X, y, Xy=None):
    X, y = check_X_y(X, y, y_numeric=True, multi_output=True)
    alpha = getattr(self, 'alpha', 0.0)
    if hasattr(self, 'n_nonzero_coefs'):
        alpha = 0.0
        max_iter = self.n_nonzero_coefs
    else:
        max_iter = self.max_iter
    self._fit(X, y, max_iter=max_iter, alpha=alpha, fit_path=self.fit_path, Xy=Xy)
    return self

def check_X_y(X, y, accept_sparse=False, accept_large_sparse=True, dtype='numeric', order=None, copy=False, force_all_finite=True, ensure_2d=True, allow_nd=False, multi_output=False, ensure_min_samples=1, ensure_min_features=1, y_numeric=False, warn_on_dtype=None, estimator=None):
    if y is None:
        raise ValueError('y cannot be None')
    X = check_array(X, accept_sparse=accept_sparse, accept_large_sparse=accept_large_sparse, dtype=dtype, order=order, copy=copy, force_all_finite=force_all_finite, ensure_2d=ensure_2d, allow_nd=allow_nd, ensure_min_samples=ensure_min_samples, ensure_min_features=ensure_min_features, warn_on_dtype=warn_on_dtype, estimator=estimator)
    if multi_output:
        y = check_array(y, 'csr', force_all_finite=True, ensure_2d=False, dtype=None)
    else:
        y = column_or_1d(y, warn=True)
        _assert_all_finite(y)
    if y_numeric and y.dtype.kind == 'O':
        y = y.astype(np.float64)
    check_consistent_length(X, y)
    return (X, y)

def check_consistent_length(*arrays):
    lengths = [_num_samples(X) for X in arrays if X is not None]
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        raise ValueError('Found input variables with inconsistent numbers of samples: %r' % [int(l) for l in lengths])

def _fit(self, X, y, max_iter, alpha, fit_path, Xy=None):
    n_features = X.shape[1]
    X, y, X_offset, y_offset, X_scale = self._preprocess_data(X, y, self.fit_intercept, self.normalize, self.copy_X)
    if y.ndim == 1:
        y = y[:, np.newaxis]
    n_targets = y.shape[1]
    Gram = self._get_gram(self.precompute, X, y)
    self.alphas_ = []
    self.n_iter_ = []
    self.coef_ = np.empty((n_targets, n_features))
    if fit_path:
        self.active_ = []
        self.coef_path_ = []
        for k in range(n_targets):
            this_Xy = None if Xy is None else Xy[:, k]
            alphas, active, coef_path, n_iter_ = lars_path(X, y[:, k], Gram=Gram, Xy=this_Xy, copy_X=self.copy_X, copy_Gram=True, alpha_min=alpha, method=self.method, verbose=max(0, self.verbose - 1), max_iter=max_iter, eps=self.eps, return_path=True, return_n_iter=True, positive=self.positive)
            self.alphas_.append(alphas)
            self.active_.append(active)
            self.n_iter_.append(n_iter_)
            self.coef_path_.append(coef_path)
            self.coef_[k] = coef_path[:, -1]
        if n_targets == 1:
            self.alphas_, self.active_, self.coef_path_, self.coef_ = [a[0] for a in (self.alphas_, self.active_, self.coef_path_, self.coef_)]
            self.n_iter_ = self.n_iter_[0]
    else:
        for k in range(n_targets):
            this_Xy = None if Xy is None else Xy[:, k]
            alphas, _, self.coef_[k], n_iter_ = lars_path(X, y[:, k], Gram=Gram, Xy=this_Xy, copy_X=self.copy_X, copy_Gram=True, alpha_min=alpha, method=self.method, verbose=max(0, self.verbose - 1), max_iter=max_iter, eps=self.eps, return_path=False, return_n_iter=True, positive=self.positive)
            self.alphas_.append(alphas)
            self.n_iter_.append(n_iter_)
        if n_targets == 1:
            self.alphas_ = self.alphas_[0]
            self.n_iter_ = self.n_iter_[0]
    self._set_intercept(X_offset, y_offset, X_scale)
    return self



import time
import sys
import itertools
from math import ceil
import numpy as np
from scipy import linalg
from joblib import Parallel, delayed, effective_n_jobs
from ..base import BaseEstimator, TransformerMixin
from ..utils import check_array, check_random_state, gen_even_slices, gen_batches
from ..utils.extmath import randomized_svd, row_norms
from ..utils.validation import check_is_fitted
from ..linear_model import Lasso, orthogonal_mp_gram, LassoLars, Lars

def _sparse_encode(X, dictionary, gram, cov=None, algorithm='lasso_lars', regularization=None, copy_cov=True, init=None, max_iter=1000, check_input=True, verbose=0, positive=False):
    if X.ndim == 1:
        X = X[:, np.newaxis]
    n_samples, n_features = X.shape
    n_components = dictionary.shape[0]
    if dictionary.shape[1] != X.shape[1]:
        raise ValueError('Dictionary and X have different numbers of features:dictionary.shape: {} X.shape{}'.format(dictionary.shape, X.shape))
    if cov is None and algorithm != 'lasso_cd':
        copy_cov = False
        cov = np.dot(dictionary, X.T)
    _check_positive_coding(algorithm, positive)
    if algorithm == 'lasso_lars':
        alpha = float(regularization) / n_features
        try:
            err_mgt = np.seterr(all='ignore')
            lasso_lars = LassoLars(alpha=alpha, fit_intercept=False, verbose=verbose, normalize=False, precompute=gram, fit_path=False, positive=positive, max_iter=max_iter)
            lasso_lars.fit(dictionary.T, X.T, Xy=cov)
            new_code = lasso_lars.coef_
        finally:
            np.seterr(**err_mgt)
    elif algorithm == 'lasso_cd':
        alpha = float(regularization) / n_features
        clf = Lasso(alpha=alpha, fit_intercept=False, normalize=False, precompute=gram, max_iter=max_iter, warm_start=True, positive=positive)
        if init is not None:
            clf.coef_ = init
        clf.fit(dictionary.T, X.T, check_input=check_input)
        new_code = clf.coef_
    elif algorithm == 'lars':
        try:
            err_mgt = np.seterr(all='ignore')
            lars = Lars(fit_intercept=False, verbose=verbose, normalize=False, precompute=gram, n_nonzero_coefs=int(regularization), fit_path=False)
            lars.fit(dictionary.T, X.T, Xy=cov)
            new_code = lars.coef_
        finally:
            np.seterr(**err_mgt)
    elif algorithm == 'threshold':
        new_code = (np.sign(cov) * np.maximum(np.abs(cov) - regularization, 0)).T
        if positive:
            np.clip(new_code, 0, None, out=new_code)
    elif algorithm == 'omp':
        new_code = orthogonal_mp_gram(Gram=gram, Xy=cov, n_nonzero_coefs=int(regularization), tol=None, norms_squared=row_norms(X, squared=True), copy_Xy=copy_cov).T
    else:
        raise ValueError('Sparse coding method must be "lasso_lars" "lasso_cd", "lasso", "threshold" or "omp", got %s.' % algorithm)
    if new_code.ndim != 2:
        return new_code.reshape(n_samples, n_components)
    return new_code