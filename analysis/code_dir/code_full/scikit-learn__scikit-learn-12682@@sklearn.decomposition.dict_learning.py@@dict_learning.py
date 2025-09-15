def _check_positive_coding(method, positive):
    if positive and method in ['omp', 'lars']:
        raise ValueError("Positive constraint not supported for '{}' coding method.".format(method))

def check_random_state(seed):
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState instance' % seed)

def sparse_encode(X, dictionary, gram=None, cov=None, algorithm='lasso_lars', n_nonzero_coefs=None, alpha=None, copy_cov=True, init=None, max_iter=1000, n_jobs=None, check_input=True, verbose=0, positive=False):
    if check_input:
        if algorithm == 'lasso_cd':
            dictionary = check_array(dictionary, order='C', dtype='float64')
            X = check_array(X, order='C', dtype='float64')
        else:
            dictionary = check_array(dictionary)
            X = check_array(X)
    n_samples, n_features = X.shape
    n_components = dictionary.shape[0]
    if gram is None and algorithm != 'threshold':
        gram = np.dot(dictionary, dictionary.T)
    if cov is None and algorithm != 'lasso_cd':
        copy_cov = False
        cov = np.dot(dictionary, X.T)
    if algorithm in ('lars', 'omp'):
        regularization = n_nonzero_coefs
        if regularization is None:
            regularization = min(max(n_features / 10, 1), n_components)
    else:
        regularization = alpha
        if regularization is None:
            regularization = 1.0
    if effective_n_jobs(n_jobs) == 1 or algorithm == 'threshold':
        code = _sparse_encode(X, dictionary, gram, cov=cov, algorithm=algorithm, regularization=regularization, copy_cov=copy_cov, init=init, max_iter=max_iter, check_input=False, verbose=verbose, positive=positive)
        return code
    code = np.empty((n_samples, n_components))
    slices = list(gen_even_slices(n_samples, effective_n_jobs(n_jobs)))
    code_views = Parallel(n_jobs=n_jobs, verbose=verbose)((delayed(_sparse_encode)(X[this_slice], dictionary, gram, cov[:, this_slice] if cov is not None else None, algorithm, regularization=regularization, copy_cov=copy_cov, init=init[this_slice] if init is not None else None, max_iter=max_iter, check_input=False, verbose=verbose, positive=positive) for this_slice in slices))
    for this_slice, this_view in zip(slices, code_views):
        code[this_slice] = this_view
    return code

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

def dict_learning(X, n_components, alpha, max_iter=100, tol=1e-08, method='lars', n_jobs=None, dict_init=None, code_init=None, callback=None, verbose=False, random_state=None, return_n_iter=False, positive_dict=False, positive_code=False, method_max_iter=1000):
    if method not in ('lars', 'cd'):
        raise ValueError('Coding method %r not supported as a fit algorithm.' % method)
    _check_positive_coding(method, positive_code)
    method = 'lasso_' + method
    t0 = time.time()
    alpha = float(alpha)
    random_state = check_random_state(random_state)
    if code_init is not None and dict_init is not None:
        code = np.array(code_init, order='F')
        dictionary = dict_init
    else:
        code, S, dictionary = linalg.svd(X, full_matrices=False)
        dictionary = S[:, np.newaxis] * dictionary
    r = len(dictionary)
    if n_components <= r:
        code = code[:, :n_components]
        dictionary = dictionary[:n_components, :]
    else:
        code = np.c_[code, np.zeros((len(code), n_components - r))]
        dictionary = np.r_[dictionary, np.zeros((n_components - r, dictionary.shape[1]))]
    dictionary = np.array(dictionary, order='F')
    residuals = 0
    errors = []
    current_cost = np.nan
    if verbose == 1:
        print('[dict_learning]', end=' ')
    ii = -1
    for ii in range(max_iter):
        dt = time.time() - t0
        if verbose == 1:
            sys.stdout.write('.')
            sys.stdout.flush()
        elif verbose:
            print('Iteration % 3i (elapsed time: % 3is, % 4.1fmn, current cost % 7.3f)' % (ii, dt, dt / 60, current_cost))
        code = sparse_encode(X, dictionary, algorithm=method, alpha=alpha, init=code, n_jobs=n_jobs, positive=positive_code, max_iter=method_max_iter, verbose=verbose)
        dictionary, residuals = _update_dict(dictionary.T, X.T, code.T, verbose=verbose, return_r2=True, random_state=random_state, positive=positive_dict)
        dictionary = dictionary.T
        current_cost = 0.5 * residuals + alpha * np.sum(np.abs(code))
        errors.append(current_cost)
        if ii > 0:
            dE = errors[-2] - errors[-1]
            if dE < tol * errors[-1]:
                if verbose == 1:
                    print('')
                elif verbose:
                    print('--- Convergence reached after %d iterations' % ii)
                break
        if ii % 5 == 0 and callback is not None:
            callback(locals())
    if return_n_iter:
        return (code, dictionary, errors, ii + 1)
    else:
        return (code, dictionary, errors)