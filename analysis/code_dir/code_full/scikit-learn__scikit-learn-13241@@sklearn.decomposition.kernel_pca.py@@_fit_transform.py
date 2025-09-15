def fit_transform(self, X, y=None, **fit_params):
    if y is None:
        return self.fit(X, **fit_params).transform(X)
    else:
        return self.fit(X, y, **fit_params).transform(X)

def fit(self, K, y=None):
    K = check_array(K, dtype=FLOAT_DTYPES)
    n_samples = K.shape[0]
    self.K_fit_rows_ = np.sum(K, axis=0) / n_samples
    self.K_fit_all_ = self.K_fit_rows_.sum() / n_samples
    return self

def check_array(array, accept_sparse=False, accept_large_sparse=True, dtype='numeric', order=None, copy=False, force_all_finite=True, ensure_2d=True, allow_nd=False, ensure_min_samples=1, ensure_min_features=1, warn_on_dtype=False, estimator=None):
    if accept_sparse is None:
        warnings.warn("Passing 'None' to parameter 'accept_sparse' in methods check_array and check_X_y is deprecated in version 0.19 and will be removed in 0.21. Use 'accept_sparse=False'  instead.", DeprecationWarning)
        accept_sparse = False
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

def transform(self, K, copy=True):
    check_is_fitted(self, 'K_fit_all_')
    K = check_array(K, copy=copy, dtype=FLOAT_DTYPES)
    K_pred_cols = (np.sum(K, axis=1) / self.K_fit_rows_.shape[0])[:, np.newaxis]
    K -= self.K_fit_rows_
    K -= K_pred_cols
    K += self.K_fit_all_
    return K

def check_is_fitted(estimator, attributes, msg=None, all_or_any=all):
    if msg is None:
        msg = "This %(name)s instance is not fitted yet. Call 'fit' with appropriate arguments before using this method."
    if not hasattr(estimator, 'fit'):
        raise TypeError('%s is not an estimator instance.' % estimator)
    if not isinstance(attributes, (list, tuple)):
        attributes = [attributes]
    if not all_or_any([hasattr(estimator, attr) for attr in attributes]):
        raise NotFittedError(msg % {'name': type(estimator).__name__})

def svd_flip(u, v, u_based_decision=True):
    if u_based_decision:
        max_abs_cols = np.argmax(np.abs(u), axis=0)
        signs = np.sign(u[max_abs_cols, range(u.shape[1])])
        u *= signs
        v *= signs[:, np.newaxis]
    else:
        max_abs_rows = np.argmax(np.abs(v), axis=1)
        signs = np.sign(v[range(v.shape[0]), max_abs_rows])
        u *= signs
        v *= signs[:, np.newaxis]
    return (u, v)

def check_random_state(seed):
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState instance' % seed)



import numpy as np
from scipy import linalg
from scipy.sparse.linalg import eigsh
from ..utils import check_random_state
from ..utils.extmath import svd_flip
from ..utils.validation import check_is_fitted, check_array
from ..exceptions import NotFittedError
from ..base import BaseEstimator, TransformerMixin, _UnstableOn32BitMixin
from ..preprocessing import KernelCenterer
from ..metrics.pairwise import pairwise_kernels

class KernelPCA(BaseEstimator, TransformerMixin, _UnstableOn32BitMixin):

    def __init__(self, n_components=None, kernel='linear', gamma=None, degree=3, coef0=1, kernel_params=None, alpha=1.0, fit_inverse_transform=False, eigen_solver='auto', tol=0, max_iter=None, remove_zero_eig=False, random_state=None, copy_X=True, n_jobs=None):
        if fit_inverse_transform and kernel == 'precomputed':
            raise ValueError('Cannot fit_inverse_transform with a precomputed kernel.')
        self.n_components = n_components
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.alpha = alpha
        self.fit_inverse_transform = fit_inverse_transform
        self.eigen_solver = eigen_solver
        self.remove_zero_eig = remove_zero_eig
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.copy_X = copy_X

    @property
    def _pairwise(self):
        return self.kernel == 'precomputed'

    def _get_kernel(self, X, Y=None):
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {'gamma': self.gamma, 'degree': self.degree, 'coef0': self.coef0}
        return pairwise_kernels(X, Y, metric=self.kernel, filter_params=True, n_jobs=self.n_jobs, **params)

    def _fit_transform(self, K):
        K = self._centerer.fit_transform(K)
        if self.n_components is None:
            n_components = K.shape[0]
        else:
            n_components = min(K.shape[0], self.n_components)
        if self.eigen_solver == 'auto':
            if K.shape[0] > 200 and n_components < 10:
                eigen_solver = 'arpack'
            else:
                eigen_solver = 'dense'
        else:
            eigen_solver = self.eigen_solver
        if eigen_solver == 'dense':
            self.lambdas_, self.alphas_ = linalg.eigh(K, eigvals=(K.shape[0] - n_components, K.shape[0] - 1))
        elif eigen_solver == 'arpack':
            random_state = check_random_state(self.random_state)
            v0 = random_state.uniform(-1, 1, K.shape[0])
            self.lambdas_, self.alphas_ = eigsh(K, n_components, which='LA', tol=self.tol, maxiter=self.max_iter, v0=v0)
        self.alphas_, _ = svd_flip(self.alphas_, np.empty_like(self.alphas_).T)
        indices = self.lambdas_.argsort()[::-1]
        self.lambdas_ = self.lambdas_[indices]
        self.alphas_ = self.alphas_[:, indices]
        if self.remove_zero_eig or self.n_components is None:
            self.alphas_ = self.alphas_[:, self.lambdas_ > 0]
            self.lambdas_ = self.lambdas_[self.lambdas_ > 0]
        return K

    def _fit_inverse_transform(self, X_transformed, X):
        if hasattr(X, 'tocsr'):
            raise NotImplementedError('Inverse transform not implemented for sparse matrices!')
        n_samples = X_transformed.shape[0]
        K = self._get_kernel(X_transformed)
        K.flat[::n_samples + 1] += self.alpha
        self.dual_coef_ = linalg.solve(K, X, sym_pos=True, overwrite_a=True)
        self.X_transformed_fit_ = X_transformed

    def fit(self, X, y=None):
        X = check_array(X, accept_sparse='csr', copy=self.copy_X)
        self._centerer = KernelCenterer()
        K = self._get_kernel(X)
        self._fit_transform(K)
        if self.fit_inverse_transform:
            sqrt_lambdas = np.diag(np.sqrt(self.lambdas_))
            X_transformed = np.dot(self.alphas_, sqrt_lambdas)
            self._fit_inverse_transform(X_transformed, X)
        self.X_fit_ = X
        return self

    def fit_transform(self, X, y=None, **params):
        self.fit(X, **params)
        X_transformed = self.alphas_ * np.sqrt(self.lambdas_)
        if self.fit_inverse_transform:
            self._fit_inverse_transform(X_transformed, X)
        return X_transformed

    def transform(self, X):
        check_is_fitted(self, 'X_fit_')
        K = self._centerer.transform(self._get_kernel(X, self.X_fit_))
        return np.dot(K, self.alphas_ / np.sqrt(self.lambdas_))

    def inverse_transform(self, X):
        if not self.fit_inverse_transform:
            raise NotFittedError('The fit_inverse_transform parameter was not set to True when instantiating and hence the inverse transform is not available.')
        K = self._get_kernel(X, self.X_transformed_fit_)
        return np.dot(K, self.dual_coef_)