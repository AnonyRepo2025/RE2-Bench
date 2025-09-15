def check_array(array, accept_sparse=False, dtype='numeric', order=None, copy=False, force_all_finite=True, ensure_2d=True, allow_nd=False, ensure_min_samples=1, ensure_min_features=1, warn_on_dtype=False, estimator=None):
    if accept_sparse is None:
        warnings.warn("Passing 'None' to parameter 'accept_sparse' in methods check_array and check_X_y is deprecated in version 0.19 and will be removed in 0.21. Use 'accept_sparse=False'  instead.", DeprecationWarning)
        accept_sparse = False
    array_orig = array
    dtype_numeric = isinstance(dtype, six.string_types) and dtype == 'numeric'
    dtype_orig = getattr(array, 'dtype', None)
    if not hasattr(dtype_orig, 'kind'):
        dtype_orig = None
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
        if isinstance(estimator, six.string_types):
            estimator_name = estimator
        else:
            estimator_name = estimator.__class__.__name__
    else:
        estimator_name = 'Estimator'
    context = ' by %s' % estimator_name if estimator is not None else ''
    if sp.issparse(array):
        _ensure_no_complex_data(array)
        array = _ensure_sparse_format(array, accept_sparse, dtype, copy, force_all_finite)
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
            warnings.warn("Beginning in version 0.22, arrays of strings will be interpreted as decimal numbers if parameter 'dtype' is 'numeric'. It is recommended that you convert the array to type np.float64 before passing it to check_array.", FutureWarning)
        if dtype_numeric and array.dtype.kind == 'O':
            array = array.astype(np.float64)
        if not allow_nd and array.ndim >= 3:
            raise ValueError('Found array with dim %d. %s expected <= 2.' % (array.ndim, estimator_name))
        if force_all_finite:
            _assert_all_finite(array, allow_nan=force_all_finite == 'allow-nan')
    shape_repr = _shape_repr(array.shape)
    if ensure_min_samples > 0:
        n_samples = _num_samples(array)
        if n_samples < ensure_min_samples:
            raise ValueError('Found array with %d sample(s) (shape=%s) while a minimum of %d is required%s.' % (n_samples, shape_repr, ensure_min_samples, context))
    if ensure_min_features > 0 and array.ndim == 2:
        n_features = array.shape[1]
        if n_features < ensure_min_features:
            raise ValueError('Found array with %d feature(s) (shape=%s) while a minimum of %d is required%s.' % (n_features, shape_repr, ensure_min_features, context))
    if warn_on_dtype and dtype_orig is not None and (array.dtype != dtype_orig):
        msg = 'Data with input dtype %s was converted to %s%s.' % (dtype_orig, array.dtype, context)
        warnings.warn(msg, DataConversionWarning)
    if copy and np.may_share_memory(array, array_orig):
        array = np.array(array, dtype=dtype, order=order)
    return array

def _ensure_no_complex_data(array):
    if hasattr(array, 'dtype') and array.dtype is not None and hasattr(array.dtype, 'kind') and (array.dtype.kind == 'c'):
        raise ValueError('Complex data not supported\n{}\n'.format(array))

def _assert_all_finite(X, allow_nan=False):
    if _get_config()['assume_finite']:
        return
    X = np.asanyarray(X)
    is_float = X.dtype.kind in 'fc'
    if is_float and np.isfinite(X.sum()):
        pass
    elif is_float:
        msg_err = 'Input contains {} or a value too large for {!r}.'
        if allow_nan and np.isinf(X).any() or (not allow_nan and (not np.isfinite(X).all())):
            type_err = 'infinity' if allow_nan else 'NaN, infinity'
            raise ValueError(msg_err.format(type_err, X.dtype))

def get_config():
    return _global_config.copy()

def _shape_repr(shape):
    if len(shape) == 0:
        return '()'
    joined = ', '.join(('%d' % e for e in shape))
    if len(shape) == 1:
        joined += ','
    return '(%s)' % joined

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
        return x.shape[0]
    else:
        return len(x)

def _handle_zeros_in_scale(scale, copy=True):
    if np.isscalar(scale):
        if scale == 0.0:
            scale = 1.0
        return scale
    elif isinstance(scale, np.ndarray):
        if copy:
            scale = scale.copy()
        scale[scale == 0.0] = 1.0
        return scale

def _ensure_sparse_format(spmatrix, accept_sparse, dtype, copy, force_all_finite):
    if dtype is None:
        dtype = spmatrix.dtype
    changed_format = False
    if isinstance(accept_sparse, six.string_types):
        accept_sparse = [accept_sparse]
    if accept_sparse is False:
        raise TypeError('A sparse matrix was passed, but dense data is required. Use X.toarray() to convert to a dense numpy array.')
    elif isinstance(accept_sparse, (list, tuple)):
        if len(accept_sparse) == 0:
            raise ValueError("When providing 'accept_sparse' as a tuple or list, it must contain at least one string value.")
        if spmatrix.format not in accept_sparse:
            spmatrix = spmatrix.asformat(accept_sparse[0])
            changed_format = True
    elif accept_sparse is not True:
        raise ValueError("Parameter 'accept_sparse' should be a string, boolean or list of strings. You provided 'accept_sparse={}'.".format(accept_sparse))
    if dtype != spmatrix.dtype:
        spmatrix = spmatrix.astype(dtype)
    elif copy and (not changed_format):
        spmatrix = spmatrix.copy()
    if force_all_finite:
        if not hasattr(spmatrix, 'data'):
            warnings.warn("Can't check %s sparse matrix for nan or inf." % spmatrix.format)
        else:
            _assert_all_finite(spmatrix.data, allow_nan=force_all_finite == 'allow-nan')
    return spmatrix

def min_max_axis(X, axis, ignore_nan=False):
    if isinstance(X, sp.csr_matrix) or isinstance(X, sp.csc_matrix):
        if ignore_nan:
            return _sparse_nan_min_max(X, axis=axis)
        else:
            return _sparse_min_max(X, axis=axis)
    else:
        _raise_typeerror(X)

def _sparse_min_max(X, axis):
    return (_sparse_min_or_max(X, axis, np.minimum), _sparse_min_or_max(X, axis, np.maximum))

def _sparse_min_or_max(X, axis, min_or_max):
    if axis is None:
        if 0 in X.shape:
            raise ValueError('zero-size array to reduction operation')
        zero = X.dtype.type(0)
        if X.nnz == 0:
            return zero
        m = min_or_max.reduce(X.data.ravel())
        if X.nnz != np.product(X.shape):
            m = min_or_max(zero, m)
        return m
    if axis < 0:
        axis += 2
    if axis == 0 or axis == 1:
        return _min_or_max_axis(X, axis, min_or_max)
    else:
        raise ValueError('invalid axis, use 0 for rows, or 1 for columns')

def _min_or_max_axis(X, axis, min_or_max):
    N = X.shape[axis]
    if N == 0:
        raise ValueError('zero-size array to reduction operation')
    M = X.shape[1 - axis]
    mat = X.tocsc() if axis == 0 else X.tocsr()
    mat.sum_duplicates()
    major_index, value = _minor_reduce(mat, min_or_max)
    not_full = np.diff(mat.indptr)[major_index] < N
    value[not_full] = min_or_max(value[not_full], 0)
    mask = value != 0
    major_index = np.compress(mask, major_index)
    value = np.compress(mask, value)
    if axis == 0:
        res = sp.coo_matrix((value, (np.zeros(len(value)), major_index)), dtype=X.dtype, shape=(1, M))
    else:
        res = sp.coo_matrix((value, (major_index, np.zeros(len(value)))), dtype=X.dtype, shape=(M, 1))
    return res.A.ravel()

def _minor_reduce(X, ufunc):
    major_index = np.flatnonzero(np.diff(X.indptr))
    value = ufunc.reduceat(X.data, X.indptr[major_index])
    return (major_index, value)



from __future__ import division
from itertools import chain, combinations
import numbers
import warnings
from itertools import combinations_with_replacement as combinations_w_r
from distutils.version import LooseVersion
import numpy as np
from scipy import sparse
from scipy import stats
from ..base import BaseEstimator, TransformerMixin
from ..externals import six
from ..externals.six import string_types
from ..utils import check_array
from ..utils.extmath import row_norms
from ..utils.extmath import _incremental_mean_and_var
from ..utils.fixes import _argmax, nanpercentile
from ..utils.sparsefuncs_fast import inplace_csr_row_normalize_l1, inplace_csr_row_normalize_l2
from ..utils.sparsefuncs import inplace_column_scale, mean_variance_axis, incr_mean_variance_axis, min_max_axis
from ..utils.validation import check_is_fitted, check_random_state, FLOAT_DTYPES
from .label import LabelEncoder
BOUNDS_THRESHOLD = 1e-07
zip = six.moves.zip
map = six.moves.map
range = six.moves.range
__all__ = ['Binarizer', 'KernelCenterer', 'MinMaxScaler', 'MaxAbsScaler', 'Normalizer', 'OneHotEncoder', 'RobustScaler', 'StandardScaler', 'QuantileTransformer', 'PowerTransformer', 'add_dummy_feature', 'binarize', 'normalize', 'scale', 'robust_scale', 'maxabs_scale', 'minmax_scale', 'quantile_transform', 'power_transform']

class StandardScaler(BaseEstimator, TransformerMixin):

    def __init__(self, copy=True, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.copy = copy

    def _reset(self):
        if hasattr(self, 'scale_'):
            del self.scale_
            del self.n_samples_seen_
            del self.mean_
            del self.var_

    def fit(self, X, y=None):
        self._reset()
        return self.partial_fit(X, y)

    def partial_fit(self, X, y=None):
        X = check_array(X, accept_sparse=('csr', 'csc'), copy=self.copy, warn_on_dtype=True, estimator=self, dtype=FLOAT_DTYPES)
        if sparse.issparse(X):
            if self.with_mean:
                raise ValueError('Cannot center sparse matrices: pass `with_mean=False` instead. See docstring for motivation and alternatives.')
            if self.with_std:
                if not hasattr(self, 'n_samples_seen_'):
                    self.mean_, self.var_ = mean_variance_axis(X, axis=0)
                    self.n_samples_seen_ = X.shape[0]
                else:
                    self.mean_, self.var_, self.n_samples_seen_ = incr_mean_variance_axis(X, axis=0, last_mean=self.mean_, last_var=self.var_, last_n=self.n_samples_seen_)
            else:
                self.mean_ = None
                self.var_ = None
                if not hasattr(self, 'n_samples_seen_'):
                    self.n_samples_seen_ = X.shape[0]
                else:
                    self.n_samples_seen_ += X.shape[0]
        else:
            if not hasattr(self, 'n_samples_seen_'):
                self.mean_ = 0.0
                self.n_samples_seen_ = 0
                if self.with_std:
                    self.var_ = 0.0
                else:
                    self.var_ = None
            if not self.with_mean and (not self.with_std):
                self.mean_ = None
                self.var_ = None
                self.n_samples_seen_ += X.shape[0]
            else:
                self.mean_, self.var_, self.n_samples_seen_ = _incremental_mean_and_var(X, self.mean_, self.var_, self.n_samples_seen_)
        if self.with_std:
            self.scale_ = _handle_zeros_in_scale(np.sqrt(self.var_))
        else:
            self.scale_ = None
        return self

    def transform(self, X, y='deprecated', copy=None):
        if not isinstance(y, string_types) or y != 'deprecated':
            warnings.warn('The parameter y on transform() is deprecated since 0.19 and will be removed in 0.21', DeprecationWarning)
        check_is_fitted(self, 'scale_')
        copy = copy if copy is not None else self.copy
        X = check_array(X, accept_sparse='csr', copy=copy, warn_on_dtype=True, estimator=self, dtype=FLOAT_DTYPES)
        if sparse.issparse(X):
            if self.with_mean:
                raise ValueError('Cannot center sparse matrices: pass `with_mean=False` instead. See docstring for motivation and alternatives.')
            if self.scale_ is not None:
                inplace_column_scale(X, 1 / self.scale_)
        else:
            if self.with_mean:
                X -= self.mean_
            if self.with_std:
                X /= self.scale_
        return X

    def inverse_transform(self, X, copy=None):
        check_is_fitted(self, 'scale_')
        copy = copy if copy is not None else self.copy
        if sparse.issparse(X):
            if self.with_mean:
                raise ValueError('Cannot uncenter sparse matrices: pass `with_mean=False` instead See docstring for motivation and alternatives.')
            if not sparse.isspmatrix_csr(X):
                X = X.tocsr()
                copy = False
            if copy:
                X = X.copy()
            if self.scale_ is not None:
                inplace_column_scale(X, self.scale_)
        else:
            X = np.asarray(X)
            if copy:
                X = X.copy()
            if self.with_std:
                X *= self.scale_
            if self.with_mean:
                X += self.mean_
        return X