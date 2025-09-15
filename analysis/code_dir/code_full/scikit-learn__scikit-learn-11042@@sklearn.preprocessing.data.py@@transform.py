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

def transform(self, y):
    check_is_fitted(self, 'classes_')
    y = column_or_1d(y, warn=True)
    if _num_samples(y) == 0:
        return np.array([])
    classes = np.unique(y)
    if len(np.intersect1d(classes, self.classes_)) < len(classes):
        diff = np.setdiff1d(classes, self.classes_)
        raise ValueError('y contains previously unseen labels: %s' % str(diff))
    return np.searchsorted(self.classes_, y)

def check_is_fitted(estimator, attributes, msg=None, all_or_any=all):
    if msg is None:
        msg = "This %(name)s instance is not fitted yet. Call 'fit' with appropriate arguments before using this method."
    if not hasattr(estimator, 'fit'):
        raise TypeError('%s is not an estimator instance.' % estimator)
    if not isinstance(attributes, (list, tuple)):
        attributes = [attributes]
    if not all_or_any([hasattr(estimator, attr) for attr in attributes]):
        raise NotFittedError(msg % {'name': type(estimator).__name__})

def column_or_1d(y, warn=False):
    shape = np.shape(y)
    if len(shape) == 1:
        return np.ravel(y)
    if len(shape) == 2 and shape[1] == 1:
        if warn:
            warnings.warn('A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().', DataConversionWarning, stacklevel=2)
        return np.ravel(y)
    raise ValueError('bad input shape {0}'.format(shape))



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

class OneHotEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, n_values='auto', categorical_features='all', dtype=np.float64, sparse=True, handle_unknown='error'):
        self.n_values = n_values
        self.categorical_features = categorical_features
        self.dtype = dtype
        self.sparse = sparse
        self.handle_unknown = handle_unknown

    def fit(self, X, y=None):
        self.fit_transform(X)
        return self

    def _fit_transform(self, X):
        X = check_array(X, dtype=np.int)
        if np.any(X < 0):
            raise ValueError('X needs to contain only non-negative integers.')
        n_samples, n_features = X.shape
        if isinstance(self.n_values, six.string_types) and self.n_values == 'auto':
            n_values = np.max(X, axis=0) + 1
        elif isinstance(self.n_values, numbers.Integral):
            if (np.max(X, axis=0) >= self.n_values).any():
                raise ValueError('Feature out of bounds for n_values=%d' % self.n_values)
            n_values = np.empty(n_features, dtype=np.int)
            n_values.fill(self.n_values)
        else:
            try:
                n_values = np.asarray(self.n_values, dtype=int)
            except (ValueError, TypeError):
                raise TypeError("Wrong type for parameter `n_values`. Expected 'auto', int or array of ints, got %r" % type(X))
            if n_values.ndim < 1 or n_values.shape[0] != X.shape[1]:
                raise ValueError('Shape mismatch: if n_values is an array, it has to be of shape (n_features,).')
        self.n_values_ = n_values
        n_values = np.hstack([[0], n_values])
        indices = np.cumsum(n_values)
        self.feature_indices_ = indices
        column_indices = (X + indices[:-1]).ravel()
        row_indices = np.repeat(np.arange(n_samples, dtype=np.int32), n_features)
        data = np.ones(n_samples * n_features)
        out = sparse.coo_matrix((data, (row_indices, column_indices)), shape=(n_samples, indices[-1]), dtype=self.dtype).tocsr()
        if isinstance(self.n_values, six.string_types) and self.n_values == 'auto':
            mask = np.array(out.sum(axis=0)).ravel() != 0
            active_features = np.where(mask)[0]
            out = out[:, active_features]
            self.active_features_ = active_features
        return out if self.sparse else out.toarray()

    def fit_transform(self, X, y=None):
        return _transform_selected(X, self._fit_transform, self.dtype, self.categorical_features, copy=True)

    def _transform(self, X):
        X = check_array(X, dtype=np.int)
        if np.any(X < 0):
            raise ValueError('X needs to contain only non-negative integers.')
        n_samples, n_features = X.shape
        indices = self.feature_indices_
        if n_features != indices.shape[0] - 1:
            raise ValueError('X has different shape than during fitting. Expected %d, got %d.' % (indices.shape[0] - 1, n_features))
        mask = (X < self.n_values_).ravel()
        if np.any(~mask):
            if self.handle_unknown not in ['error', 'ignore']:
                raise ValueError('handle_unknown should be either error or unknown got %s' % self.handle_unknown)
            if self.handle_unknown == 'error':
                raise ValueError('unknown categorical feature present %s during transform.' % X.ravel()[~mask])
        column_indices = (X + indices[:-1]).ravel()[mask]
        row_indices = np.repeat(np.arange(n_samples, dtype=np.int32), n_features)[mask]
        data = np.ones(np.sum(mask))
        out = sparse.coo_matrix((data, (row_indices, column_indices)), shape=(n_samples, indices[-1]), dtype=self.dtype).tocsr()
        if isinstance(self.n_values, six.string_types) and self.n_values == 'auto':
            out = out[:, self.active_features_]
        return out if self.sparse else out.toarray()

    def transform(self, X):
        return _transform_selected(X, self._transform, self.dtype, self.categorical_features, copy=True)