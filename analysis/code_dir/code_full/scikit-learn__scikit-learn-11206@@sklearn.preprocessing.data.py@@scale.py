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

def mean_variance_axis(X, axis):
    _raise_error_wrong_axis(axis)
    if isinstance(X, sp.csr_matrix):
        if axis == 0:
            return _csr_mean_var_axis0(X)
        else:
            return _csc_mean_var_axis0(X.T)
    elif isinstance(X, sp.csc_matrix):
        if axis == 0:
            return _csc_mean_var_axis0(X)
        else:
            return _csr_mean_var_axis0(X.T)
    else:
        _raise_typeerror(X)

def _raise_error_wrong_axis(axis):
    if axis not in (0, 1):
        raise ValueError('Unknown axis value: %d. Use 0 for rows, or 1 for columns' % axis)

def inplace_column_scale(X, scale):
    if isinstance(X, sp.csc_matrix):
        inplace_csr_row_scale(X.T, scale)
    elif isinstance(X, sp.csr_matrix):
        inplace_csr_column_scale(X, scale)
    else:
        _raise_typeerror(X)

def inplace_csr_row_scale(X, scale):
    assert scale.shape[0] == X.shape[0]
    X.data *= np.repeat(scale, np.diff(X.indptr))



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

def scale(X, axis=0, with_mean=True, with_std=True, copy=True):
    X = check_array(X, accept_sparse='csc', copy=copy, ensure_2d=False, warn_on_dtype=True, estimator='the scale function', dtype=FLOAT_DTYPES, force_all_finite='allow-nan')
    if sparse.issparse(X):
        if with_mean:
            raise ValueError('Cannot center sparse matrices: pass `with_mean=False` instead See docstring for motivation and alternatives.')
        if axis != 0:
            raise ValueError('Can only scale sparse matrix on axis=0,  got axis=%d' % axis)
        if with_std:
            _, var = mean_variance_axis(X, axis=0)
            var = _handle_zeros_in_scale(var, copy=False)
            inplace_column_scale(X, 1 / np.sqrt(var))
    else:
        X = np.asarray(X)
        if with_mean:
            mean_ = np.nanmean(X, axis)
        if with_std:
            scale_ = np.nanstd(X, axis)
        Xr = np.rollaxis(X, axis)
        if with_mean:
            Xr -= mean_
            mean_1 = np.nanmean(Xr, axis=0)
            if not np.allclose(mean_1, 0):
                warnings.warn('Numerical issues were encountered when centering the data and might not be solved. Dataset may contain too large values. You may need to prescale your features.')
                Xr -= mean_1
        if with_std:
            scale_ = _handle_zeros_in_scale(scale_, copy=False)
            Xr /= scale_
            if with_mean:
                mean_2 = np.nanmean(Xr, axis=0)
                if not np.allclose(mean_2, 0):
                    warnings.warn('Numerical issues were encountered when scaling the data and might not be solved. The standard deviation of the data is probably very close to 0. ')
                    Xr -= mean_2
    return X