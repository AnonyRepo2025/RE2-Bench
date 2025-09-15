def check_array(array, accept_sparse=False, *, accept_large_sparse=True, dtype='numeric', order=None, copy=False, force_all_finite=True, ensure_2d=True, allow_nd=False, ensure_min_samples=1, ensure_min_features=1, estimator=None, input_name=''):
    if isinstance(array, np.matrix):
        raise TypeError('np.matrix is not supported. Please convert to a numpy array with np.asarray. For more information see: https://numpy.org/doc/stable/reference/generated/numpy.matrix.html')
    xp, is_array_api = get_namespace(array)
    array_orig = array
    dtype_numeric = isinstance(dtype, str) and dtype == 'numeric'
    dtype_orig = getattr(array, 'dtype', None)
    if not hasattr(dtype_orig, 'kind'):
        dtype_orig = None
    dtypes_orig = None
    pandas_requires_conversion = False
    if hasattr(array, 'dtypes') and hasattr(array.dtypes, '__array__'):
        with suppress(ImportError):
            from pandas.api.types import is_sparse
            if not hasattr(array, 'sparse') and array.dtypes.apply(is_sparse).any():
                warnings.warn('pandas.DataFrame with sparse columns found.It will be converted to a dense numpy array.')
        dtypes_orig = list(array.dtypes)
        pandas_requires_conversion = any((_pandas_dtype_needs_early_conversion(i) for i in dtypes_orig))
        if all((isinstance(dtype_iter, np.dtype) for dtype_iter in dtypes_orig)):
            dtype_orig = np.result_type(*dtypes_orig)
    elif hasattr(array, 'iloc') and hasattr(array, 'dtype'):
        pandas_requires_conversion = _pandas_dtype_needs_early_conversion(array.dtype)
        if isinstance(array.dtype, np.dtype):
            dtype_orig = array.dtype
        else:
            dtype_orig = None
    if dtype_numeric:
        if dtype_orig is not None and dtype_orig.kind == 'O':
            dtype = xp.float64
        else:
            dtype = None
    if isinstance(dtype, (list, tuple)):
        if dtype_orig is not None and dtype_orig in dtype:
            dtype = None
        else:
            dtype = dtype[0]
    if pandas_requires_conversion:
        new_dtype = dtype_orig if dtype is None else dtype
        array = array.astype(new_dtype)
        dtype = None
    if force_all_finite not in (True, False, 'allow-nan'):
        raise ValueError('force_all_finite should be a bool or "allow-nan". Got {!r} instead'.format(force_all_finite))
    estimator_name = _check_estimator_name(estimator)
    context = ' by %s' % estimator_name if estimator is not None else ''
    if hasattr(array, 'sparse') and array.ndim > 1:
        with suppress(ImportError):
            from pandas.api.types import is_sparse
            if array.dtypes.apply(is_sparse).all():
                array = array.sparse.to_coo()
                if array.dtype == np.dtype('object'):
                    unique_dtypes = set([dt.subtype.name for dt in array_orig.dtypes])
                    if len(unique_dtypes) > 1:
                        raise ValueError('Pandas DataFrame with mixed sparse extension arrays generated a sparse matrix with object dtype which can not be converted to a scipy sparse matrix.Sparse extension arrays should all have the same numeric type.')
    if sp.issparse(array):
        _ensure_no_complex_data(array)
        array = _ensure_sparse_format(array, accept_sparse=accept_sparse, dtype=dtype, copy=copy, force_all_finite=force_all_finite, accept_large_sparse=accept_large_sparse, estimator_name=estimator_name, input_name=input_name)
    else:
        with warnings.catch_warnings():
            try:
                warnings.simplefilter('error', ComplexWarning)
                if dtype is not None and np.dtype(dtype).kind in 'iu':
                    array = _asarray_with_order(array, order=order, xp=xp)
                    if array.dtype.kind == 'f':
                        _assert_all_finite(array, allow_nan=False, msg_dtype=dtype, estimator_name=estimator_name, input_name=input_name)
                    array = xp.astype(array, dtype, copy=False)
                else:
                    array = _asarray_with_order(array, order=order, dtype=dtype, xp=xp)
            except ComplexWarning as complex_warning:
                raise ValueError('Complex data not supported\n{}\n'.format(array)) from complex_warning
        _ensure_no_complex_data(array)
        if ensure_2d:
            if array.ndim == 0:
                raise ValueError('Expected 2D array, got scalar array instead:\narray={}.\nReshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.'.format(array))
            if array.ndim == 1:
                raise ValueError('Expected 2D array, got 1D array instead:\narray={}.\nReshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.'.format(array))
        if dtype_numeric and array.dtype.kind in 'USV':
            raise ValueError("dtype='numeric' is not compatible with arrays of bytes/strings.Convert your data to numeric values explicitly instead.")
        if not allow_nd and array.ndim >= 3:
            raise ValueError('Found array with dim %d. %s expected <= 2.' % (array.ndim, estimator_name))
        if force_all_finite:
            _assert_all_finite(array, input_name=input_name, estimator_name=estimator_name, allow_nan=force_all_finite == 'allow-nan')
    if ensure_min_samples > 0:
        n_samples = _num_samples(array)
        if n_samples < ensure_min_samples:
            raise ValueError('Found array with %d sample(s) (shape=%s) while a minimum of %d is required%s.' % (n_samples, array.shape, ensure_min_samples, context))
    if ensure_min_features > 0 and array.ndim == 2:
        n_features = array.shape[1]
        if n_features < ensure_min_features:
            raise ValueError('Found array with %d feature(s) (shape=%s) while a minimum of %d is required%s.' % (n_features, array.shape, ensure_min_features, context))
    if copy:
        if xp.__name__ in {'numpy', 'numpy.array_api'}:
            if np.may_share_memory(array, array_orig):
                array = _asarray_with_order(array, dtype=dtype, order=order, copy=True, xp=xp)
        else:
            array = _asarray_with_order(array, dtype=dtype, order=order, copy=True, xp=xp)
    return array

def get_namespace(*arrays):
    if not get_config()['array_api_dispatch']:
        return (_NumPyApiWrapper(), False)
    namespaces = {x.__array_namespace__() if hasattr(x, '__array_namespace__') else None for x in arrays if not isinstance(x, (bool, int, float, complex))}
    if not namespaces:
        raise ValueError('Unrecognized array input')
    if len(namespaces) != 1:
        raise ValueError(f'Multiple namespaces for array inputs: {namespaces}')
    xp, = namespaces
    if xp is None:
        return (_NumPyApiWrapper(), False)
    return (_ArrayAPIWrapper(xp), True)

def get_config():
    return _get_threadlocal_config().copy()

def _get_threadlocal_config():
    if not hasattr(_threadlocal, 'global_config'):
        _threadlocal.global_config = _global_config.copy()
    return _threadlocal.global_config

def _check_estimator_name(estimator):
    if estimator is not None:
        if isinstance(estimator, str):
            return estimator
        else:
            return estimator.__class__.__name__
    return None

def _asarray_with_order(array, dtype=None, order=None, copy=None, xp=None):
    if xp is None:
        xp, _ = get_namespace(array)
    if xp.__name__ in {'numpy', 'numpy.array_api'}:
        array = numpy.asarray(array, order=order, dtype=dtype)
        return xp.asarray(array, copy=copy)
    else:
        return xp.asarray(array, dtype=dtype, copy=copy)

def __getattr__(self, name):
    return getattr(numpy, name)

def asarray(self, x, *, dtype=None, device=None, copy=None):
    if copy is True:
        return numpy.array(x, copy=True, dtype=dtype)
    else:
        return numpy.asarray(x, dtype=dtype)

def _ensure_no_complex_data(array):
    if hasattr(array, 'dtype') and array.dtype is not None and hasattr(array.dtype, 'kind') and (array.dtype.kind == 'c'):
        raise ValueError('Complex data not supported\n{}\n'.format(array))

def _assert_all_finite(X, allow_nan=False, msg_dtype=None, estimator_name=None, input_name=''):
    xp, _ = get_namespace(X)
    if _get_config()['assume_finite']:
        return
    X = xp.asarray(X)
    if X.dtype == np.dtype('object') and (not allow_nan):
        if _object_dtype_isnan(X).any():
            raise ValueError('Input contains NaN')
    if X.dtype.kind not in 'fc':
        return
    with np.errstate(over='ignore'):
        first_pass_isfinite = xp.isfinite(xp.sum(X))
    if first_pass_isfinite:
        return
    use_cython = xp is np and X.data.contiguous and (X.dtype.type in {np.float32, np.float64})
    if use_cython:
        out = cy_isfinite(X.reshape(-1), allow_nan=allow_nan)
        has_nan_error = False if allow_nan else out == FiniteStatus.has_nan
        has_inf = out == FiniteStatus.has_infinite
    else:
        has_inf = xp.any(xp.isinf(X))
        has_nan_error = False if allow_nan else xp.any(xp.isnan(X))
    if has_inf or has_nan_error:
        if has_nan_error:
            type_err = 'NaN'
        else:
            msg_dtype = msg_dtype if msg_dtype is not None else X.dtype
            type_err = f'infinity or a value too large for {msg_dtype!r}'
        padded_input_name = input_name + ' ' if input_name else ''
        msg_err = f'Input {padded_input_name}contains {type_err}.'
        if estimator_name and input_name == 'X' and has_nan_error:
            msg_err += f'\n{estimator_name} does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is possible to preprocess the data, for instance by using an imputer transformer in a pipeline or drop samples with missing values. See https://scikit-learn.org/stable/modules/impute.html You can find a list of all estimators that handle NaN values at the following page: https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values'
        raise ValueError(msg_err)

def _num_samples(x):
    message = 'Expected sequence or array-like, got %s' % type(x)
    if hasattr(x, 'fit') and callable(x.fit):
        raise TypeError(message)
    if not hasattr(x, '__len__') and (not hasattr(x, 'shape')):
        if hasattr(x, '__array__'):
            x = np.asarray(x)
        else:
            raise TypeError(message)
    if hasattr(x, 'shape') and x.shape is not None:
        if len(x.shape) == 0:
            raise TypeError('Singleton array %r cannot be considered a valid collection.' % x)
        if isinstance(x.shape[0], numbers.Integral):
            return x.shape[0]
    try:
        return len(x)
    except TypeError as type_error:
        raise TypeError(message) from type_error

def check_consistent_length(*arrays):
    lengths = [_num_samples(X) for X in arrays if X is not None]
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        raise ValueError('Found input variables with inconsistent numbers of samples: %r' % [int(l) for l in lengths])

def __init__(self, *, neg_label=0, pos_label=1, sparse_output=False):
    self.neg_label = neg_label
    self.pos_label = pos_label
    self.sparse_output = sparse_output

def fit(self, y):
    self._validate_params()
    if self.neg_label >= self.pos_label:
        raise ValueError(f'neg_label={self.neg_label} must be strictly less than pos_label={self.pos_label}.')
    if self.sparse_output and (self.pos_label == 0 or self.neg_label != 0):
        raise ValueError(f'Sparse binarization is only supported with non zero pos_label and zero neg_label, got pos_label={self.pos_label} and neg_label={self.neg_label}')
    self.y_type_ = type_of_target(y, input_name='y')
    if 'multioutput' in self.y_type_:
        raise ValueError('Multioutput target data is not supported with label binarization')
    if _num_samples(y) == 0:
        raise ValueError('y has 0 samples: %r' % y)
    self.sparse_input_ = sp.issparse(y)
    self.classes_ = unique_labels(y)
    return self

def _validate_params(self):
    validate_parameter_constraints(self._parameter_constraints, self.get_params(deep=False), caller_name=self.__class__.__name__)



from numbers import Integral, Real
import warnings
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from scipy.special import xlogy
from ..preprocessing import LabelBinarizer
from ..preprocessing import LabelEncoder
from ..utils import assert_all_finite
from ..utils import check_array
from ..utils import check_consistent_length
from ..utils import column_or_1d
from ..utils.multiclass import unique_labels
from ..utils.multiclass import type_of_target
from ..utils.validation import _num_samples
from ..utils.sparsefuncs import count_nonzero
from ..utils._param_validation import StrOptions, Options, Interval, validate_params
from ..exceptions import UndefinedMetricWarning
from ._base import _check_pos_label_consistency

def log_loss(y_true, y_pred, *, eps='auto', normalize=True, sample_weight=None, labels=None):
    y_pred = check_array(y_pred, ensure_2d=False, dtype=[np.float64, np.float32, np.float16])
    if eps == 'auto':
        eps = np.finfo(y_pred.dtype).eps
    else:
        warnings.warn('Setting the eps parameter is deprecated and will be removed in 1.5. Instead eps will always havea default value of `np.finfo(y_pred.dtype).eps`.', FutureWarning)
    check_consistent_length(y_pred, y_true, sample_weight)
    lb = LabelBinarizer()
    if labels is not None:
        lb.fit(labels)
    else:
        lb.fit(y_true)
    if len(lb.classes_) == 1:
        if labels is None:
            raise ValueError('y_true contains only one label ({0}). Please provide the true labels explicitly through the labels argument.'.format(lb.classes_[0]))
        else:
            raise ValueError('The labels array needs to contain at least two labels for log_loss, got {0}.'.format(lb.classes_))
    transformed_labels = lb.transform(y_true)
    if transformed_labels.shape[1] == 1:
        transformed_labels = np.append(1 - transformed_labels, transformed_labels, axis=1)
    y_pred = np.clip(y_pred, eps, 1 - eps)
    if y_pred.ndim == 1:
        y_pred = y_pred[:, np.newaxis]
    if y_pred.shape[1] == 1:
        y_pred = np.append(1 - y_pred, y_pred, axis=1)
    transformed_labels = check_array(transformed_labels)
    if len(lb.classes_) != y_pred.shape[1]:
        if labels is None:
            raise ValueError('y_true and y_pred contain different number of classes {0}, {1}. Please provide the true labels explicitly through the labels argument. Classes found in y_true: {2}'.format(transformed_labels.shape[1], y_pred.shape[1], lb.classes_))
        else:
            raise ValueError('The number of classes in labels is different from that in y_pred. Classes found in labels: {0}'.format(lb.classes_))
    y_pred_sum = y_pred.sum(axis=1)
    if not np.isclose(y_pred_sum, 1, rtol=1e-15, atol=5 * eps).all():
        warnings.warn('The y_pred values do not sum to one. Starting from 1.5 thiswill result in an error.', UserWarning)
    y_pred = y_pred / y_pred_sum[:, np.newaxis]
    loss = -xlogy(transformed_labels, y_pred).sum(axis=1)
    return _weighted_sum(loss, sample_weight, normalize)