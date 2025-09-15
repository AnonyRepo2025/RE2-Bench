def check_pairwise_arrays(X, Y, precomputed=False, dtype=None):
    X, Y, dtype_float = _return_float_dtype(X, Y)
    estimator = 'check_pairwise_arrays'
    if dtype is None:
        dtype = dtype_float
    if Y is X or Y is None:
        X = Y = check_array(X, accept_sparse='csr', dtype=dtype, estimator=estimator)
    else:
        X = check_array(X, accept_sparse='csr', dtype=dtype, estimator=estimator)
        Y = check_array(Y, accept_sparse='csr', dtype=dtype, estimator=estimator)
    if precomputed:
        if X.shape[1] != Y.shape[0]:
            raise ValueError('Precomputed metric requires shape (n_queries, n_indexed). Got (%d, %d) for %d indexed.' % (X.shape[0], X.shape[1], Y.shape[0]))
    elif X.shape[1] != Y.shape[1]:
        raise ValueError('Incompatible dimension for X and Y matrices: X.shape[1] == %d while Y.shape[1] == %d' % (X.shape[1], Y.shape[1]))
    return (X, Y)

def _return_float_dtype(X, Y):
    if not issparse(X) and (not isinstance(X, np.ndarray)):
        X = np.asarray(X)
    if Y is None:
        Y_dtype = X.dtype
    elif not issparse(Y) and (not isinstance(Y, np.ndarray)):
        Y = np.asarray(Y)
        Y_dtype = Y.dtype
    else:
        Y_dtype = Y.dtype
    if X.dtype == Y_dtype == np.float32:
        dtype = np.float32
    else:
        dtype = np.float
    return (X, Y, dtype)

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

def safe_sparse_dot(a, b, dense_output=False):
    if sparse.issparse(a) or sparse.issparse(b):
        ret = a * b
        if dense_output and hasattr(ret, 'toarray'):
            ret = ret.toarray()
        return ret
    else:
        return np.dot(a, b)

def _ensure_sparse_format(spmatrix, accept_sparse, dtype, copy, force_all_finite, accept_large_sparse):
    if dtype is None:
        dtype = spmatrix.dtype
    changed_format = False
    if isinstance(accept_sparse, str):
        accept_sparse = [accept_sparse]
    _check_large_sparse(spmatrix, accept_large_sparse)
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

def _check_large_sparse(X, accept_large_sparse=False):
    if not accept_large_sparse:
        supported_indices = ['int32']
        if X.getformat() == 'coo':
            index_keys = ['col', 'row']
        elif X.getformat() in ['csr', 'csc', 'bsr']:
            index_keys = ['indices', 'indptr']
        else:
            return
        for key in index_keys:
            indices_datatype = getattr(X, key).dtype
            if indices_datatype not in supported_indices:
                raise ValueError('Only sparse matrices with 32-bit integer indices are accepted. Got %s indices.' % indices_datatype)

def rbf_kernel(X, Y=None, gamma=None):
    X, Y = check_pairwise_arrays(X, Y)
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    K = euclidean_distances(X, Y, squared=True)
    K *= -gamma
    np.exp(K, K)
    return K

def _euclidean_distances_upcast(X, XX=None, Y=None, YY=None):
    n_samples_X = X.shape[0]
    n_samples_Y = Y.shape[0]
    n_features = X.shape[1]
    distances = np.empty((n_samples_X, n_samples_Y), dtype=np.float32)
    x_density = X.nnz / np.prod(X.shape) if issparse(X) else 1
    y_density = Y.nnz / np.prod(Y.shape) if issparse(Y) else 1
    maxmem = max(((x_density * n_samples_X + y_density * n_samples_Y) * n_features + x_density * n_samples_X * y_density * n_samples_Y) / 10, 10 * 2 ** 17)
    tmp = (x_density + y_density) * n_features
    batch_size = (-tmp + np.sqrt(tmp ** 2 + 4 * maxmem)) / 2
    batch_size = max(int(batch_size), 1)
    x_batches = gen_batches(X.shape[0], batch_size)
    y_batches = gen_batches(Y.shape[0], batch_size)
    for i, x_slice in enumerate(x_batches):
        X_chunk = X[x_slice].astype(np.float64)
        if XX is None:
            XX_chunk = row_norms(X_chunk, squared=True)[:, np.newaxis]
        else:
            XX_chunk = XX[x_slice]
        for j, y_slice in enumerate(y_batches):
            if X is Y and j < i:
                d = distances[y_slice, x_slice].T
            else:
                Y_chunk = Y[y_slice].astype(np.float64)
                if YY is None:
                    YY_chunk = row_norms(Y_chunk, squared=True)[np.newaxis, :]
                else:
                    YY_chunk = YY[:, y_slice]
                d = -2 * safe_sparse_dot(X_chunk, Y_chunk.T, dense_output=True)
                d += XX_chunk
                d += YY_chunk
            distances[x_slice, y_slice] = d.astype(np.float32, copy=False)
    return distances

def gen_batches(n, batch_size, min_batch_size=0):
    start = 0
    for _ in range(int(n // batch_size)):
        end = start + batch_size
        if end + min_batch_size > n:
            continue
        yield slice(start, end)
        start = end
    if start < n:
        yield slice(start, n)



import itertools
from functools import partial
import warnings
import numpy as np
from scipy.spatial import distance
from scipy.sparse import csr_matrix
from scipy.sparse import issparse
from ..utils.validation import _num_samples
from ..utils.validation import check_non_negative
from ..utils import check_array
from ..utils import gen_even_slices
from ..utils import gen_batches, get_chunk_n_rows
from ..utils.extmath import row_norms, safe_sparse_dot
from ..preprocessing import normalize
from ..utils._joblib import Parallel
from ..utils._joblib import delayed
from ..utils._joblib import effective_n_jobs
from .pairwise_fast import _chi2_kernel_fast, _sparse_manhattan
from ..exceptions import DataConversionWarning
from sklearn.neighbors import DistanceMetric
from ..gaussian_process.kernels import Kernel as GPKernel
PAIRED_DISTANCES = {'cosine': paired_cosine_distances, 'euclidean': paired_euclidean_distances, 'l2': paired_euclidean_distances, 'l1': paired_manhattan_distances, 'manhattan': paired_manhattan_distances, 'cityblock': paired_manhattan_distances}
PAIRWISE_DISTANCE_FUNCTIONS = {'cityblock': manhattan_distances, 'cosine': cosine_distances, 'euclidean': euclidean_distances, 'haversine': haversine_distances, 'l2': euclidean_distances, 'l1': manhattan_distances, 'manhattan': manhattan_distances, 'precomputed': None}
_VALID_METRICS = ['euclidean', 'l2', 'l1', 'manhattan', 'cityblock', 'braycurtis', 'canberra', 'chebyshev', 'correlation', 'cosine', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule', 'wminkowski', 'haversine']
PAIRWISE_BOOLEAN_FUNCTIONS = ['dice', 'jaccard', 'kulsinski', 'matching', 'rogerstanimoto', 'russellrao', 'sokalmichener', 'sokalsneath', 'yule']
PAIRWISE_KERNEL_FUNCTIONS = {'additive_chi2': additive_chi2_kernel, 'chi2': chi2_kernel, 'linear': linear_kernel, 'polynomial': polynomial_kernel, 'poly': polynomial_kernel, 'rbf': rbf_kernel, 'laplacian': laplacian_kernel, 'sigmoid': sigmoid_kernel, 'cosine': cosine_similarity}
KERNEL_PARAMS = {'additive_chi2': (), 'chi2': frozenset(['gamma']), 'cosine': (), 'linear': (), 'poly': frozenset(['gamma', 'degree', 'coef0']), 'polynomial': frozenset(['gamma', 'degree', 'coef0']), 'rbf': frozenset(['gamma']), 'laplacian': frozenset(['gamma']), 'sigmoid': frozenset(['gamma', 'coef0'])}

def euclidean_distances(X, Y=None, Y_norm_squared=None, squared=False, X_norm_squared=None):
    X, Y = check_pairwise_arrays(X, Y)
    if X_norm_squared is not None:
        XX = check_array(X_norm_squared)
        if XX.shape == (1, X.shape[0]):
            XX = XX.T
        elif XX.shape != (X.shape[0], 1):
            raise ValueError('Incompatible dimensions for X and X_norm_squared')
        if XX.dtype == np.float32:
            XX = None
    elif X.dtype == np.float32:
        XX = None
    else:
        XX = row_norms(X, squared=True)[:, np.newaxis]
    if X is Y and XX is not None:
        YY = XX.T
    elif Y_norm_squared is not None:
        YY = np.atleast_2d(Y_norm_squared)
        if YY.shape != (1, Y.shape[0]):
            raise ValueError('Incompatible dimensions for Y and Y_norm_squared')
        if YY.dtype == np.float32:
            YY = None
    elif Y.dtype == np.float32:
        YY = None
    else:
        YY = row_norms(Y, squared=True)[np.newaxis, :]
    if X.dtype == np.float32:
        distances = _euclidean_distances_upcast(X, XX, Y, YY)
    else:
        distances = -2 * safe_sparse_dot(X, Y.T, dense_output=True)
        distances += XX
        distances += YY
    np.maximum(distances, 0, out=distances)
    if X is Y:
        np.fill_diagonal(distances, 0)
    return distances if squared else np.sqrt(distances, out=distances)