def _transform(self, X, handle_unknown='error', force_all_finite=True, warn_on_unknown=False):
    self._check_feature_names(X, reset=False)
    self._check_n_features(X, reset=False)
    X_list, n_samples, n_features = self._check_X(X, force_all_finite=force_all_finite)
    X_int = np.zeros((n_samples, n_features), dtype=int)
    X_mask = np.ones((n_samples, n_features), dtype=bool)
    columns_with_unknown = []
    for i in range(n_features):
        Xi = X_list[i]
        diff, valid_mask = _check_unknown(Xi, self.categories_[i], return_mask=True)
        if not np.all(valid_mask):
            if handle_unknown == 'error':
                msg = 'Found unknown categories {0} in column {1} during transform'.format(diff, i)
                raise ValueError(msg)
            else:
                if warn_on_unknown:
                    columns_with_unknown.append(i)
                X_mask[:, i] = valid_mask
                if self.categories_[i].dtype.kind in ('U', 'S') and self.categories_[i].itemsize > Xi.itemsize:
                    Xi = Xi.astype(self.categories_[i].dtype)
                elif self.categories_[i].dtype.kind == 'O' and Xi.dtype.kind == 'U':
                    Xi = Xi.astype('O')
                else:
                    Xi = Xi.copy()
                Xi[~valid_mask] = self.categories_[i][0]
        X_int[:, i] = _encode(Xi, uniques=self.categories_[i], check_unknown=False)
    if columns_with_unknown:
        warnings.warn(f'Found unknown categories in columns {columns_with_unknown} during transform. These unknown categories will be encoded as all zeros', UserWarning)
    return (X_int, X_mask)

def _check_feature_names(self, X, *, reset):
    if reset:
        feature_names_in = _get_feature_names(X)
        if feature_names_in is not None:
            self.feature_names_in_ = feature_names_in
        elif hasattr(self, 'feature_names_in_'):
            delattr(self, 'feature_names_in_')
        return
    fitted_feature_names = getattr(self, 'feature_names_in_', None)
    X_feature_names = _get_feature_names(X)
    if fitted_feature_names is None and X_feature_names is None:
        return
    if X_feature_names is not None and fitted_feature_names is None:
        warnings.warn(f'X has feature names, but {self.__class__.__name__} was fitted without feature names')
        return
    if X_feature_names is None and fitted_feature_names is not None:
        warnings.warn(f'X does not have valid feature names, but {self.__class__.__name__} was fitted with feature names')
        return
    if len(fitted_feature_names) != len(X_feature_names) or np.any(fitted_feature_names != X_feature_names):
        message = 'The feature names should match those that were passed during fit.\n'
        fitted_feature_names_set = set(fitted_feature_names)
        X_feature_names_set = set(X_feature_names)
        unexpected_names = sorted(X_feature_names_set - fitted_feature_names_set)
        missing_names = sorted(fitted_feature_names_set - X_feature_names_set)

        def add_names(names):
            output = ''
            max_n_names = 5
            for i, name in enumerate(names):
                if i >= max_n_names:
                    output += '- ...\n'
                    break
                output += f'- {name}\n'
            return output
        if unexpected_names:
            message += 'Feature names unseen at fit time:\n'
            message += add_names(unexpected_names)
        if missing_names:
            message += 'Feature names seen at fit time, yet now missing:\n'
            message += add_names(missing_names)
        if not missing_names and (not unexpected_names):
            message += 'Feature names must be in the same order as they were in fit.\n'
        raise ValueError(message)

def _get_feature_names(X):
    feature_names = None
    if hasattr(X, 'columns'):
        feature_names = np.asarray(X.columns, dtype=object)
    if feature_names is None or len(feature_names) == 0:
        return
    types = sorted((t.__qualname__ for t in set((type(v) for v in feature_names))))
    if len(types) > 1 and 'str' in types:
        raise TypeError(f'Feature names are only supported if all input features have string names, but your input has {types} as feature name / column name types. If you want feature names to be stored and validated, you must convert them all to strings, by using X.columns = X.columns.astype(str) for example. Otherwise you can remove feature / column names from your input data, or convert them all to a non-string data type.')
    if len(types) == 1 and types[0] == 'str':
        return feature_names

def _check_n_features(self, X, reset):
    try:
        n_features = _num_features(X)
    except TypeError as e:
        if not reset and hasattr(self, 'n_features_in_'):
            raise ValueError(f'X does not contain any features, but {self.__class__.__name__} is expecting {self.n_features_in_} features') from e
        return
    if reset:
        self.n_features_in_ = n_features
        return
    if not hasattr(self, 'n_features_in_'):
        return
    if n_features != self.n_features_in_:
        raise ValueError(f'X has {n_features} features, but {self.__class__.__name__} is expecting {self.n_features_in_} features as input.')

def _num_features(X):
    type_ = type(X)
    if type_.__module__ == 'builtins':
        type_name = type_.__qualname__
    else:
        type_name = f'{type_.__module__}.{type_.__qualname__}'
    message = f'Unable to find the number of features from X of type {type_name}'
    if not hasattr(X, '__len__') and (not hasattr(X, 'shape')):
        if not hasattr(X, '__array__'):
            raise TypeError(message)
        X = np.asarray(X)
    if hasattr(X, 'shape'):
        if not hasattr(X.shape, '__len__') or len(X.shape) <= 1:
            message += f' with shape {X.shape}'
            raise TypeError(message)
        return X.shape[1]
    first_sample = X[0]
    if isinstance(first_sample, (str, bytes, dict)):
        message += f' where the samples are of type {type(first_sample).__qualname__}'
        raise TypeError(message)
    try:
        return len(first_sample)
    except Exception as err:
        raise TypeError(message) from err

def _check_X(self, X, force_all_finite=True):
    if not (hasattr(X, 'iloc') and getattr(X, 'ndim', 0) == 2):
        X_temp = check_array(X, dtype=None, force_all_finite=force_all_finite)
        if not hasattr(X, 'dtype') and np.issubdtype(X_temp.dtype, np.str_):
            X = check_array(X, dtype=object, force_all_finite=force_all_finite)
        else:
            X = X_temp
        needs_validation = False
    else:
        needs_validation = force_all_finite
    n_samples, n_features = X.shape
    X_columns = []
    for i in range(n_features):
        Xi = _safe_indexing(X, indices=i, axis=1)
        Xi = check_array(Xi, ensure_2d=False, dtype=None, force_all_finite=needs_validation)
        X_columns.append(Xi)
    return (X_columns, n_samples, n_features)

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



import numbers
from numbers import Integral, Real
import warnings
import numpy as np
from scipy import sparse
from ..base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from ..utils import check_array, is_scalar_nan, _safe_indexing
from ..utils.validation import check_is_fitted
from ..utils.validation import _check_feature_names_in
from ..utils._param_validation import Interval, StrOptions, Hidden
from ..utils._mask import _get_mask
from ..utils._encode import _encode, _check_unknown, _unique, _get_counts
__all__ = ['OneHotEncoder', 'OrdinalEncoder']

class OneHotEncoder(_BaseEncoder):
    _parameter_constraints: dict = {'categories': [StrOptions({'auto'}), list], 'drop': [StrOptions({'first', 'if_binary'}), 'array-like', None], 'dtype': 'no_validation', 'handle_unknown': [StrOptions({'error', 'ignore', 'infrequent_if_exist'})], 'max_categories': [Interval(Integral, 1, None, closed='left'), None], 'min_frequency': [Interval(Integral, 1, None, closed='left'), Interval(Real, 0, 1, closed='neither'), None], 'sparse': [Hidden(StrOptions({'deprecated'})), 'boolean'], 'sparse_output': ['boolean'], 'feature_name_combiner': [StrOptions({'concat'}), callable]}

    def __init__(self, *, categories='auto', drop=None, sparse='deprecated', sparse_output=True, dtype=np.float64, handle_unknown='error', min_frequency=None, max_categories=None, feature_name_combiner='concat'):
        self.categories = categories
        self.sparse = sparse
        self.sparse_output = sparse_output
        self.dtype = dtype
        self.handle_unknown = handle_unknown
        self.drop = drop
        self.min_frequency = min_frequency
        self.max_categories = max_categories
        self.feature_name_combiner = feature_name_combiner

    @property
    def infrequent_categories_(self):
        infrequent_indices = self._infrequent_indices
        return [None if indices is None else category[indices] for category, indices in zip(self.categories_, infrequent_indices)]

    def _check_infrequent_enabled(self):
        self._infrequent_enabled = self.max_categories is not None and self.max_categories >= 1 or self.min_frequency is not None

    def _map_drop_idx_to_infrequent(self, feature_idx, drop_idx):
        if not self._infrequent_enabled:
            return drop_idx
        default_to_infrequent = self._default_to_infrequent_mappings[feature_idx]
        if default_to_infrequent is None:
            return drop_idx
        infrequent_indices = self._infrequent_indices[feature_idx]
        if infrequent_indices is not None and drop_idx in infrequent_indices:
            categories = self.categories_[feature_idx]
            raise ValueError(f'Unable to drop category {categories[drop_idx]!r} from feature {feature_idx} because it is infrequent')
        return default_to_infrequent[drop_idx]

    def _set_drop_idx(self):
        if self.drop is None:
            drop_idx_after_grouping = None
        elif isinstance(self.drop, str):
            if self.drop == 'first':
                drop_idx_after_grouping = np.zeros(len(self.categories_), dtype=object)
            elif self.drop == 'if_binary':
                n_features_out_no_drop = [len(cat) for cat in self.categories_]
                if self._infrequent_enabled:
                    for i, infreq_idx in enumerate(self._infrequent_indices):
                        if infreq_idx is None:
                            continue
                        n_features_out_no_drop[i] -= infreq_idx.size - 1
                drop_idx_after_grouping = np.array([0 if n_features_out == 2 else None for n_features_out in n_features_out_no_drop], dtype=object)
        else:
            drop_array = np.asarray(self.drop, dtype=object)
            droplen = len(drop_array)
            if droplen != len(self.categories_):
                msg = '`drop` should have length equal to the number of features ({}), got {}'
                raise ValueError(msg.format(len(self.categories_), droplen))
            missing_drops = []
            drop_indices = []
            for feature_idx, (drop_val, cat_list) in enumerate(zip(drop_array, self.categories_)):
                if not is_scalar_nan(drop_val):
                    drop_idx = np.where(cat_list == drop_val)[0]
                    if drop_idx.size:
                        drop_indices.append(self._map_drop_idx_to_infrequent(feature_idx, drop_idx[0]))
                    else:
                        missing_drops.append((feature_idx, drop_val))
                    continue
                for cat_idx, cat in enumerate(cat_list):
                    if is_scalar_nan(cat):
                        drop_indices.append(self._map_drop_idx_to_infrequent(feature_idx, cat_idx))
                        break
                else:
                    missing_drops.append((feature_idx, drop_val))
            if any(missing_drops):
                msg = 'The following categories were supposed to be dropped, but were not found in the training data.\n{}'.format('\n'.join(['Category: {}, Feature: {}'.format(c, v) for c, v in missing_drops]))
                raise ValueError(msg)
            drop_idx_after_grouping = np.array(drop_indices, dtype=object)
        self._drop_idx_after_grouping = drop_idx_after_grouping
        if not self._infrequent_enabled or drop_idx_after_grouping is None:
            self.drop_idx_ = self._drop_idx_after_grouping
        else:
            drop_idx_ = []
            for feature_idx, drop_idx in enumerate(drop_idx_after_grouping):
                default_to_infrequent = self._default_to_infrequent_mappings[feature_idx]
                if drop_idx is None or default_to_infrequent is None:
                    orig_drop_idx = drop_idx
                else:
                    orig_drop_idx = np.flatnonzero(default_to_infrequent == drop_idx)[0]
                drop_idx_.append(orig_drop_idx)
            self.drop_idx_ = np.asarray(drop_idx_, dtype=object)

    def _identify_infrequent(self, category_count, n_samples, col_idx):
        if isinstance(self.min_frequency, numbers.Integral):
            infrequent_mask = category_count < self.min_frequency
        elif isinstance(self.min_frequency, numbers.Real):
            min_frequency_abs = n_samples * self.min_frequency
            infrequent_mask = category_count < min_frequency_abs
        else:
            infrequent_mask = np.zeros(category_count.shape[0], dtype=bool)
        n_current_features = category_count.size - infrequent_mask.sum() + 1
        if self.max_categories is not None and self.max_categories < n_current_features:
            smallest_levels = np.argsort(category_count, kind='mergesort')[:-self.max_categories + 1]
            infrequent_mask[smallest_levels] = True
        output = np.flatnonzero(infrequent_mask)
        return output if output.size > 0 else None

    def _fit_infrequent_category_mapping(self, n_samples, category_counts):
        self._infrequent_indices = [self._identify_infrequent(category_count, n_samples, col_idx) for col_idx, category_count in enumerate(category_counts)]
        self._default_to_infrequent_mappings = []
        for cats, infreq_idx in zip(self.categories_, self._infrequent_indices):
            if infreq_idx is None:
                self._default_to_infrequent_mappings.append(None)
                continue
            n_cats = len(cats)
            mapping = np.empty(n_cats, dtype=np.int64)
            n_infrequent_cats = infreq_idx.size
            n_frequent_cats = n_cats - n_infrequent_cats
            mapping[infreq_idx] = n_frequent_cats
            frequent_indices = np.setdiff1d(np.arange(n_cats), infreq_idx)
            mapping[frequent_indices] = np.arange(n_frequent_cats)
            self._default_to_infrequent_mappings.append(mapping)

    def _map_infrequent_categories(self, X_int, X_mask):
        if not self._infrequent_enabled:
            return
        for col_idx in range(X_int.shape[1]):
            infrequent_idx = self._infrequent_indices[col_idx]
            if infrequent_idx is None:
                continue
            X_int[~X_mask[:, col_idx], col_idx] = infrequent_idx[0]
            if self.handle_unknown == 'infrequent_if_exist':
                X_mask[:, col_idx] = True
        for i, mapping in enumerate(self._default_to_infrequent_mappings):
            if mapping is None:
                continue
            X_int[:, i] = np.take(mapping, X_int[:, i])

    def _compute_transformed_categories(self, i, remove_dropped=True):
        cats = self.categories_[i]
        if self._infrequent_enabled:
            infreq_map = self._default_to_infrequent_mappings[i]
            if infreq_map is not None:
                frequent_mask = infreq_map < infreq_map.max()
                infrequent_cat = 'infrequent_sklearn'
                cats = np.concatenate((cats[frequent_mask], np.array([infrequent_cat], dtype=object)))
        if remove_dropped:
            cats = self._remove_dropped_categories(cats, i)
        return cats

    def _remove_dropped_categories(self, categories, i):
        if self._drop_idx_after_grouping is not None and self._drop_idx_after_grouping[i] is not None:
            return np.delete(categories, self._drop_idx_after_grouping[i])
        return categories

    def _compute_n_features_outs(self):
        output = [len(cats) for cats in self.categories_]
        if self._drop_idx_after_grouping is not None:
            for i, drop_idx in enumerate(self._drop_idx_after_grouping):
                if drop_idx is not None:
                    output[i] -= 1
        if not self._infrequent_enabled:
            return output
        for i, infreq_idx in enumerate(self._infrequent_indices):
            if infreq_idx is None:
                continue
            output[i] -= infreq_idx.size - 1
        return output

    def fit(self, X, y=None):
        self._validate_params()
        if self.sparse != 'deprecated':
            warnings.warn('`sparse` was renamed to `sparse_output` in version 1.2 and will be removed in 1.4. `sparse_output` is ignored unless you leave `sparse` to its default value.', FutureWarning)
            self.sparse_output = self.sparse
        self._check_infrequent_enabled()
        fit_results = self._fit(X, handle_unknown=self.handle_unknown, force_all_finite='allow-nan', return_counts=self._infrequent_enabled)
        if self._infrequent_enabled:
            self._fit_infrequent_category_mapping(fit_results['n_samples'], fit_results['category_counts'])
        self._set_drop_idx()
        self._n_features_outs = self._compute_n_features_outs()
        return self

    def transform(self, X):
        check_is_fitted(self)
        warn_on_unknown = self.drop is not None and self.handle_unknown in {'ignore', 'infrequent_if_exist'}
        X_int, X_mask = self._transform(X, handle_unknown=self.handle_unknown, force_all_finite='allow-nan', warn_on_unknown=warn_on_unknown)
        self._map_infrequent_categories(X_int, X_mask)
        n_samples, n_features = X_int.shape
        if self._drop_idx_after_grouping is not None:
            to_drop = self._drop_idx_after_grouping.copy()
            keep_cells = X_int != to_drop
            for i, cats in enumerate(self.categories_):
                if to_drop[i] is None:
                    to_drop[i] = len(cats)
            to_drop = to_drop.reshape(1, -1)
            X_int[X_int > to_drop] -= 1
            X_mask &= keep_cells
        mask = X_mask.ravel()
        feature_indices = np.cumsum([0] + self._n_features_outs)
        indices = (X_int + feature_indices[:-1]).ravel()[mask]
        indptr = np.empty(n_samples + 1, dtype=int)
        indptr[0] = 0
        np.sum(X_mask, axis=1, out=indptr[1:], dtype=indptr.dtype)
        np.cumsum(indptr[1:], out=indptr[1:])
        data = np.ones(indptr[-1])
        out = sparse.csr_matrix((data, indices, indptr), shape=(n_samples, feature_indices[-1]), dtype=self.dtype)
        if not self.sparse_output:
            return out.toarray()
        else:
            return out

    def inverse_transform(self, X):
        check_is_fitted(self)
        X = check_array(X, accept_sparse='csr')
        n_samples, _ = X.shape
        n_features = len(self.categories_)
        n_features_out = np.sum(self._n_features_outs)
        msg = 'Shape of the passed X data is not correct. Expected {0} columns, got {1}.'
        if X.shape[1] != n_features_out:
            raise ValueError(msg.format(n_features_out, X.shape[1]))
        transformed_features = [self._compute_transformed_categories(i, remove_dropped=False) for i, _ in enumerate(self.categories_)]
        dt = np.result_type(*[cat.dtype for cat in transformed_features])
        X_tr = np.empty((n_samples, n_features), dtype=dt)
        j = 0
        found_unknown = {}
        if self._infrequent_enabled:
            infrequent_indices = self._infrequent_indices
        else:
            infrequent_indices = [None] * n_features
        for i in range(n_features):
            cats_wo_dropped = self._remove_dropped_categories(transformed_features[i], i)
            n_categories = cats_wo_dropped.shape[0]
            if n_categories == 0:
                X_tr[:, i] = self.categories_[i][self._drop_idx_after_grouping[i]]
                j += n_categories
                continue
            sub = X[:, j:j + n_categories]
            labels = np.asarray(sub.argmax(axis=1)).flatten()
            X_tr[:, i] = cats_wo_dropped[labels]
            if self.handle_unknown == 'ignore' or (self.handle_unknown == 'infrequent_if_exist' and infrequent_indices[i] is None):
                unknown = np.asarray(sub.sum(axis=1) == 0).flatten()
                if unknown.any():
                    if self._drop_idx_after_grouping is None or self._drop_idx_after_grouping[i] is None:
                        found_unknown[i] = unknown
                    else:
                        X_tr[unknown, i] = self.categories_[i][self._drop_idx_after_grouping[i]]
            else:
                dropped = np.asarray(sub.sum(axis=1) == 0).flatten()
                if dropped.any():
                    if self._drop_idx_after_grouping is None:
                        all_zero_samples = np.flatnonzero(dropped)
                        raise ValueError(f"Samples {all_zero_samples} can not be inverted when drop=None and handle_unknown='error' because they contain all zeros")
                    drop_idx = self._drop_idx_after_grouping[i]
                    X_tr[dropped, i] = transformed_features[i][drop_idx]
            j += n_categories
        if found_unknown:
            if X_tr.dtype != object:
                X_tr = X_tr.astype(object)
            for idx, mask in found_unknown.items():
                X_tr[mask, idx] = None
        return X_tr

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        input_features = _check_feature_names_in(self, input_features)
        cats = [self._compute_transformed_categories(i) for i, _ in enumerate(self.categories_)]
        name_combiner = self._check_get_feature_name_combiner()
        feature_names = []
        for i in range(len(cats)):
            names = [name_combiner(input_features[i], t) for t in cats[i]]
            feature_names.extend(names)
        return np.array(feature_names, dtype=object)

    def _check_get_feature_name_combiner(self):
        if self.feature_name_combiner == 'concat':
            return lambda feature, category: feature + '_' + str(category)
        else:
            dry_run_combiner = self.feature_name_combiner('feature', 'category')
            if not isinstance(dry_run_combiner, str):
                raise TypeError(f'When `feature_name_combiner` is a callable, it should return a Python string. Got {type(dry_run_combiner)} instead.')
            return self.feature_name_combiner