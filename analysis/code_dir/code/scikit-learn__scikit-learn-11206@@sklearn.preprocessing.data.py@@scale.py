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