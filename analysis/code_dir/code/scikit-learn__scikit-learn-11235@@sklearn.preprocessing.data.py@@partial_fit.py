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