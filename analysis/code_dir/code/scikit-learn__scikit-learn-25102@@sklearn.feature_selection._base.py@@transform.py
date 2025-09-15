import warnings
from abc import ABCMeta, abstractmethod
from operator import attrgetter
import numpy as np
from scipy.sparse import issparse, csc_matrix
from ..base import TransformerMixin
from ..cross_decomposition._pls import _PLS
from ..utils import check_array, safe_sqr
from ..utils._tags import _safe_tags
from ..utils import _safe_indexing
from ..utils._set_output import _get_output_config
from ..utils.validation import _check_feature_names_in, check_is_fitted

class SelectorMixin(TransformerMixin, metaclass=ABCMeta):

    def get_support(self, indices=False):
        mask = self._get_support_mask()
        return mask if not indices else np.where(mask)[0]

    @abstractmethod
    def _get_support_mask(self):

    def transform(self, X):
        output_config_dense = _get_output_config('transform', estimator=self)['dense']
        preserve_X = hasattr(X, 'iloc') and output_config_dense == 'pandas'
        X = self._validate_data(X, dtype=None, accept_sparse='csr', force_all_finite=not _safe_tags(self, key='allow_nan'), cast_to_ndarray=not preserve_X, reset=False)
        return self._transform(X)

    def _transform(self, X):
        mask = self.get_support()
        if not mask.any():
            warnings.warn('No features were selected: either the data is too noisy or the selection test too strict.', UserWarning)
            if hasattr(X, 'iloc'):
                return X.iloc[:, :0]
            return np.empty(0, dtype=X.dtype).reshape((X.shape[0], 0))
        return _safe_indexing(X, mask, axis=1)

    def inverse_transform(self, X):
        if issparse(X):
            X = X.tocsc()
            it = self.inverse_transform(np.diff(X.indptr).reshape(1, -1))
            col_nonzeros = it.ravel()
            indptr = np.concatenate([[0], np.cumsum(col_nonzeros)])
            Xt = csc_matrix((X.data, X.indices, indptr), shape=(X.shape[0], len(indptr) - 1), dtype=X.dtype)
            return Xt
        support = self.get_support()
        X = check_array(X, dtype=None)
        if support.sum() != X.shape[1]:
            raise ValueError('X has a different shape than during fitting.')
        if X.ndim == 1:
            X = X[None, :]
        Xt = np.zeros((X.shape[0], support.size), dtype=X.dtype)
        Xt[:, support] = X
        return Xt

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        input_features = _check_feature_names_in(self, input_features)
        return input_features[self.get_support()]