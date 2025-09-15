from abc import ABCMeta, abstractmethod
from warnings import warn
import numpy as np
from scipy.sparse import issparse, csc_matrix
from ..base import TransformerMixin
from ..utils import check_array, safe_mask

class SelectorMixin(TransformerMixin, metaclass=ABCMeta):

    def get_support(self, indices=False):
        mask = self._get_support_mask()
        return mask if not indices else np.where(mask)[0]

    @abstractmethod
    def _get_support_mask(self):

    def transform(self, X):
        tags = self._get_tags()
        X = check_array(X, dtype=None, accept_sparse='csr', force_all_finite=not tags.get('allow_nan', True))
        mask = self.get_support()
        if not mask.any():
            warn('No features were selected: either the data is too noisy or the selection test too strict.', UserWarning)
            return np.empty(0).reshape((X.shape[0], 0))
        if len(mask) != X.shape[1]:
            raise ValueError('X has a different shape than during fitting.')
        return X[:, safe_mask(X, mask)]

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