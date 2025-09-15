

import warnings
import numpy as np
import scipy.sparse as sp
from scipy.linalg import svd
from .base import BaseEstimator
from .base import TransformerMixin
from .utils import check_array, check_random_state, as_float_array
from .utils.extmath import safe_sparse_dot
from .utils.validation import check_is_fitted
from .metrics.pairwise import pairwise_kernels, KERNEL_PARAMS

class Nystroem(BaseEstimator, TransformerMixin):

    def __init__(self, kernel='rbf', gamma=None, coef0=None, degree=None, kernel_params=None, n_components=100, random_state=None):
        self.kernel = kernel
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.kernel_params = kernel_params
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        X = check_array(X, accept_sparse='csr')
        rnd = check_random_state(self.random_state)
        n_samples = X.shape[0]
        if self.n_components > n_samples:
            n_components = n_samples
            warnings.warn('n_components > n_samples. This is not possible.\nn_components was set to n_samples, which results in inefficient evaluation of the full kernel.')
        else:
            n_components = self.n_components
        n_components = min(n_samples, n_components)
        inds = rnd.permutation(n_samples)
        basis_inds = inds[:n_components]
        basis = X[basis_inds]
        basis_kernel = pairwise_kernels(basis, metric=self.kernel, filter_params=True, **self._get_kernel_params())
        U, S, V = svd(basis_kernel)
        S = np.maximum(S, 1e-12)
        self.normalization_ = np.dot(U / np.sqrt(S), V)
        self.components_ = basis
        self.component_indices_ = inds
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = check_array(X, accept_sparse='csr')
        kernel_params = self._get_kernel_params()
        embedded = pairwise_kernels(X, self.components_, metric=self.kernel, filter_params=True, **kernel_params)
        return np.dot(embedded, self.normalization_.T)

    def _get_kernel_params(self):
        params = self.kernel_params
        if params is None:
            params = {}
        if not callable(self.kernel) and self.kernel != 'precomputed':
            for param in KERNEL_PARAMS[self.kernel]:
                if getattr(self, param) is not None:
                    params[param] = getattr(self, param)
        elif self.gamma is not None or self.coef0 is not None or self.degree is not None:
            raise ValueError("Don't pass gamma, coef0 or degree to Nystroem if using a callable or precomputed kernel")
        return params