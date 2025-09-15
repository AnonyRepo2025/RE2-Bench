import warnings
from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.linalg import pinv2, svd
from scipy.sparse.linalg import svds
from ..base import BaseEstimator, RegressorMixin, TransformerMixin
from ..utils import check_array, check_consistent_length
from ..utils.extmath import svd_flip
from ..utils.validation import check_is_fitted, FLOAT_DTYPES
from ..exceptions import ConvergenceWarning
from ..externals import six
__all__ = ['PLSCanonical', 'PLSRegression', 'PLSSVD']

def _nipals_twoblocks_inner_loop(X, Y, mode='A', max_iter=500, tol=1e-06, norm_y_weights=False):
    y_score = Y[:, [0]]
    x_weights_old = 0
    ite = 1
    X_pinv = Y_pinv = None
    eps = np.finfo(X.dtype).eps
    while True:
        if mode == 'B':
            if X_pinv is None:
                X_pinv = pinv2(X, check_finite=False)
            x_weights = np.dot(X_pinv, y_score)
        else:
            x_weights = np.dot(X.T, y_score) / np.dot(y_score.T, y_score)
        if np.dot(x_weights.T, x_weights) < eps:
            x_weights += eps
        x_weights /= np.sqrt(np.dot(x_weights.T, x_weights)) + eps
        x_score = np.dot(X, x_weights)
        if mode == 'B':
            if Y_pinv is None:
                Y_pinv = pinv2(Y, check_finite=False)
            y_weights = np.dot(Y_pinv, x_score)
        else:
            y_weights = np.dot(Y.T, x_score) / np.dot(x_score.T, x_score)
        if norm_y_weights:
            y_weights /= np.sqrt(np.dot(y_weights.T, y_weights)) + eps
        y_score = np.dot(Y, y_weights) / (np.dot(y_weights.T, y_weights) + eps)
        x_weights_diff = x_weights - x_weights_old
        if np.dot(x_weights_diff.T, x_weights_diff) < tol or Y.shape[1] == 1:
            break
        if ite == max_iter:
            warnings.warn('Maximum number of iterations reached', ConvergenceWarning)
            break
        x_weights_old = x_weights
        ite += 1
    return (x_weights, y_weights, ite)