import warnings
import numpy as np
from scipy import linalg
from ..base import BaseEstimator, TransformerMixin
from ..exceptions import ConvergenceWarning
from ..externals import six
from ..externals.six import moves
from ..externals.six import string_types
from ..utils import check_array, as_float_array, check_random_state
from ..utils.validation import check_is_fitted
from ..utils.validation import FLOAT_DTYPES

__all__ = ['fastica', 'FastICA']

def _ica_par(X, tol, g, fun_args, max_iter, w_init):
    """Parallel FastICA.

    Used internally by FastICA --main loop

    """
    W = _sym_decorrelation(w_init)
    del w_init
    p_ = float(X.shape[1])
    for ii in moves.xrange(max_iter):
        gwtx, g_wtx = g(np.dot(W, X), fun_args)
        W1 = _sym_decorrelation(np.dot(gwtx, X.T) / p_
                                - g_wtx[:, np.newaxis] * W)
        del gwtx, g_wtx
        # builtin max, abs are faster than numpy counter parts.
        lim = max(abs(abs(np.diag(np.dot(W1, W.T))) - 1))
        W = W1
        if lim < tol:
            break
    else:
        warnings.warn('FastICA did not converge. Consider increasing '
                      'tolerance or the maximum number of iterations.',
                      ConvergenceWarning)

    return W, ii + 1
