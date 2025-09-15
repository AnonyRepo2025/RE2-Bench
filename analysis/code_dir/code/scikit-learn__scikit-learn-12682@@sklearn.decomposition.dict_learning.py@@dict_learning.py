import time
import sys
import itertools
from math import ceil
import numpy as np
from scipy import linalg
from joblib import Parallel, delayed, effective_n_jobs
from ..base import BaseEstimator, TransformerMixin
from ..utils import check_array, check_random_state, gen_even_slices, gen_batches
from ..utils.extmath import randomized_svd, row_norms
from ..utils.validation import check_is_fitted
from ..linear_model import Lasso, orthogonal_mp_gram, LassoLars, Lars

def dict_learning(X, n_components, alpha, max_iter=100, tol=1e-08, method='lars', n_jobs=None, dict_init=None, code_init=None, callback=None, verbose=False, random_state=None, return_n_iter=False, positive_dict=False, positive_code=False, method_max_iter=1000):
    if method not in ('lars', 'cd'):
        raise ValueError('Coding method %r not supported as a fit algorithm.' % method)
    _check_positive_coding(method, positive_code)
    method = 'lasso_' + method
    t0 = time.time()
    alpha = float(alpha)
    random_state = check_random_state(random_state)
    if code_init is not None and dict_init is not None:
        code = np.array(code_init, order='F')
        dictionary = dict_init
    else:
        code, S, dictionary = linalg.svd(X, full_matrices=False)
        dictionary = S[:, np.newaxis] * dictionary
    r = len(dictionary)
    if n_components <= r:
        code = code[:, :n_components]
        dictionary = dictionary[:n_components, :]
    else:
        code = np.c_[code, np.zeros((len(code), n_components - r))]
        dictionary = np.r_[dictionary, np.zeros((n_components - r, dictionary.shape[1]))]
    dictionary = np.array(dictionary, order='F')
    residuals = 0
    errors = []
    current_cost = np.nan
    if verbose == 1:
        print('[dict_learning]', end=' ')
    ii = -1
    for ii in range(max_iter):
        dt = time.time() - t0
        if verbose == 1:
            sys.stdout.write('.')
            sys.stdout.flush()
        elif verbose:
            print('Iteration % 3i (elapsed time: % 3is, % 4.1fmn, current cost % 7.3f)' % (ii, dt, dt / 60, current_cost))
        code = sparse_encode(X, dictionary, algorithm=method, alpha=alpha, init=code, n_jobs=n_jobs, positive=positive_code, max_iter=method_max_iter, verbose=verbose)
        dictionary, residuals = _update_dict(dictionary.T, X.T, code.T, verbose=verbose, return_r2=True, random_state=random_state, positive=positive_dict)
        dictionary = dictionary.T
        current_cost = 0.5 * residuals + alpha * np.sum(np.abs(code))
        errors.append(current_cost)
        if ii > 0:
            dE = errors[-2] - errors[-1]
            if dE < tol * errors[-1]:
                if verbose == 1:
                    print('')
                elif verbose:
                    print('--- Convergence reached after %d iterations' % ii)
                break
        if ii % 5 == 0 and callback is not None:
            callback(locals())
    if return_n_iter:
        return (code, dictionary, errors, ii + 1)
    else:
        return (code, dictionary, errors)