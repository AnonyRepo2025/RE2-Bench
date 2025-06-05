import time
import sys
import itertools
from math import ceil
import numpy as np
from scipy import linalg
from joblib import Parallel, delayed, effective_n_jobs
from ..base import BaseEstimator, TransformerMixin
from ..utils import (check_array, check_random_state, gen_even_slices,
                     gen_batches)
from ..utils.extmath import randomized_svd, row_norms
from ..utils.validation import check_is_fitted
from ..linear_model import Lasso, orthogonal_mp_gram, LassoLars, Lars



def dict_learning_online(X, n_components=2, alpha=1, n_iter=100,
                         return_code=True, dict_init=None, callback=None,
                         batch_size=3, verbose=False, shuffle=True,
                         n_jobs=None, method='lars', iter_offset=0,
                         random_state=None, return_inner_stats=False,
                         inner_stats=None, return_n_iter=False,
                         positive_dict=False, positive_code=False,
                         method_max_iter=1000):
    if n_components is None:
        n_components = X.shape[1]

    if method not in ('lars', 'cd'):
        raise ValueError('Coding method not supported as a fit algorithm.')

    _check_positive_coding(method, positive_code)

    method = 'lasso_' + method

    t0 = time.time()
    n_samples, n_features = X.shape
    alpha = float(alpha)
    random_state = check_random_state(random_state)

    if dict_init is not None:
        dictionary = dict_init
    else:
        _, S, dictionary = randomized_svd(X, n_components,
                                          random_state=random_state)
        dictionary = S[:, np.newaxis] * dictionary
    r = len(dictionary)
    if n_components <= r:
        dictionary = dictionary[:n_components, :]
    else:
        dictionary = np.r_[dictionary,
                           np.zeros((n_components - r, dictionary.shape[1]))]

    if verbose == 1:
        print('[dict_learning]', end=' ')

    if shuffle:
        X_train = X.copy()
        random_state.shuffle(X_train)
    else:
        X_train = X

    dictionary = check_array(dictionary.T, order='F', dtype=np.float64,
                             copy=False)
    dictionary = np.require(dictionary, requirements='W')

    X_train = check_array(X_train, order='C', dtype=np.float64, copy=False)

    batches = gen_batches(n_samples, batch_size)
    batches = itertools.cycle(batches)

    if inner_stats is None:
        A = np.zeros((n_components, n_components))
        B = np.zeros((n_features, n_components))
    else:
        A = inner_stats[0].copy()
        B = inner_stats[1].copy()

    ii = iter_offset - 1

    for ii, batch in zip(range(iter_offset, iter_offset + n_iter), batches):
        this_X = X_train[batch]
        dt = (time.time() - t0)
        if verbose == 1:
            sys.stdout.write(".")
            sys.stdout.flush()
        elif verbose:
            if verbose > 10 or ii % ceil(100. / verbose) == 0:
                print("Iteration % 3i (elapsed time: % 3is, % 4.1fmn)"
                      % (ii, dt, dt / 60))

        this_code = sparse_encode(this_X, dictionary.T, algorithm=method,
                                  alpha=alpha, n_jobs=n_jobs,
                                  check_input=False,
                                  positive=positive_code,
                                  max_iter=method_max_iter, verbose=verbose).T

        if ii < batch_size - 1:
            theta = float((ii + 1) * batch_size)
        else:
            theta = float(batch_size ** 2 + ii + 1 - batch_size)
        beta = (theta + 1 - batch_size) / (theta + 1)

        A *= beta
        A += np.dot(this_code, this_code.T)
        B *= beta
        B += np.dot(this_X.T, this_code.T)

        dictionary = _update_dict(dictionary, B, A, verbose=verbose,
                                  random_state=random_state,
                                  positive=positive_dict)
        if callback is not None:
            callback(locals())

    if return_inner_stats:
        if return_n_iter:
            return dictionary.T, (A, B), ii - iter_offset + 1
        else:
            return dictionary.T, (A, B)
    if return_code:
        if verbose > 1:
            print('Learning code...', end=' ')
        elif verbose == 1:
            print('|', end=' ')
        code = sparse_encode(X, dictionary.T, algorithm=method, alpha=alpha,
                             n_jobs=n_jobs, check_input=False,
                             positive=positive_code, max_iter=method_max_iter,
                             verbose=verbose)
        if verbose > 1:
            dt = (time.time() - t0)
            print('done (total time: % 3is, % 4.1fmn)' % (dt, dt / 60))
        if return_n_iter:
            return code, dictionary.T, ii - iter_offset + 1
        else:
            return code, dictionary.T

    if return_n_iter:
        return dictionary.T, ii - iter_offset + 1
    else:
        return dictionary.T
