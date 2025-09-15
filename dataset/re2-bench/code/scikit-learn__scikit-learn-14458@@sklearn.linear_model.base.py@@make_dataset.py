from abc import ABCMeta, abstractmethod
import numbers
import warnings
import numpy as np
import scipy.sparse as sp
from scipy import linalg
from scipy import sparse
from scipy.special import expit
from joblib import Parallel, delayed
from ..base import (BaseEstimator, ClassifierMixin, RegressorMixin,
                    MultiOutputMixin)
from ..utils import check_array, check_X_y
from ..utils.validation import FLOAT_DTYPES
from ..utils import check_random_state
from ..utils.extmath import safe_sparse_dot
from ..utils.sparsefuncs import mean_variance_axis, inplace_column_scale
from ..utils.fixes import sparse_lsqr
from ..utils.seq_dataset import ArrayDataset32, CSRDataset32
from ..utils.seq_dataset import ArrayDataset64, CSRDataset64
from ..utils.validation import check_is_fitted
from ..preprocessing.data import normalize as f_normalize

SPARSE_INTERCEPT_DECAY = 0.01

def make_dataset(X, y, sample_weight, random_state=None):
    """Create ``Dataset`` abstraction for sparse and dense inputs.

    This also returns the ``intercept_decay`` which is different
    for sparse datasets.

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
        Training data

    y : array_like, shape (n_samples, )
        Target values.

    sample_weight : numpy array of shape (n_samples,)
        The weight of each sample

    random_state : int, RandomState instance or None (default)
        Determines random number generation for dataset shuffling and noise.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    dataset
        The ``Dataset`` abstraction
    intercept_decay
        The intercept decay
    """

    rng = check_random_state(random_state)
    # seed should never be 0 in SequentialDataset64
    seed = rng.randint(1, np.iinfo(np.int32).max)

    if X.dtype == np.float32:
        CSRData = CSRDataset32
        ArrayData = ArrayDataset32
    else:
        CSRData = CSRDataset64
        ArrayData = ArrayDataset64

    if sp.issparse(X):
        dataset = CSRData(X.data, X.indptr, X.indices, y, sample_weight,
                          seed=seed)
        intercept_decay = SPARSE_INTERCEPT_DECAY
    else:
        X = np.ascontiguousarray(X)
        dataset = ArrayData(X, y, sample_weight, seed=seed)
        intercept_decay = 1.0

    return dataset, intercept_decay
