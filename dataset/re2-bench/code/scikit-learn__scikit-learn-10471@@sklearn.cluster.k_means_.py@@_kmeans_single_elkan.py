import warnings
import numpy as np
import scipy.sparse as sp
from ..base import BaseEstimator, ClusterMixin, TransformerMixin
from ..metrics.pairwise import euclidean_distances
from ..metrics.pairwise import pairwise_distances_argmin_min
from ..utils.extmath import row_norms, squared_norm, stable_cumsum
from ..utils.sparsefuncs_fast import assign_rows_csr
from ..utils.sparsefuncs import mean_variance_axis
from ..utils.validation import _num_samples
from ..utils import check_array
from ..utils import check_random_state
from ..utils import as_float_array
from ..utils import gen_batches
from ..utils.validation import check_is_fitted
from ..utils.validation import FLOAT_DTYPES
from ..externals.joblib import Parallel
from ..externals.joblib import delayed
from ..externals.six import string_types
from ..exceptions import ConvergenceWarning
from . import _k_means
from ._k_means_elkan import k_means_elkan



def _kmeans_single_elkan(X, n_clusters, max_iter=300, init='k-means++',
                         verbose=False, x_squared_norms=None,
                         random_state=None, tol=1e-4,
                         precompute_distances=True):
    if sp.issparse(X):
        raise TypeError("algorithm='elkan' not supported for sparse input X")
    random_state = check_random_state(random_state)
    if x_squared_norms is None:
        x_squared_norms = row_norms(X, squared=True)
    # init
    centers = _init_centroids(X, n_clusters, init, random_state=random_state,
                              x_squared_norms=x_squared_norms)
    centers = np.ascontiguousarray(centers)
    if verbose:
        print('Initialization complete')
    centers, labels, n_iter = k_means_elkan(X, n_clusters, centers, tol=tol,
                                            max_iter=max_iter, verbose=verbose)
    inertia = np.sum((X - centers[labels]) ** 2, dtype=np.float64)
    return labels, inertia, centers, n_iter
