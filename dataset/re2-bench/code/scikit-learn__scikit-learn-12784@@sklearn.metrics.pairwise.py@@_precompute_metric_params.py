import itertools
from functools import partial
import warnings
import numpy as np
from scipy.spatial import distance
from scipy.sparse import csr_matrix
from scipy.sparse import issparse
from ..utils.validation import _num_samples
from ..utils import check_array
from ..utils import gen_even_slices
from ..utils import gen_batches, get_chunk_n_rows
from ..utils.extmath import row_norms, safe_sparse_dot
from ..preprocessing import normalize
from ..utils._joblib import Parallel
from ..utils._joblib import delayed
from ..utils._joblib import effective_n_jobs
from .pairwise_fast import _chi2_kernel_fast, _sparse_manhattan
from ..gaussian_process.kernels import Kernel as GPKernel

PAIRED_DISTANCES = {
    'cosine': paired_cosine_distances,
    'euclidean': paired_euclidean_distances,
    'l2': paired_euclidean_distances,
    'l1': paired_manhattan_distances,
    'manhattan': paired_manhattan_distances,
    'cityblock': paired_manhattan_distances}
PAIRWISE_DISTANCE_FUNCTIONS = {
    # If updating this dictionary, update the doc in both distance_metrics()
    # and also in pairwise_distances()!
    'cityblock': manhattan_distances,
    'cosine': cosine_distances,
    'euclidean': euclidean_distances,
    'l2': euclidean_distances,
    'l1': manhattan_distances,
    'manhattan': manhattan_distances,
    'precomputed': None,  # HACK: precomputed is always allowed, never called
}
_VALID_METRICS = ['euclidean', 'l2', 'l1', 'manhattan', 'cityblock',
                  'braycurtis', 'canberra', 'chebyshev', 'correlation',
                  'cosine', 'dice', 'hamming', 'jaccard', 'kulsinski',
                  'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
                  'russellrao', 'seuclidean', 'sokalmichener',
                  'sokalsneath', 'sqeuclidean', 'yule', "wminkowski"]
PAIRWISE_BOOLEAN_FUNCTIONS = [
    'dice',
    'jaccard',
    'kulsinski',
    'matching',
    'rogerstanimoto',
    'russellrao',
    'sokalmichener',
    'sokalsneath',
    'yule',
]
PAIRWISE_KERNEL_FUNCTIONS = {
    # If updating this dictionary, update the doc in both distance_metrics()
    # and also in pairwise_distances()!
    'additive_chi2': additive_chi2_kernel,
    'chi2': chi2_kernel,
    'linear': linear_kernel,
    'polynomial': polynomial_kernel,
    'poly': polynomial_kernel,
    'rbf': rbf_kernel,
    'laplacian': laplacian_kernel,
    'sigmoid': sigmoid_kernel,
    'cosine': cosine_similarity, }
KERNEL_PARAMS = {
    "additive_chi2": (),
    "chi2": frozenset(["gamma"]),
    "cosine": (),
    "linear": (),
    "poly": frozenset(["gamma", "degree", "coef0"]),
    "polynomial": frozenset(["gamma", "degree", "coef0"]),
    "rbf": frozenset(["gamma"]),
    "laplacian": frozenset(["gamma"]),
    "sigmoid": frozenset(["gamma", "coef0"]),
}

def _precompute_metric_params(X, Y, metric=None, **kwds):
    """Precompute data-derived metric parameters if not provided
    """
    if metric == "seuclidean" and 'V' not in kwds:
        if X is Y:
            V = np.var(X, axis=0, ddof=1)
        else:
            V = np.var(np.vstack([X, Y]), axis=0, ddof=1)
        return {'V': V}
    if metric == "mahalanobis" and 'VI' not in kwds:
        if X is Y:
            VI = np.linalg.inv(np.cov(X.T)).T
        else:
            VI = np.linalg.inv(np.cov(np.vstack([X, Y]).T)).T
        return {'VI': VI}
    return {}
