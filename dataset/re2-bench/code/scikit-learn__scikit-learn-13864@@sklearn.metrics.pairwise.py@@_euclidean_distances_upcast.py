import itertools
from functools import partial
import warnings
import numpy as np
from scipy.spatial import distance
from scipy.sparse import csr_matrix
from scipy.sparse import issparse
from ..utils.validation import _num_samples
from ..utils.validation import check_non_negative
from ..utils import check_array
from ..utils import gen_even_slices
from ..utils import gen_batches, get_chunk_n_rows
from ..utils.extmath import row_norms, safe_sparse_dot
from ..preprocessing import normalize
from ..utils._joblib import Parallel
from ..utils._joblib import delayed
from ..utils._joblib import effective_n_jobs
from .pairwise_fast import _chi2_kernel_fast, _sparse_manhattan
from ..exceptions import DataConversionWarning
from sklearn.neighbors import DistanceMetric
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
    'haversine': haversine_distances,
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
                  'sokalsneath', 'sqeuclidean', 'yule', 'wminkowski',
                  'haversine']
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

def _euclidean_distances_upcast(X, XX=None, Y=None, YY=None):
    """Euclidean distances between X and Y

    Assumes X and Y have float32 dtype.
    Assumes XX and YY have float64 dtype or are None.

    X and Y are upcast to float64 by chunks, which size is chosen to limit
    memory increase by approximately 10% (at least 10MiB).
    """
    n_samples_X = X.shape[0]
    n_samples_Y = Y.shape[0]
    n_features = X.shape[1]

    distances = np.empty((n_samples_X, n_samples_Y), dtype=np.float32)

    x_density = X.nnz / np.prod(X.shape) if issparse(X) else 1
    y_density = Y.nnz / np.prod(Y.shape) if issparse(Y) else 1

    # Allow 10% more memory than X, Y and the distance matrix take (at least
    # 10MiB)
    maxmem = max(
        ((x_density * n_samples_X + y_density * n_samples_Y) * n_features
         + (x_density * n_samples_X * y_density * n_samples_Y)) / 10,
        10 * 2 ** 17)

    # The increase amount of memory in 8-byte blocks is:
    # - x_density * batch_size * n_features (copy of chunk of X)
    # - y_density * batch_size * n_features (copy of chunk of Y)
    # - batch_size * batch_size (chunk of distance matrix)
    # Hence x² + (xd+yd)kx = M, where x=batch_size, k=n_features, M=maxmem
    #                                 xd=x_density and yd=y_density
    tmp = (x_density + y_density) * n_features
    batch_size = (-tmp + np.sqrt(tmp ** 2 + 4 * maxmem)) / 2
    batch_size = max(int(batch_size), 1)

    x_batches = gen_batches(X.shape[0], batch_size)
    y_batches = gen_batches(Y.shape[0], batch_size)

    for i, x_slice in enumerate(x_batches):
        X_chunk = X[x_slice].astype(np.float64)
        if XX is None:
            XX_chunk = row_norms(X_chunk, squared=True)[:, np.newaxis]
        else:
            XX_chunk = XX[x_slice]

        for j, y_slice in enumerate(y_batches):
            if X is Y and j < i:
                # when X is Y the distance matrix is symmetric so we only need
                # to compute half of it.
                d = distances[y_slice, x_slice].T

            else:
                Y_chunk = Y[y_slice].astype(np.float64)
                if YY is None:
                    YY_chunk = row_norms(Y_chunk, squared=True)[np.newaxis, :]
                else:
                    YY_chunk = YY[:, y_slice]

                d = -2 * safe_sparse_dot(X_chunk, Y_chunk.T, dense_output=True)
                d += XX_chunk
                d += YY_chunk

            distances[x_slice, y_slice] = d.astype(np.float32, copy=False)

    return distances
