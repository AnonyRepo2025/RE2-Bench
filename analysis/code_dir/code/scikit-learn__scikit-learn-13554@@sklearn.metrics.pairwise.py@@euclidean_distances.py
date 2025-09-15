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
PAIRED_DISTANCES = {'cosine': paired_cosine_distances, 'euclidean': paired_euclidean_distances, 'l2': paired_euclidean_distances, 'l1': paired_manhattan_distances, 'manhattan': paired_manhattan_distances, 'cityblock': paired_manhattan_distances}
PAIRWISE_DISTANCE_FUNCTIONS = {'cityblock': manhattan_distances, 'cosine': cosine_distances, 'euclidean': euclidean_distances, 'haversine': haversine_distances, 'l2': euclidean_distances, 'l1': manhattan_distances, 'manhattan': manhattan_distances, 'precomputed': None}
_VALID_METRICS = ['euclidean', 'l2', 'l1', 'manhattan', 'cityblock', 'braycurtis', 'canberra', 'chebyshev', 'correlation', 'cosine', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule', 'wminkowski', 'haversine']
PAIRWISE_BOOLEAN_FUNCTIONS = ['dice', 'jaccard', 'kulsinski', 'matching', 'rogerstanimoto', 'russellrao', 'sokalmichener', 'sokalsneath', 'yule']
PAIRWISE_KERNEL_FUNCTIONS = {'additive_chi2': additive_chi2_kernel, 'chi2': chi2_kernel, 'linear': linear_kernel, 'polynomial': polynomial_kernel, 'poly': polynomial_kernel, 'rbf': rbf_kernel, 'laplacian': laplacian_kernel, 'sigmoid': sigmoid_kernel, 'cosine': cosine_similarity}
KERNEL_PARAMS = {'additive_chi2': (), 'chi2': frozenset(['gamma']), 'cosine': (), 'linear': (), 'poly': frozenset(['gamma', 'degree', 'coef0']), 'polynomial': frozenset(['gamma', 'degree', 'coef0']), 'rbf': frozenset(['gamma']), 'laplacian': frozenset(['gamma']), 'sigmoid': frozenset(['gamma', 'coef0'])}

def euclidean_distances(X, Y=None, Y_norm_squared=None, squared=False, X_norm_squared=None):
    X, Y = check_pairwise_arrays(X, Y)
    if X_norm_squared is not None:
        XX = check_array(X_norm_squared)
        if XX.shape == (1, X.shape[0]):
            XX = XX.T
        elif XX.shape != (X.shape[0], 1):
            raise ValueError('Incompatible dimensions for X and X_norm_squared')
        if XX.dtype == np.float32:
            XX = None
    elif X.dtype == np.float32:
        XX = None
    else:
        XX = row_norms(X, squared=True)[:, np.newaxis]
    if X is Y and XX is not None:
        YY = XX.T
    elif Y_norm_squared is not None:
        YY = np.atleast_2d(Y_norm_squared)
        if YY.shape != (1, Y.shape[0]):
            raise ValueError('Incompatible dimensions for Y and Y_norm_squared')
        if YY.dtype == np.float32:
            YY = None
    elif Y.dtype == np.float32:
        YY = None
    else:
        YY = row_norms(Y, squared=True)[np.newaxis, :]
    if X.dtype == np.float32:
        distances = _euclidean_distances_upcast(X, XX, Y, YY)
    else:
        distances = -2 * safe_sparse_dot(X, Y.T, dense_output=True)
        distances += XX
        distances += YY
    np.maximum(distances, 0, out=distances)
    if X is Y:
        np.fill_diagonal(distances, 0)
    return distances if squared else np.sqrt(distances, out=distances)