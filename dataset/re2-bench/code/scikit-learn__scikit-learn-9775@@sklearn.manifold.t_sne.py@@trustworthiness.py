from __future__ import division
import warnings
from time import time
import numpy as np
from scipy import linalg
import scipy.sparse as sp
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.sparse import csr_matrix
from ..neighbors import NearestNeighbors
from ..base import BaseEstimator
from ..utils import check_array
from ..utils import check_random_state
from ..decomposition import PCA
from ..metrics.pairwise import pairwise_distances
from . import _utils
from . import _barnes_hut_tsne
from ..externals.six import string_types
from ..utils import deprecated

MACHINE_EPSILON = np.finfo(np.double).eps

def trustworthiness(X, X_embedded, n_neighbors=5,
                    precomputed=False, metric='euclidean'):
    r"""Expresses to what extent the local structure is retained.

    The trustworthiness is within [0, 1]. It is defined as

    .. math::

        T(k) = 1 - \frac{2}{nk (2n - 3k - 1)} \sum^n_{i=1}
            \sum_{j \in \mathcal{N}_{i}^{k}} \max(0, (r(i, j) - k))

    where for each sample i, :math:`\mathcal{N}_{i}^{k}` are its k nearest
    neighbors in the output space, and every sample j is its :math:`r(i, j)`-th
    nearest neighbor in the input space. In other words, any unexpected nearest
    neighbors in the output space are penalised in proportion to their rank in
    the input space.

    * "Neighborhood Preservation in Nonlinear Projection Methods: An
      Experimental Study"
      J. Venna, S. Kaski
    * "Learning a Parametric Embedding by Preserving Local Structure"
      L.J.P. van der Maaten

    Parameters
    ----------
    X : array, shape (n_samples, n_features) or (n_samples, n_samples)
        If the metric is 'precomputed' X must be a square distance
        matrix. Otherwise it contains a sample per row.

    X_embedded : array, shape (n_samples, n_components)
        Embedding of the training data in low-dimensional space.

    n_neighbors : int, optional (default: 5)
        Number of neighbors k that will be considered.

    precomputed : bool, optional (default: False)
        Set this flag if X is a precomputed square distance matrix.

        ..deprecated:: 0.20
            ``precomputed`` has been deprecated in version 0.20 and will be
            removed in version 0.22. Use ``metric`` instead.

    metric : string, or callable, optional, default 'euclidean'
        Which metric to use for computing pairwise distances between samples
        from the original input space. If metric is 'precomputed', X must be a
        matrix of pairwise distances or squared distances. Otherwise, see the
        documentation of argument metric in sklearn.pairwise.pairwise_distances
        for a list of available metrics.

    Returns
    -------
    trustworthiness : float
        Trustworthiness of the low-dimensional embedding.
    """
    if precomputed:
        warnings.warn("The flag 'precomputed' has been deprecated in version "
                      "0.20 and will be removed in 0.22. See 'metric' "
                      "parameter instead.", DeprecationWarning)
        metric = 'precomputed'
    dist_X = pairwise_distances(X, metric=metric)
    ind_X = np.argsort(dist_X, axis=1)
    ind_X_embedded = NearestNeighbors(n_neighbors).fit(X_embedded).kneighbors(
        return_distance=False)

    n_samples = X.shape[0]
    t = 0.0
    ranks = np.zeros(n_neighbors)
    for i in range(n_samples):
        for j in range(n_neighbors):
            ranks[j] = np.where(ind_X[i] == ind_X_embedded[i, j])[0][0]
        ranks -= n_neighbors
        t += np.sum(ranks[ranks > 0])
    t = 1.0 - t * (2.0 / (n_samples * n_neighbors *
                          (2.0 * n_samples - 3.0 * n_neighbors - 1.0)))
    return t
