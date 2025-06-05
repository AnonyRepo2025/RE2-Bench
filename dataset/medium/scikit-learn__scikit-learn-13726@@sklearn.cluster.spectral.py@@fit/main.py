import warnings
import numpy as np
from ..base import BaseEstimator, ClusterMixin
from ..utils import check_random_state, as_float_array
from ..utils.validation import check_array
from ..metrics.pairwise import pairwise_kernels
from ..neighbors import kneighbors_graph
from ..manifold import spectral_embedding
from .k_means_ import k_means
from scipy.sparse import csc_matrix
from scipy.linalg import LinAlgError



class SpectralClustering(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=8, eigen_solver=None, n_components=None,
                 random_state=None, n_init=10, gamma=1., affinity='rbf',
                 n_neighbors=10, eigen_tol=0.0, assign_labels='kmeans',
                 degree=3, coef0=1, kernel_params=None, n_jobs=None):
        self.n_clusters = n_clusters
        self.eigen_solver = eigen_solver
        self.n_components = n_components
        self.random_state = random_state
        self.n_init = n_init
        self.gamma = gamma
        self.affinity = affinity
        self.n_neighbors = n_neighbors
        self.eigen_tol = eigen_tol
        self.assign_labels = assign_labels
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        X = check_array(X, accept_sparse=['csr', 'csc', 'coo'],
                        dtype=np.float64, ensure_min_samples=2)
        if X.shape[0] == X.shape[1] and self.affinity != "precomputed":
            warnings.warn("The spectral clustering API has changed. ``fit``"
                          "now constructs an affinity matrix from data. To use"
                          " a custom affinity matrix, "
                          "set ``affinity=precomputed``.")

        if self.affinity == 'nearest_neighbors':
            connectivity = kneighbors_graph(X, n_neighbors=self.n_neighbors,
                                            include_self=True,
                                            n_jobs=self.n_jobs)
            self.affinity_matrix_ = 0.5 * (connectivity + connectivity.T)
        elif self.affinity == 'precomputed':
            self.affinity_matrix_ = X
        else:
            params = self.kernel_params
            if params is None:
                params = {}
            if not callable(self.affinity):
                params['gamma'] = self.gamma
                params['degree'] = self.degree
                params['coef0'] = self.coef0
            self.affinity_matrix_ = pairwise_kernels(X, metric=self.affinity,
                                                     filter_params=True,
                                                     **params)

        random_state = check_random_state(self.random_state)
        self.labels_ = spectral_clustering(self.affinity_matrix_,
                                           n_clusters=self.n_clusters,
                                           n_components=self.n_components,
                                           eigen_solver=self.eigen_solver,
                                           random_state=random_state,
                                           n_init=self.n_init,
                                           eigen_tol=self.eigen_tol,
                                           assign_labels=self.assign_labels)
        return self

    @property
    def _pairwise(self):
        return self.affinity == "precomputed"
