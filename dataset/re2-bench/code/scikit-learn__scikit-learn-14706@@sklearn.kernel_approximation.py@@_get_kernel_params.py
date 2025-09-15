import warnings
import numpy as np
import scipy.sparse as sp
from scipy.linalg import svd
from .base import BaseEstimator
from .base import TransformerMixin
from .utils import check_array, check_random_state, as_float_array
from .utils.extmath import safe_sparse_dot
from .utils.validation import check_is_fitted
from .metrics.pairwise import pairwise_kernels, KERNEL_PARAMS



class Nystroem(BaseEstimator, TransformerMixin):
    """Approximate a kernel map using a subset of the training data.

    Constructs an approximate feature map for an arbitrary kernel
    using a subset of the data as basis.

    Read more in the :ref:`User Guide <nystroem_kernel_approx>`.

    Parameters
    ----------
    kernel : string or callable, default="rbf"
        Kernel map to be approximated. A callable should accept two arguments
        and the keyword arguments passed to this object as kernel_params, and
        should return a floating point number.

    gamma : float, default=None
        Gamma parameter for the RBF, laplacian, polynomial, exponential chi2
        and sigmoid kernels. Interpretation of the default value is left to
        the kernel; see the documentation for sklearn.metrics.pairwise.
        Ignored by other kernels.

    coef0 : float, default=None
        Zero coefficient for polynomial and sigmoid kernels.
        Ignored by other kernels.

    degree : float, default=None
        Degree of the polynomial kernel. Ignored by other kernels.

    kernel_params : mapping of string to any, optional
        Additional parameters (keyword arguments) for kernel function passed
        as callable object.

    n_components : int
        Number of features to construct.
        How many data points will be used to construct the mapping.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    components_ : array, shape (n_components, n_features)
        Subset of training points used to construct the feature map.

    component_indices_ : array, shape (n_components)
        Indices of ``components_`` in the training set.

    normalization_ : array, shape (n_components, n_components)
        Normalization matrix needed for embedding.
        Square root of the kernel matrix on ``components_``.

    Examples
    --------
    >>> from sklearn import datasets, svm
    >>> from sklearn.kernel_approximation import Nystroem
    >>> X, y = datasets.load_digits(n_class=9, return_X_y=True)
    >>> data = X / 16.
    >>> clf = svm.LinearSVC()
    >>> feature_map_nystroem = Nystroem(gamma=.2,
    ...                                 random_state=1,
    ...                                 n_components=300)
    >>> data_transformed = feature_map_nystroem.fit_transform(data)
    >>> clf.fit(data_transformed, y)
    LinearSVC()
    >>> clf.score(data_transformed, y)
    0.9987...

    References
    ----------
    * Williams, C.K.I. and Seeger, M.
      "Using the Nystroem method to speed up kernel machines",
      Advances in neural information processing systems 2001

    * T. Yang, Y. Li, M. Mahdavi, R. Jin and Z. Zhou
      "Nystroem Method vs Random Fourier Features: A Theoretical and Empirical
      Comparison",
      Advances in Neural Information Processing Systems 2012


    See also
    --------
    RBFSampler : An approximation to the RBF kernel using random Fourier
                 features.

    sklearn.metrics.pairwise.kernel_metrics : List of built-in kernels.
    """

    def __init__(self, kernel="rbf", gamma=None, coef0=None, degree=None,
                 kernel_params=None, n_components=100, random_state=None):
        self.kernel = kernel
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.kernel_params = kernel_params
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit estimator to data.

        Samples a subset of training points, computes kernel
        on these and computes normalization matrix.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_feature)
            Training data.
        """
        X = check_array(X, accept_sparse='csr')
        rnd = check_random_state(self.random_state)
        n_samples = X.shape[0]

        # get basis vectors
        if self.n_components > n_samples:
            # XXX should we just bail?
            n_components = n_samples
            warnings.warn("n_components > n_samples. This is not possible.\n"
                          "n_components was set to n_samples, which results"
                          " in inefficient evaluation of the full kernel.")

        else:
            n_components = self.n_components
        n_components = min(n_samples, n_components)
        inds = rnd.permutation(n_samples)
        basis_inds = inds[:n_components]
        basis = X[basis_inds]

        basis_kernel = pairwise_kernels(basis, metric=self.kernel,
                                        filter_params=True,
                                        **self._get_kernel_params())

        # sqrt of kernel matrix on basis vectors
        U, S, V = svd(basis_kernel)
        S = np.maximum(S, 1e-12)
        self.normalization_ = np.dot(U / np.sqrt(S), V)
        self.components_ = basis
        self.component_indices_ = inds
        return self

    def transform(self, X):
        """Apply feature map to X.

        Computes an approximate feature map using the kernel
        between some training points and X.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_transformed : array, shape=(n_samples, n_components)
            Transformed data.
        """
        check_is_fitted(self)
        X = check_array(X, accept_sparse='csr')

        kernel_params = self._get_kernel_params()
        embedded = pairwise_kernels(X, self.components_,
                                    metric=self.kernel,
                                    filter_params=True,
                                    **kernel_params)
        return np.dot(embedded, self.normalization_.T)

    def _get_kernel_params(self):
        params = self.kernel_params
        if params is None:
            params = {}
        if not callable(self.kernel) and self.kernel != 'precomputed':
            for param in (KERNEL_PARAMS[self.kernel]):
                if getattr(self, param) is not None:
                    params[param] = getattr(self, param)
        else:
            if (self.gamma is not None or
                    self.coef0 is not None or
                    self.degree is not None):
                raise ValueError("Don't pass gamma, coef0 or degree to "
                                 "Nystroem if using a callable "
                                 "or precomputed kernel")

        return params
