from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
import warnings
import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from ..base import BaseEstimator, ClassifierMixin
from ..metrics.pairwise import rbf_kernel
from ..neighbors import NearestNeighbors
from ..utils.extmath import safe_sparse_dot
from ..utils.multiclass import check_classification_targets
from ..utils.validation import check_is_fitted
from ..utils._param_validation import Interval, StrOptions
from ..exceptions import ConvergenceWarning



class BaseLabelPropagation(ClassifierMixin, BaseEstimator, metaclass=ABCMeta):
    """Base class for label propagation module.

     Parameters
     ----------
     kernel : {'knn', 'rbf'} or callable, default='rbf'
         String identifier for kernel function to use or the kernel function
         itself. Only 'rbf' and 'knn' strings are valid inputs. The function
         passed should take two inputs, each of shape (n_samples, n_features),
         and return a (n_samples, n_samples) shaped weight matrix.

     gamma : float, default=20
         Parameter for rbf kernel.

     n_neighbors : int, default=7
         Parameter for knn kernel. Need to be strictly positive.

     alpha : float, default=1.0
         Clamping factor.

     max_iter : int, default=30
         Change maximum number of iterations allowed.

     tol : float, default=1e-3
         Convergence tolerance: threshold to consider the system at steady
         state.

    n_jobs : int, default=None
         The number of parallel jobs to run.
         ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
         ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
         for more details.
    """

    _parameter_constraints: dict = {
        "kernel": [StrOptions({"knn", "rbf"}), callable],
        "gamma": [Interval(Real, 0, None, closed="left")],
        "n_neighbors": [Interval(Integral, 0, None, closed="neither")],
        "alpha": [None, Interval(Real, 0, 1, closed="neither")],
        "max_iter": [Interval(Integral, 0, None, closed="neither")],
        "tol": [Interval(Real, 0, None, closed="left")],
        "n_jobs": [None, Integral],
    }

    def __init__(
        self,
        kernel="rbf",
        *,
        gamma=20,
        n_neighbors=7,
        alpha=1,
        max_iter=30,
        tol=1e-3,
        n_jobs=None,
    ):

        self.max_iter = max_iter
        self.tol = tol

        # kernel parameters
        self.kernel = kernel
        self.gamma = gamma
        self.n_neighbors = n_neighbors

        # clamping factor
        self.alpha = alpha

        self.n_jobs = n_jobs

    def _get_kernel(self, X, y=None):
        if self.kernel == "rbf":
            if y is None:
                return rbf_kernel(X, X, gamma=self.gamma)
            else:
                return rbf_kernel(X, y, gamma=self.gamma)
        elif self.kernel == "knn":
            if self.nn_fit is None:
                self.nn_fit = NearestNeighbors(
                    n_neighbors=self.n_neighbors, n_jobs=self.n_jobs
                ).fit(X)
            if y is None:
                return self.nn_fit.kneighbors_graph(
                    self.nn_fit._fit_X, self.n_neighbors, mode="connectivity"
                )
            else:
                return self.nn_fit.kneighbors(y, return_distance=False)
        elif callable(self.kernel):
            if y is None:
                return self.kernel(X, X)
            else:
                return self.kernel(X, y)

    @abstractmethod
    def _build_graph(self):
        raise NotImplementedError(
            "Graph construction must be implemented to fit a label propagation model."
        )

    def predict(self, X):
        """Perform inductive inference across the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data matrix.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            Predictions for input data.
        """
        # Note: since `predict` does not accept semi-supervised labels as input,
        # `fit(X, y).predict(X) != fit(X, y).transduction_`.
        # Hence, `fit_predict` is not implemented.
        # See https://github.com/scikit-learn/scikit-learn/pull/24898
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)].ravel()

    def predict_proba(self, X):
        """Predict probability for each possible outcome.

        Compute the probability estimates for each single sample in X
        and each possible outcome seen during training (categorical
        distribution).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data matrix.

        Returns
        -------
        probabilities : ndarray of shape (n_samples, n_classes)
            Normalized probability distributions across
            class labels.
        """
        check_is_fitted(self)

        X_2d = self._validate_data(
            X,
            accept_sparse=["csc", "csr", "coo", "dok", "bsr", "lil", "dia"],
            reset=False,
        )
        weight_matrices = self._get_kernel(self.X_, X_2d)
        if self.kernel == "knn":
            probabilities = np.array(
                [
                    np.sum(self.label_distributions_[weight_matrix], axis=0)
                    for weight_matrix in weight_matrices
                ]
            )
        else:
            weight_matrices = weight_matrices.T
            probabilities = safe_sparse_dot(weight_matrices, self.label_distributions_)
        normalizer = np.atleast_2d(np.sum(probabilities, axis=1)).T
        probabilities /= normalizer
        return probabilities

    def fit(self, X, y):
        """Fit a semi-supervised label propagation model to X.

        The input samples (labeled and unlabeled) are provided by matrix X,
        and target labels are provided by matrix y. We conventionally apply the
        label -1 to unlabeled samples in matrix y in a semi-supervised
        classification.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target class values with unlabeled points marked as -1.
            All unlabeled samples will be transductively assigned labels
            internally, which are stored in `transduction_`.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self._validate_params()
        X, y = self._validate_data(
            X,
            y,
            accept_sparse=["csr", "csc"],
            reset=True,
        )
        self.X_ = X
        check_classification_targets(y)

        # actual graph construction (implementations should override this)
        graph_matrix = self._build_graph()

        # label construction
        # construct a categorical distribution for classification only
        classes = np.unique(y)
        classes = classes[classes != -1]
        self.classes_ = classes

        n_samples, n_classes = len(y), len(classes)

        y = np.asarray(y)
        unlabeled = y == -1

        # initialize distributions
        self.label_distributions_ = np.zeros((n_samples, n_classes))
        for label in classes:
            self.label_distributions_[y == label, classes == label] = 1

        y_static = np.copy(self.label_distributions_)
        if self._variant == "propagation":
            # LabelPropagation
            y_static[unlabeled] = 0
        else:
            # LabelSpreading
            y_static *= 1 - self.alpha

        l_previous = np.zeros((self.X_.shape[0], n_classes))

        unlabeled = unlabeled[:, np.newaxis]
        if sparse.isspmatrix(graph_matrix):
            graph_matrix = graph_matrix.tocsr()

        for self.n_iter_ in range(self.max_iter):
            if np.abs(self.label_distributions_ - l_previous).sum() < self.tol:
                break

            l_previous = self.label_distributions_
            self.label_distributions_ = safe_sparse_dot(
                graph_matrix, self.label_distributions_
            )

            if self._variant == "propagation":
                normalizer = np.sum(self.label_distributions_, axis=1)[:, np.newaxis]
                normalizer[normalizer == 0] = 1
                self.label_distributions_ /= normalizer
                self.label_distributions_ = np.where(
                    unlabeled, self.label_distributions_, y_static
                )
            else:
                # clamp
                self.label_distributions_ = (
                    np.multiply(self.alpha, self.label_distributions_) + y_static
                )
        else:
            warnings.warn(
                "max_iter=%d was reached without convergence." % self.max_iter,
                category=ConvergenceWarning,
            )
            self.n_iter_ += 1

        normalizer = np.sum(self.label_distributions_, axis=1)[:, np.newaxis]
        normalizer[normalizer == 0] = 1
        self.label_distributions_ /= normalizer

        # set the transduction item
        transduction = self.classes_[np.argmax(self.label_distributions_, axis=1)]
        self.transduction_ = transduction.ravel()
        return self
