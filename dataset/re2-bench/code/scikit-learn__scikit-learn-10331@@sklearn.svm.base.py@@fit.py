from __future__ import print_function
import numpy as np
import scipy.sparse as sp
import warnings
from abc import ABCMeta, abstractmethod
from . import libsvm, liblinear
from . import libsvm_sparse
from ..base import BaseEstimator, ClassifierMixin
from ..preprocessing import LabelEncoder
from ..utils.multiclass import _ovr_decision_function
from ..utils import check_array, check_consistent_length, check_random_state
from ..utils import column_or_1d, check_X_y
from ..utils import compute_class_weight
from ..utils.extmath import safe_sparse_dot
from ..utils.validation import check_is_fitted
from ..utils.multiclass import check_classification_targets
from ..externals import six
from ..exceptions import ConvergenceWarning
from ..exceptions import NotFittedError

LIBSVM_IMPL = ['c_svc', 'nu_svc', 'one_class', 'epsilon_svr', 'nu_svr']

class BaseLibSVM(six.with_metaclass(ABCMeta, BaseEstimator)):
    """Base class for estimators that use libsvm as backing library

    This implements support vector machine classification and regression.

    Parameter documentation is in the derived `SVC` class.
    """

    # The order of these must match the integer values in LibSVM.
    # XXX These are actually the same in the dense case. Need to factor
    # this out.
    _sparse_kernels = ["linear", "poly", "rbf", "sigmoid", "precomputed"]

    @abstractmethod
    def __init__(self, kernel, degree, gamma, coef0,
                 tol, C, nu, epsilon, shrinking, probability, cache_size,
                 class_weight, verbose, max_iter, random_state):

        if self._impl not in LIBSVM_IMPL:  # pragma: no cover
            raise ValueError("impl should be one of %s, %s was given" % (
                LIBSVM_IMPL, self._impl))

        if gamma == 0:
            msg = ("The gamma value of 0.0 is invalid. Use 'auto' to set"
                   " gamma to a value of 1 / n_features.")
            raise ValueError(msg)

        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.C = C
        self.nu = nu
        self.epsilon = epsilon
        self.shrinking = shrinking
        self.probability = probability
        self.cache_size = cache_size
        self.class_weight = class_weight
        self.verbose = verbose
        self.max_iter = max_iter
        self.random_state = random_state

    @property
    def _pairwise(self):
        # Used by cross_val_score.
        return self.kernel == "precomputed"

    def fit(self, X, y, sample_weight=None):
        """Fit the SVM model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
            For kernel="precomputed", the expected shape of X is
            (n_samples, n_samples).

        y : array-like, shape (n_samples,)
            Target values (class labels in classification, real numbers in
            regression)

        sample_weight : array-like, shape (n_samples,)
            Per-sample weights. Rescale C per sample. Higher weights
            force the classifier to put more emphasis on these points.

        Returns
        -------
        self : object

        Notes
        ------
        If X and y are not C-ordered and contiguous arrays of np.float64 and
        X is not a scipy.sparse.csr_matrix, X and/or y may be copied.

        If X is a dense array, then the other methods will not support sparse
        matrices as input.
        """

        rnd = check_random_state(self.random_state)

        sparse = sp.isspmatrix(X)
        if sparse and self.kernel == "precomputed":
            raise TypeError("Sparse precomputed kernels are not supported.")
        self._sparse = sparse and not callable(self.kernel)

        X, y = check_X_y(X, y, dtype=np.float64, order='C', accept_sparse='csr')
        y = self._validate_targets(y)

        sample_weight = np.asarray([]
                                   if sample_weight is None
                                   else sample_weight, dtype=np.float64)
        solver_type = LIBSVM_IMPL.index(self._impl)

        # input validation
        if solver_type != 2 and X.shape[0] != y.shape[0]:
            raise ValueError("X and y have incompatible shapes.\n" +
                             "X has %s samples, but y has %s." %
                             (X.shape[0], y.shape[0]))

        if self.kernel == "precomputed" and X.shape[0] != X.shape[1]:
            raise ValueError("X.shape[0] should be equal to X.shape[1]")

        if sample_weight.shape[0] > 0 and sample_weight.shape[0] != X.shape[0]:
            raise ValueError("sample_weight and X have incompatible shapes: "
                             "%r vs %r\n"
                             "Note: Sparse matrices cannot be indexed w/"
                             "boolean masks (use `indices=True` in CV)."
                             % (sample_weight.shape, X.shape))

        if self.gamma in ('scale', 'auto_deprecated'):
            if sparse:
                # std = sqrt(E[X^2] - E[X]^2)
                X_std = np.sqrt((X.multiply(X)).mean() - (X.mean())**2)
            else:
                X_std = X.std()
            if self.gamma == 'scale':
                if X_std != 0:
                    self._gamma = 1.0 / (X.shape[1] * X_std)
                else:
                    self._gamma = 1.0
            else:
                kernel_uses_gamma = (not callable(self.kernel) and self.kernel
                                     not in ('linear', 'precomputed'))
                if kernel_uses_gamma and not np.isclose(X_std, 1.0):
                    # NOTE: when deprecation ends we need to remove explicitly
                    # setting `gamma` in examples (also in tests). See
                    # https://github.com/scikit-learn/scikit-learn/pull/10331
                    # for the examples/tests that need to be reverted.
                    warnings.warn("The default value of gamma will change "
                                  "from 'auto' to 'scale' in version 0.22 to "
                                  "account better for unscaled features. Set "
                                  "gamma explicitly to 'auto' or 'scale' to "
                                  "avoid this warning.", FutureWarning)
                self._gamma = 1.0 / X.shape[1]
        elif self.gamma == 'auto':
            self._gamma = 1.0 / X.shape[1]
        else:
            self._gamma = self.gamma

        kernel = self.kernel
        if callable(kernel):
            kernel = 'precomputed'

        fit = self._sparse_fit if self._sparse else self._dense_fit
        if self.verbose:  # pragma: no cover
            print('[LibSVM]', end='')

        seed = rnd.randint(np.iinfo('i').max)
        fit(X, y, sample_weight, solver_type, kernel, random_seed=seed)
        # see comment on the other call to np.iinfo in this file

        self.shape_fit_ = X.shape

        # In binary case, we need to flip the sign of coef, intercept and
        # decision function. Use self._intercept_ and self._dual_coef_ internally.
        self._intercept_ = self.intercept_.copy()
        self._dual_coef_ = self.dual_coef_
        if self._impl in ['c_svc', 'nu_svc'] and len(self.classes_) == 2:
            self.intercept_ *= -1
            self.dual_coef_ = -self.dual_coef_

        return self

    def _validate_targets(self, y):
        """Validation of y and class_weight.

        Default implementation for SVR and one-class; overridden in BaseSVC.
        """
        # XXX this is ugly.
        # Regression models should not have a class_weight_ attribute.
        self.class_weight_ = np.empty(0)
        return column_or_1d(y, warn=True).astype(np.float64)

    def _warn_from_fit_status(self):
        assert self.fit_status_ in (0, 1)
        if self.fit_status_ == 1:
            warnings.warn('Solver terminated early (max_iter=%i).'
                          '  Consider pre-processing your data with'
                          ' StandardScaler or MinMaxScaler.'
                          % self.max_iter, ConvergenceWarning)

    def _dense_fit(self, X, y, sample_weight, solver_type, kernel,
                   random_seed):
        if callable(self.kernel):
            # you must store a reference to X to compute the kernel in predict
            # TODO: add keyword copy to copy on demand
            self.__Xfit = X
            X = self._compute_kernel(X)

            if X.shape[0] != X.shape[1]:
                raise ValueError("X.shape[0] should be equal to X.shape[1]")

        libsvm.set_verbosity_wrap(self.verbose)

        # we don't pass **self.get_params() to allow subclasses to
        # add other parameters to __init__
        self.support_, self.support_vectors_, self.n_support_, \
            self.dual_coef_, self.intercept_, self.probA_, \
            self.probB_, self.fit_status_ = libsvm.fit(
                X, y,
                svm_type=solver_type, sample_weight=sample_weight,
                class_weight=self.class_weight_, kernel=kernel, C=self.C,
                nu=self.nu, probability=self.probability, degree=self.degree,
                shrinking=self.shrinking, tol=self.tol,
                cache_size=self.cache_size, coef0=self.coef0,
                gamma=self._gamma, epsilon=self.epsilon,
                max_iter=self.max_iter, random_seed=random_seed)

        self._warn_from_fit_status()

    def _sparse_fit(self, X, y, sample_weight, solver_type, kernel,
                    random_seed):
        X.data = np.asarray(X.data, dtype=np.float64, order='C')
        X.sort_indices()

        kernel_type = self._sparse_kernels.index(kernel)

        libsvm_sparse.set_verbosity_wrap(self.verbose)

        self.support_, self.support_vectors_, dual_coef_data, \
            self.intercept_, self.n_support_, \
            self.probA_, self.probB_, self.fit_status_ = \
            libsvm_sparse.libsvm_sparse_train(
                X.shape[1], X.data, X.indices, X.indptr, y, solver_type,
                kernel_type, self.degree, self._gamma, self.coef0, self.tol,
                self.C, self.class_weight_,
                sample_weight, self.nu, self.cache_size, self.epsilon,
                int(self.shrinking), int(self.probability), self.max_iter,
                random_seed)

        self._warn_from_fit_status()

        if hasattr(self, "classes_"):
            n_class = len(self.classes_) - 1
        else:   # regression
            n_class = 1
        n_SV = self.support_vectors_.shape[0]

        dual_coef_indices = np.tile(np.arange(n_SV), n_class)
        dual_coef_indptr = np.arange(0, dual_coef_indices.size + 1,
                                     dual_coef_indices.size / n_class)
        self.dual_coef_ = sp.csr_matrix(
            (dual_coef_data, dual_coef_indices, dual_coef_indptr),
            (n_class, n_SV))

    def predict(self, X):
        """Perform regression on samples in X.

        For an one-class model, +1 (inlier) or -1 (outlier) is returned.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            For kernel="precomputed", the expected shape of X is
            (n_samples_test, n_samples_train).

        Returns
        -------
        y_pred : array, shape (n_samples,)
        """
        X = self._validate_for_predict(X)
        predict = self._sparse_predict if self._sparse else self._dense_predict
        return predict(X)

    def _dense_predict(self, X):
        n_samples, n_features = X.shape
        X = self._compute_kernel(X)
        if X.ndim == 1:
            X = check_array(X, order='C')

        kernel = self.kernel
        if callable(self.kernel):
            kernel = 'precomputed'
            if X.shape[1] != self.shape_fit_[0]:
                raise ValueError("X.shape[1] = %d should be equal to %d, "
                                 "the number of samples at training time" %
                                 (X.shape[1], self.shape_fit_[0]))

        svm_type = LIBSVM_IMPL.index(self._impl)

        return libsvm.predict(
            X, self.support_, self.support_vectors_, self.n_support_,
            self._dual_coef_, self._intercept_,
            self.probA_, self.probB_, svm_type=svm_type, kernel=kernel,
            degree=self.degree, coef0=self.coef0, gamma=self._gamma,
            cache_size=self.cache_size)

    def _sparse_predict(self, X):
        # Precondition: X is a csr_matrix of dtype np.float64.
        kernel = self.kernel
        if callable(kernel):
            kernel = 'precomputed'

        kernel_type = self._sparse_kernels.index(kernel)

        C = 0.0  # C is not useful here

        return libsvm_sparse.libsvm_sparse_predict(
            X.data, X.indices, X.indptr,
            self.support_vectors_.data,
            self.support_vectors_.indices,
            self.support_vectors_.indptr,
            self._dual_coef_.data, self._intercept_,
            LIBSVM_IMPL.index(self._impl), kernel_type,
            self.degree, self._gamma, self.coef0, self.tol,
            C, self.class_weight_,
            self.nu, self.epsilon, self.shrinking,
            self.probability, self.n_support_,
            self.probA_, self.probB_)

    def _compute_kernel(self, X):
        """Return the data transformed by a callable kernel"""
        if callable(self.kernel):
            # in the case of precomputed kernel given as a function, we
            # have to compute explicitly the kernel matrix
            kernel = self.kernel(X, self.__Xfit)
            if sp.issparse(kernel):
                kernel = kernel.toarray()
            X = np.asarray(kernel, dtype=np.float64, order='C')
        return X

    def _decision_function(self, X):
        """Distance of the samples X to the separating hyperplane.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        X : array-like, shape (n_samples, n_class * (n_class-1) / 2)
            Returns the decision function of the sample for each class
            in the model.
        """
        # NOTE: _validate_for_predict contains check for is_fitted
        # hence must be placed before any other attributes are used.
        X = self._validate_for_predict(X)
        X = self._compute_kernel(X)

        if self._sparse:
            dec_func = self._sparse_decision_function(X)
        else:
            dec_func = self._dense_decision_function(X)

        # In binary case, we need to flip the sign of coef, intercept and
        # decision function.
        if self._impl in ['c_svc', 'nu_svc'] and len(self.classes_) == 2:
            return -dec_func.ravel()

        return dec_func

    def _dense_decision_function(self, X):
        X = check_array(X, dtype=np.float64, order="C")

        kernel = self.kernel
        if callable(kernel):
            kernel = 'precomputed'

        return libsvm.decision_function(
            X, self.support_, self.support_vectors_, self.n_support_,
            self._dual_coef_, self._intercept_,
            self.probA_, self.probB_,
            svm_type=LIBSVM_IMPL.index(self._impl),
            kernel=kernel, degree=self.degree, cache_size=self.cache_size,
            coef0=self.coef0, gamma=self._gamma)

    def _sparse_decision_function(self, X):
        X.data = np.asarray(X.data, dtype=np.float64, order='C')

        kernel = self.kernel
        if hasattr(kernel, '__call__'):
            kernel = 'precomputed'

        kernel_type = self._sparse_kernels.index(kernel)

        return libsvm_sparse.libsvm_sparse_decision_function(
            X.data, X.indices, X.indptr,
            self.support_vectors_.data,
            self.support_vectors_.indices,
            self.support_vectors_.indptr,
            self._dual_coef_.data, self._intercept_,
            LIBSVM_IMPL.index(self._impl), kernel_type,
            self.degree, self._gamma, self.coef0, self.tol,
            self.C, self.class_weight_,
            self.nu, self.epsilon, self.shrinking,
            self.probability, self.n_support_,
            self.probA_, self.probB_)

    def _validate_for_predict(self, X):
        check_is_fitted(self, 'support_')

        X = check_array(X, accept_sparse='csr', dtype=np.float64, order="C")
        if self._sparse and not sp.isspmatrix(X):
            X = sp.csr_matrix(X)
        if self._sparse:
            X.sort_indices()

        if sp.issparse(X) and not self._sparse and not callable(self.kernel):
            raise ValueError(
                "cannot use sparse input in %r trained on dense data"
                % type(self).__name__)
        n_samples, n_features = X.shape

        if self.kernel == "precomputed":
            if X.shape[1] != self.shape_fit_[0]:
                raise ValueError("X.shape[1] = %d should be equal to %d, "
                                 "the number of samples at training time" %
                                 (X.shape[1], self.shape_fit_[0]))
        elif n_features != self.shape_fit_[1]:
            raise ValueError("X.shape[1] = %d should be equal to %d, "
                             "the number of features at training time" %
                             (n_features, self.shape_fit_[1]))
        return X

    @property
    def coef_(self):
        if self.kernel != 'linear':
            raise AttributeError('coef_ is only available when using a '
                                 'linear kernel')

        coef = self._get_coef()

        # coef_ being a read-only property, it's better to mark the value as
        # immutable to avoid hiding potential bugs for the unsuspecting user.
        if sp.issparse(coef):
            # sparse matrix do not have global flags
            coef.data.flags.writeable = False
        else:
            # regular dense array
            coef.flags.writeable = False
        return coef

    def _get_coef(self):
        return safe_sparse_dot(self._dual_coef_, self.support_vectors_)
