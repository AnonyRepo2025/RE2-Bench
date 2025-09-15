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

def _fit_liblinear(X, y, C, fit_intercept, intercept_scaling, class_weight,
                   penalty, dual, verbose, max_iter, tol,
                   random_state=None, multi_class='ovr',
                   loss='logistic_regression', epsilon=0.1,
                   sample_weight=None):
    """Used by Logistic Regression (and CV) and LinearSVC.

    Preprocessing is done in this function before supplying it to liblinear.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training vector, where n_samples in the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples,)
        Target vector relative to X

    C : float
        Inverse of cross-validation parameter. Lower the C, the more
        the penalization.

    fit_intercept : bool
        Whether or not to fit the intercept, that is to add a intercept
        term to the decision function.

    intercept_scaling : float
        LibLinear internally penalizes the intercept and this term is subject
        to regularization just like the other terms of the feature vector.
        In order to avoid this, one should increase the intercept_scaling.
        such that the feature vector becomes [x, intercept_scaling].

    class_weight : {dict, 'balanced'}, optional
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

    penalty : str, {'l1', 'l2'}
        The norm of the penalty used in regularization.

    dual : bool
        Dual or primal formulation,

    verbose : int
        Set verbose to any positive number for verbosity.

    max_iter : int
        Number of iterations.

    tol : float
        Stopping condition.

    random_state : int, RandomState instance or None, optional (default=None)
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by `np.random`.

    multi_class : str, {'ovr', 'crammer_singer'}
        `ovr` trains n_classes one-vs-rest classifiers, while `crammer_singer`
        optimizes a joint objective over all classes.
        While `crammer_singer` is interesting from an theoretical perspective
        as it is consistent it is seldom used in practice and rarely leads to
        better accuracy and is more expensive to compute.
        If `crammer_singer` is chosen, the options loss, penalty and dual will
        be ignored.

    loss : str, {'logistic_regression', 'hinge', 'squared_hinge',
                 'epsilon_insensitive', 'squared_epsilon_insensitive}
        The loss function used to fit the model.

    epsilon : float, optional (default=0.1)
        Epsilon parameter in the epsilon-insensitive loss function. Note
        that the value of this parameter depends on the scale of the target
        variable y. If unsure, set epsilon=0.

    sample_weight : array-like, optional
        Weights assigned to each sample.

    Returns
    -------
    coef_ : ndarray, shape (n_features, n_features + 1)
        The coefficient vector got by minimizing the objective function.

    intercept_ : float
        The intercept term added to the vector.

    n_iter_ : int
        Maximum number of iterations run across all classes.
    """
    if loss not in ['epsilon_insensitive', 'squared_epsilon_insensitive']:
        enc = LabelEncoder()
        y_ind = enc.fit_transform(y)
        classes_ = enc.classes_
        if len(classes_) < 2:
            raise ValueError("This solver needs samples of at least 2 classes"
                             " in the data, but the data contains only one"
                             " class: %r" % classes_[0])

        class_weight_ = compute_class_weight(class_weight, classes_, y)
    else:
        class_weight_ = np.empty(0, dtype=np.float64)
        y_ind = y
    liblinear.set_verbosity_wrap(verbose)
    rnd = check_random_state(random_state)
    if verbose:
        print('[LibLinear]', end='')

    # LinearSVC breaks when intercept_scaling is <= 0
    bias = -1.0
    if fit_intercept:
        if intercept_scaling <= 0:
            raise ValueError("Intercept scaling is %r but needs to be greater than 0."
                             " To disable fitting an intercept,"
                             " set fit_intercept=False." % intercept_scaling)
        else:
            bias = intercept_scaling

    libsvm.set_verbosity_wrap(verbose)
    libsvm_sparse.set_verbosity_wrap(verbose)
    liblinear.set_verbosity_wrap(verbose)

    # LibLinear wants targets as doubles, even for classification
    y_ind = np.asarray(y_ind, dtype=np.float64).ravel()
    if sample_weight is None:
        sample_weight = np.ones(X.shape[0])
    else:
        sample_weight = np.array(sample_weight, dtype=np.float64, order='C')
        check_consistent_length(sample_weight, X)

    solver_type = _get_liblinear_solver_type(multi_class, penalty, loss, dual)
    raw_coef_, n_iter_ = liblinear.train_wrap(
        X, y_ind, sp.isspmatrix(X), solver_type, tol, bias, C,
        class_weight_, max_iter, rnd.randint(np.iinfo('i').max),
        epsilon, sample_weight)
    # Regarding rnd.randint(..) in the above signature:
    # seed for srand in range [0..INT_MAX); due to limitations in Numpy
    # on 32-bit platforms, we can't get to the UINT_MAX limit that
    # srand supports
    n_iter_ = max(n_iter_)
    if n_iter_ >= max_iter:
        warnings.warn("Liblinear failed to converge, increase "
                      "the number of iterations.", ConvergenceWarning)

    if fit_intercept:
        coef_ = raw_coef_[:, :-1]
        intercept_ = intercept_scaling * raw_coef_[:, -1]
    else:
        coef_ = raw_coef_
        intercept_ = 0.

    return coef_, intercept_, n_iter_
