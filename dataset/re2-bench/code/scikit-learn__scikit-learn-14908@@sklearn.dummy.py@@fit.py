import warnings
import numpy as np
import scipy.sparse as sp
from .base import BaseEstimator, ClassifierMixin, RegressorMixin
from .base import MultiOutputMixin
from .utils import check_random_state
from .utils.validation import _num_samples
from .utils.validation import check_array
from .utils.validation import check_consistent_length
from .utils.validation import check_is_fitted
from .utils.random import random_choice_csc
from .utils.stats import _weighted_percentile
from .utils.multiclass import class_distribution



class DummyClassifier(MultiOutputMixin, ClassifierMixin, BaseEstimator):
    """
    DummyClassifier is a classifier that makes predictions using simple rules.

    This classifier is useful as a simple baseline to compare with other
    (real) classifiers. Do not use it for real problems.

    Read more in the :ref:`User Guide <dummy_estimators>`.

    Parameters
    ----------
    strategy : str, default="stratified"
        Strategy to use to generate predictions.

        * "stratified": generates predictions by respecting the training
          set's class distribution.
        * "most_frequent": always predicts the most frequent label in the
          training set.
        * "prior": always predicts the class that maximizes the class prior
          (like "most_frequent") and ``predict_proba`` returns the class prior.
        * "uniform": generates predictions uniformly at random.
        * "constant": always predicts a constant label that is provided by
          the user. This is useful for metrics that evaluate a non-majority
          class

          .. versionadded:: 0.17
             Dummy Classifier now supports prior fitting strategy using
             parameter *prior*.

    random_state : int, RandomState instance or None, optional, default=None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    constant : int or str or array of shape = [n_outputs]
        The explicit constant as predicted by the "constant" strategy. This
        parameter is useful only for the "constant" strategy.

    Attributes
    ----------
    classes_ : array or list of array of shape = [n_classes]
        Class labels for each output.

    n_classes_ : array or list of array of shape = [n_classes]
        Number of label for each output.

    class_prior_ : array or list of array of shape = [n_classes]
        Probability of each class for each output.

    n_outputs_ : int,
        Number of outputs.

    sparse_output_ : bool,
        True if the array returned from predict is to be in sparse CSC format.
        Is automatically set to True if the input y is passed in sparse format.
    """

    def __init__(self, strategy="stratified", random_state=None,
                 constant=None):
        self.strategy = strategy
        self.random_state = random_state
        self.constant = constant

    def fit(self, X, y, sample_weight=None):
        """Fit the random classifier.

        Parameters
        ----------
        X : {array-like, object with finite length or shape}
            Training data, requires length = n_samples

        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            Target values.

        sample_weight : array-like of shape = [n_samples], optional
            Sample weights.

        Returns
        -------
        self : object
        """
        allowed_strategies = ("most_frequent", "stratified", "uniform",
                              "constant", "prior")
        if self.strategy not in allowed_strategies:
            raise ValueError("Unknown strategy type: %s, expected one of %s."
                             % (self.strategy, allowed_strategies))

        if self.strategy == "uniform" and sp.issparse(y):
            y = y.toarray()
            warnings.warn('A local copy of the target data has been converted '
                          'to a numpy array. Predicting on sparse target data '
                          'with the uniform strategy would not save memory '
                          'and would be slower.',
                          UserWarning)

        self.sparse_output_ = sp.issparse(y)

        if not self.sparse_output_:
            y = np.atleast_1d(y)

        self.output_2d_ = y.ndim == 2 and y.shape[1] > 1

        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]

        check_consistent_length(X, y, sample_weight)

        if self.strategy == "constant":
            if self.constant is None:
                raise ValueError("Constant target value has to be specified "
                                 "when the constant strategy is used.")
            else:
                constant = np.reshape(np.atleast_1d(self.constant), (-1, 1))
                if constant.shape[0] != self.n_outputs_:
                    raise ValueError("Constant target value should have "
                                     "shape (%d, 1)." % self.n_outputs_)

        (self.classes_,
         self.n_classes_,
         self.class_prior_) = class_distribution(y, sample_weight)

        if self.strategy == "constant":
            for k in range(self.n_outputs_):
                if not any(constant[k][0] == c for c in self.classes_[k]):
                    # Checking in case of constant strategy if the constant
                    # provided by the user is in y.
                    err_msg = ("The constant target value must be present in "
                               "the training data. You provided constant={}. "
                               "Possible values are: {}."
                               .format(self.constant, list(self.classes_[k])))
                    raise ValueError(err_msg)

        if self.n_outputs_ == 1 and not self.output_2d_:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]
            self.class_prior_ = self.class_prior_[0]

        return self

    def predict(self, X):
        """Perform classification on test vectors X.

        Parameters
        ----------
        X : {array-like, object with finite length or shape}
            Training data, requires length = n_samples

        Returns
        -------
        y : array, shape = [n_samples] or [n_samples, n_outputs]
            Predicted target values for X.
        """
        check_is_fitted(self)

        # numpy random_state expects Python int and not long as size argument
        # under Windows
        n_samples = _num_samples(X)
        rs = check_random_state(self.random_state)

        n_classes_ = self.n_classes_
        classes_ = self.classes_
        class_prior_ = self.class_prior_
        constant = self.constant
        if self.n_outputs_ == 1 and not self.output_2d_:
            # Get same type even for self.n_outputs_ == 1
            n_classes_ = [n_classes_]
            classes_ = [classes_]
            class_prior_ = [class_prior_]
            constant = [constant]
        # Compute probability only once
        if self.strategy == "stratified":
            proba = self.predict_proba(X)
            if self.n_outputs_ == 1 and not self.output_2d_:
                proba = [proba]

        if self.sparse_output_:
            class_prob = None
            if self.strategy in ("most_frequent", "prior"):
                classes_ = [np.array([cp.argmax()]) for cp in class_prior_]

            elif self.strategy == "stratified":
                class_prob = class_prior_

            elif self.strategy == "uniform":
                raise ValueError("Sparse target prediction is not "
                                 "supported with the uniform strategy")

            elif self.strategy == "constant":
                classes_ = [np.array([c]) for c in constant]

            y = random_choice_csc(n_samples, classes_, class_prob,
                                  self.random_state)
        else:
            if self.strategy in ("most_frequent", "prior"):
                y = np.tile([classes_[k][class_prior_[k].argmax()] for
                             k in range(self.n_outputs_)], [n_samples, 1])

            elif self.strategy == "stratified":
                y = np.vstack([classes_[k][proba[k].argmax(axis=1)] for
                               k in range(self.n_outputs_)]).T

            elif self.strategy == "uniform":
                ret = [classes_[k][rs.randint(n_classes_[k], size=n_samples)]
                       for k in range(self.n_outputs_)]
                y = np.vstack(ret).T

            elif self.strategy == "constant":
                y = np.tile(self.constant, (n_samples, 1))

            if self.n_outputs_ == 1 and not self.output_2d_:
                y = np.ravel(y)

        return y

    def predict_proba(self, X):
        """
        Return probability estimates for the test vectors X.

        Parameters
        ----------
        X : {array-like, object with finite length or shape}
            Training data, requires length = n_samples

        Returns
        -------
        P : array-like or list of array-lke of shape = [n_samples, n_classes]
            Returns the probability of the sample for each class in
            the model, where classes are ordered arithmetically, for each
            output.
        """
        check_is_fitted(self)

        # numpy random_state expects Python int and not long as size argument
        # under Windows
        n_samples = _num_samples(X)
        rs = check_random_state(self.random_state)

        n_classes_ = self.n_classes_
        classes_ = self.classes_
        class_prior_ = self.class_prior_
        constant = self.constant
        if self.n_outputs_ == 1 and not self.output_2d_:
            # Get same type even for self.n_outputs_ == 1
            n_classes_ = [n_classes_]
            classes_ = [classes_]
            class_prior_ = [class_prior_]
            constant = [constant]

        P = []
        for k in range(self.n_outputs_):
            if self.strategy == "most_frequent":
                ind = class_prior_[k].argmax()
                out = np.zeros((n_samples, n_classes_[k]), dtype=np.float64)
                out[:, ind] = 1.0
            elif self.strategy == "prior":
                out = np.ones((n_samples, 1)) * class_prior_[k]

            elif self.strategy == "stratified":
                out = rs.multinomial(1, class_prior_[k], size=n_samples)
                out = out.astype(np.float64)

            elif self.strategy == "uniform":
                out = np.ones((n_samples, n_classes_[k]), dtype=np.float64)
                out /= n_classes_[k]

            elif self.strategy == "constant":
                ind = np.where(classes_[k] == constant[k])
                out = np.zeros((n_samples, n_classes_[k]), dtype=np.float64)
                out[:, ind] = 1.0

            P.append(out)

        if self.n_outputs_ == 1 and not self.output_2d_:
            P = P[0]

        return P

    def predict_log_proba(self, X):
        """
        Return log probability estimates for the test vectors X.

        Parameters
        ----------
        X : {array-like, object with finite length or shape}
            Training data, requires length = n_samples

        Returns
        -------
        P : array-like or list of array-like of shape = [n_samples, n_classes]
            Returns the log probability of the sample for each class in
            the model, where classes are ordered arithmetically for each
            output.
        """
        proba = self.predict_proba(X)
        if self.n_outputs_ == 1:
            return np.log(proba)
        else:
            return [np.log(p) for p in proba]

    def _more_tags(self):
        return {'poor_score': True, 'no_validation': True}

    def score(self, X, y, sample_weight=None):
        """Returns the mean accuracy on the given test data and labels.

        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.

        Parameters
        ----------
        X : {array-like, None}
            Test samples with shape = (n_samples, n_features) or
            None. Passing None as test samples gives the same result
            as passing real test samples, since DummyClassifier
            operates independently of the sampled observations.

        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True labels for X.

        sample_weight : array-like, shape = [n_samples], optional
            Sample weights.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.

        """
        if X is None:
            X = np.zeros(shape=(len(y), 1))
        return super().score(X, y, sample_weight)
