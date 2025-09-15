from abc import ABCMeta, abstractmethod
from copy import deepcopy
import numpy as np
from joblib import Parallel, delayed
import scipy.sparse as sparse
from ..base import clone
from ..base import ClassifierMixin, RegressorMixin, TransformerMixin
from ..base import is_classifier, is_regressor
from ._base import _parallel_fit_estimator
from ._base import _BaseHeterogeneousEnsemble
from ..linear_model import LogisticRegression
from ..linear_model import RidgeCV
from ..model_selection import cross_val_predict
from ..model_selection import check_cv
from ..preprocessing import LabelEncoder
from ..utils import Bunch
from ..utils.metaestimators import if_delegate_has_method
from ..utils.multiclass import check_classification_targets
from ..utils.validation import check_is_fitted
from ..utils.validation import column_or_1d



class _BaseStacking(TransformerMixin, _BaseHeterogeneousEnsemble,
                    metaclass=ABCMeta):
    """Base class for stacking method."""

    @abstractmethod
    def __init__(self, estimators, final_estimator=None, cv=None,
                 stack_method='auto', n_jobs=None, verbose=0,
                 passthrough=False):
        super().__init__(estimators=estimators)
        self.final_estimator = final_estimator
        self.cv = cv
        self.stack_method = stack_method
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.passthrough = passthrough

    def _clone_final_estimator(self, default):
        if self.final_estimator is not None:
            self.final_estimator_ = clone(self.final_estimator)
        else:
            self.final_estimator_ = clone(default)

    def _concatenate_predictions(self, X, predictions):
        """Concatenate the predictions of each first layer learner and
        possibly the input dataset `X`.

        If `X` is sparse and `self.passthrough` is False, the output of
        `transform` will be dense (the predictions). If `X` is sparse
        and `self.passthrough` is True, the output of `transform` will
        be sparse.

        This helper is in charge of ensuring the preditions are 2D arrays and
        it will drop one of the probability column when using probabilities
        in the binary case. Indeed, the p(y|c=0) = 1 - p(y|c=1)
        """
        X_meta = []
        for est_idx, preds in enumerate(predictions):
            # case where the the estimator returned a 1D array
            if preds.ndim == 1:
                X_meta.append(preds.reshape(-1, 1))
            else:
                if (self.stack_method_[est_idx] == 'predict_proba' and
                        len(self.classes_) == 2):
                    # Remove the first column when using probabilities in
                    # binary classification because both features are perfectly
                    # collinear.
                    X_meta.append(preds[:, 1:])
                else:
                    X_meta.append(preds)
        if self.passthrough:
            X_meta.append(X)
            if sparse.issparse(X):
                return sparse.hstack(X_meta, format=X.format)

        return np.hstack(X_meta)

    @staticmethod
    def _method_name(name, estimator, method):
        if estimator == 'drop':
            return None
        if method == 'auto':
            if getattr(estimator, 'predict_proba', None):
                return 'predict_proba'
            elif getattr(estimator, 'decision_function', None):
                return 'decision_function'
            else:
                return 'predict'
        else:
            if not hasattr(estimator, method):
                raise ValueError('Underlying estimator {} does not implement '
                                 'the method {}.'.format(name, method))
            return method

    def fit(self, X, y, sample_weight=None):
        """Fit the estimators.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,) or None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if all underlying estimators
            support sample weights.

        Returns
        -------
        self : object
        """
        # all_estimators contains all estimators, the one to be fitted and the
        # 'drop' string.
        names, all_estimators = self._validate_estimators()
        self._validate_final_estimator()

        stack_method = [self.stack_method] * len(all_estimators)

        # Fit the base estimators on the whole training data. Those
        # base estimators will be used in transform, predict, and
        # predict_proba. They are exposed publicly.
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_parallel_fit_estimator)(clone(est), X, y, sample_weight)
            for est in all_estimators if est != 'drop'
        )

        self.named_estimators_ = Bunch()
        est_fitted_idx = 0
        for name_est, org_est in zip(names, all_estimators):
            if org_est != 'drop':
                self.named_estimators_[name_est] = self.estimators_[
                    est_fitted_idx]
                est_fitted_idx += 1

        # To train the meta-classifier using the most data as possible, we use
        # a cross-validation to obtain the output of the stacked estimators.

        # To ensure that the data provided to each estimator are the same, we
        # need to set the random state of the cv if there is one and we need to
        # take a copy.
        cv = check_cv(self.cv, y=y, classifier=is_classifier(self))
        if hasattr(cv, 'random_state') and cv.random_state is None:
            cv.random_state = np.random.RandomState()

        self.stack_method_ = [
            self._method_name(name, est, meth)
            for name, est, meth in zip(names, all_estimators, stack_method)
        ]

        predictions = Parallel(n_jobs=self.n_jobs)(
            delayed(cross_val_predict)(clone(est), X, y, cv=deepcopy(cv),
                                       method=meth, n_jobs=self.n_jobs,
                                       verbose=self.verbose)
            for est, meth in zip(all_estimators, self.stack_method_)
            if est != 'drop'
        )

        # Only not None or not 'drop' estimators will be used in transform.
        # Remove the None from the method as well.
        self.stack_method_ = [
            meth for (meth, est) in zip(self.stack_method_, all_estimators)
            if est != 'drop'
        ]

        X_meta = self._concatenate_predictions(X, predictions)
        if sample_weight is not None:
            try:
                self.final_estimator_.fit(
                    X_meta, y, sample_weight=sample_weight
                )
            except TypeError as exc:
                if "unexpected keyword argument 'sample_weight'" in str(exc):
                    raise TypeError(
                        "Underlying estimator {} does not support sample "
                        "weights."
                        .format(self.final_estimator_.__class__.__name__)
                    ) from exc
                raise
        else:
            self.final_estimator_.fit(X_meta, y)

        return self

    def _transform(self, X):
        """Concatenate and return the predictions of the estimators."""
        check_is_fitted(self)
        predictions = [
            getattr(est, meth)(X)
            for est, meth in zip(self.estimators_, self.stack_method_)
            if est != 'drop'
        ]
        return self._concatenate_predictions(X, predictions)

    @if_delegate_has_method(delegate='final_estimator_')
    def predict(self, X, **predict_params):
        """Predict target for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        **predict_params : dict of str -> obj
            Parameters to the `predict` called by the `final_estimator`. Note
            that this may be used to return uncertainties from some estimators
            with `return_std` or `return_cov`. Be aware that it will only
            accounts for uncertainty in the final estimator.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_output)
            Predicted targets.
        """

        check_is_fitted(self)
        return self.final_estimator_.predict(
            self.transform(X), **predict_params
        )
