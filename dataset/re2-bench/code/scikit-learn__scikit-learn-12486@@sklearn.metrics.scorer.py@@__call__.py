from abc import ABCMeta
import numpy as np
from . import (r2_score, median_absolute_error, max_error, mean_absolute_error,
               mean_squared_error, mean_squared_log_error, accuracy_score,
               f1_score, roc_auc_score, average_precision_score,
               precision_score, recall_score, log_loss,
               balanced_accuracy_score, explained_variance_score,
               brier_score_loss)
from .cluster import adjusted_rand_score
from .cluster import homogeneity_score
from .cluster import completeness_score
from .cluster import v_measure_score
from .cluster import mutual_info_score
from .cluster import adjusted_mutual_info_score
from .cluster import normalized_mutual_info_score
from .cluster import fowlkes_mallows_score
from ..utils.multiclass import type_of_target
from ..utils.fixes import _Iterable as Iterable
from ..externals import six
from ..base import is_regressor

explained_variance_scorer = make_scorer(explained_variance_score)
r2_scorer = make_scorer(r2_score)
max_error_scorer = make_scorer(max_error,
                               greater_is_better=False)
neg_mean_squared_error_scorer = make_scorer(mean_squared_error,
                                            greater_is_better=False)
neg_mean_squared_log_error_scorer = make_scorer(mean_squared_log_error,
                                                greater_is_better=False)
neg_mean_absolute_error_scorer = make_scorer(mean_absolute_error,
                                             greater_is_better=False)
neg_median_absolute_error_scorer = make_scorer(median_absolute_error,
                                               greater_is_better=False)
accuracy_scorer = make_scorer(accuracy_score)
f1_scorer = make_scorer(f1_score)
balanced_accuracy_scorer = make_scorer(balanced_accuracy_score)
roc_auc_scorer = make_scorer(roc_auc_score, greater_is_better=True,
                             needs_threshold=True)
average_precision_scorer = make_scorer(average_precision_score,
                                       needs_threshold=True)
precision_scorer = make_scorer(precision_score)
recall_scorer = make_scorer(recall_score)
neg_log_loss_scorer = make_scorer(log_loss, greater_is_better=False,
                                  needs_proba=True)
brier_score_loss_scorer = make_scorer(brier_score_loss,
                                      greater_is_better=False,
                                      needs_proba=True)
adjusted_rand_scorer = make_scorer(adjusted_rand_score)
homogeneity_scorer = make_scorer(homogeneity_score)
completeness_scorer = make_scorer(completeness_score)
v_measure_scorer = make_scorer(v_measure_score)
mutual_info_scorer = make_scorer(mutual_info_score)
adjusted_mutual_info_scorer = make_scorer(adjusted_mutual_info_score)
normalized_mutual_info_scorer = make_scorer(normalized_mutual_info_score)
fowlkes_mallows_scorer = make_scorer(fowlkes_mallows_score)
SCORERS = dict(explained_variance=explained_variance_scorer,
               r2=r2_scorer,
               max_error=max_error_scorer,
               neg_median_absolute_error=neg_median_absolute_error_scorer,
               neg_mean_absolute_error=neg_mean_absolute_error_scorer,
               neg_mean_squared_error=neg_mean_squared_error_scorer,
               neg_mean_squared_log_error=neg_mean_squared_log_error_scorer,
               accuracy=accuracy_scorer, roc_auc=roc_auc_scorer,
               balanced_accuracy=balanced_accuracy_scorer,
               average_precision=average_precision_scorer,
               neg_log_loss=neg_log_loss_scorer,
               brier_score_loss=brier_score_loss_scorer,
               # Cluster metrics that use supervised evaluation
               adjusted_rand_score=adjusted_rand_scorer,
               homogeneity_score=homogeneity_scorer,
               completeness_score=completeness_scorer,
               v_measure_score=v_measure_scorer,
               mutual_info_score=mutual_info_scorer,
               adjusted_mutual_info_score=adjusted_mutual_info_scorer,
               normalized_mutual_info_score=normalized_mutual_info_scorer,
               fowlkes_mallows_score=fowlkes_mallows_scorer)

class _ThresholdScorer(_BaseScorer):
    def __call__(self, clf, X, y, sample_weight=None):
        """Evaluate decision function output for X relative to y_true.

        Parameters
        ----------
        clf : object
            Trained classifier to use for scoring. Must have either a
            decision_function method or a predict_proba method; the output of
            that is used to compute the score.

        X : array-like or sparse matrix
            Test data that will be fed to clf.decision_function or
            clf.predict_proba.

        y : array-like
            Gold standard target values for X. These must be class labels,
            not decision function values.

        sample_weight : array-like, optional (default=None)
            Sample weights.

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """
        y_type = type_of_target(y)
        if y_type not in ("binary", "multilabel-indicator"):
            raise ValueError("{0} format is not supported".format(y_type))

        if is_regressor(clf):
            y_pred = clf.predict(X)
        else:
            try:
                y_pred = clf.decision_function(X)

                # For multi-output multi-class estimator
                if isinstance(y_pred, list):
                    y_pred = np.vstack([p for p in y_pred]).T

            except (NotImplementedError, AttributeError):
                y_pred = clf.predict_proba(X)

                if y_type == "binary":
                    if y_pred.shape[1] == 2:
                        y_pred = y_pred[:, 1]
                    else:
                        raise ValueError('got predict_proba of shape {},'
                                         ' but need classifier with two'
                                         ' classes for {} scoring'.format(
                                             y_pred.shape,
                                             self._score_func.__name__))
                elif isinstance(y_pred, list):
                    y_pred = np.vstack([p[:, -1] for p in y_pred]).T

        if sample_weight is not None:
            return self._sign * self._score_func(y, y_pred,
                                                 sample_weight=sample_weight,
                                                 **self._kwargs)
        else:
            return self._sign * self._score_func(y, y_pred, **self._kwargs)

    def _factory_args(self):
        return ", needs_threshold=True"
