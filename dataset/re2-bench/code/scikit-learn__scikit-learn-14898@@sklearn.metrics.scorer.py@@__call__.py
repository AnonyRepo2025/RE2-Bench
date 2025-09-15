from collections.abc import Iterable
from functools import partial
from collections import Counter
import warnings
import numpy as np
from . import (r2_score, median_absolute_error, max_error, mean_absolute_error,
               mean_squared_error, mean_squared_log_error,
               mean_poisson_deviance, mean_gamma_deviance, accuracy_score,
               f1_score, roc_auc_score, average_precision_score,
               precision_score, recall_score, log_loss,
               balanced_accuracy_score, explained_variance_score,
               brier_score_loss, jaccard_score)
from .cluster import adjusted_rand_score
from .cluster import homogeneity_score
from .cluster import completeness_score
from .cluster import v_measure_score
from .cluster import mutual_info_score
from .cluster import adjusted_mutual_info_score
from .cluster import normalized_mutual_info_score
from .cluster import fowlkes_mallows_score
from ..utils.multiclass import type_of_target
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
neg_root_mean_squared_error_scorer = make_scorer(mean_squared_error,
                                                 greater_is_better=False,
                                                 squared=False)
neg_mean_poisson_deviance_scorer = make_scorer(
    mean_poisson_deviance, greater_is_better=False
)
neg_mean_gamma_deviance_scorer = make_scorer(
    mean_gamma_deviance, greater_is_better=False
)
accuracy_scorer = make_scorer(accuracy_score)
balanced_accuracy_scorer = make_scorer(balanced_accuracy_score)
roc_auc_scorer = make_scorer(roc_auc_score, greater_is_better=True,
                             needs_threshold=True)
average_precision_scorer = make_scorer(average_precision_score,
                                       needs_threshold=True)
roc_auc_ovo_scorer = make_scorer(roc_auc_score, needs_threshold=True,
                                 multi_class='ovo')
roc_auc_ovo_weighted_scorer = make_scorer(roc_auc_score, needs_threshold=True,
                                          multi_class='ovo',
                                          average='weighted')
roc_auc_ovr_scorer = make_scorer(roc_auc_score, needs_threshold=True,
                                 multi_class='ovr')
roc_auc_ovr_weighted_scorer = make_scorer(roc_auc_score, needs_threshold=True,
                                          multi_class='ovr',
                                          average='weighted')
neg_log_loss_scorer = make_scorer(log_loss, greater_is_better=False,
                                  needs_proba=True)
neg_brier_score_scorer = make_scorer(brier_score_loss,
                                     greater_is_better=False,
                                     needs_proba=True)
brier_score_loss_scorer = make_scorer(brier_score_loss,
                                      greater_is_better=False,
                                      needs_proba=True)
deprecation_msg = ('Scoring method brier_score_loss was renamed to '
                   'neg_brier_score in version 0.22 and will '
                   'be removed in 0.24.')
brier_score_loss_scorer._deprecation_msg = deprecation_msg
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
               neg_root_mean_squared_error=neg_root_mean_squared_error_scorer,
               neg_mean_poisson_deviance=neg_mean_poisson_deviance_scorer,
               neg_mean_gamma_deviance=neg_mean_gamma_deviance_scorer,
               accuracy=accuracy_scorer, roc_auc=roc_auc_scorer,
               roc_auc_ovr=roc_auc_ovr_scorer,
               roc_auc_ovo=roc_auc_ovo_scorer,
               roc_auc_ovr_weighted=roc_auc_ovr_weighted_scorer,
               roc_auc_ovo_weighted=roc_auc_ovo_weighted_scorer,
               balanced_accuracy=balanced_accuracy_scorer,
               average_precision=average_precision_scorer,
               neg_log_loss=neg_log_loss_scorer,
               neg_brier_score=neg_brier_score_scorer,
               # Cluster metrics that use supervised evaluation
               adjusted_rand_score=adjusted_rand_scorer,
               homogeneity_score=homogeneity_scorer,
               completeness_score=completeness_scorer,
               v_measure_score=v_measure_scorer,
               mutual_info_score=mutual_info_scorer,
               adjusted_mutual_info_score=adjusted_mutual_info_scorer,
               normalized_mutual_info_score=normalized_mutual_info_scorer,
               fowlkes_mallows_score=fowlkes_mallows_scorer)

class _BaseScorer:
    def __init__(self, score_func, sign, kwargs):
        self._kwargs = kwargs
        self._score_func = score_func
        self._sign = sign
        # XXX After removing the deprecated scorers (v0.24) remove the
        # XXX deprecation_msg property again and remove __call__'s body again
        self._deprecation_msg = None

    def __repr__(self):
        kwargs_string = "".join([", %s=%s" % (str(k), str(v))
                                 for k, v in self._kwargs.items()])
        return ("make_scorer(%s%s%s%s)"
                % (self._score_func.__name__,
                   "" if self._sign > 0 else ", greater_is_better=False",
                   self._factory_args(), kwargs_string))

    def __call__(self, estimator, X, y_true, sample_weight=None):
        """Evaluate predicted target values for X relative to y_true.

        Parameters
        ----------
        estimator : object
            Trained estimator to use for scoring. Must have a predict_proba
            method; the output of that is used to compute the score.

        X : array-like or sparse matrix
            Test data that will be fed to estimator.predict.

        y_true : array-like
            Gold standard target values for X.

        sample_weight : array-like, optional (default=None)
            Sample weights.

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """
        if self._deprecation_msg is not None:
            warnings.warn(self._deprecation_msg,
                          category=DeprecationWarning,
                          stacklevel=2)
        return self._score(partial(_cached_call, None), estimator, X, y_true,
                           sample_weight=sample_weight)

    def _factory_args(self):
        """Return non-default make_scorer arguments for repr."""
        return ""
