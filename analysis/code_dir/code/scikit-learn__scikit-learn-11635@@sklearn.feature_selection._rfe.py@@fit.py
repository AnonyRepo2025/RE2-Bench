import numpy as np
from joblib import Parallel, delayed, effective_n_jobs
from ..utils import check_X_y, safe_sqr
from ..utils.metaestimators import if_delegate_has_method
from ..utils.metaestimators import _safe_split
from ..utils.validation import check_is_fitted
from ..base import BaseEstimator
from ..base import MetaEstimatorMixin
from ..base import clone
from ..base import is_classifier
from ..model_selection import check_cv
from ..model_selection._validation import _score
from ..metrics import check_scoring
from ._base import SelectorMixin

class RFECV(RFE):

    def __init__(self, estimator, step=1, min_features_to_select=1, cv=None, scoring=None, verbose=0, n_jobs=None):
        self.estimator = estimator
        self.step = step
        self.cv = cv
        self.scoring = scoring
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.min_features_to_select = min_features_to_select

    def fit(self, X, y, groups=None):
        X, y = check_X_y(X, y, 'csr', ensure_min_features=2, force_all_finite=False)
        cv = check_cv(self.cv, y, is_classifier(self.estimator))
        scorer = check_scoring(self.estimator, scoring=self.scoring)
        n_features = X.shape[1]
        if 0.0 < self.step < 1.0:
            step = int(max(1, self.step * n_features))
        else:
            step = int(self.step)
        if step <= 0:
            raise ValueError('Step must be >0')
        rfe = RFE(estimator=self.estimator, n_features_to_select=self.min_features_to_select, step=self.step, verbose=self.verbose)
        if effective_n_jobs(self.n_jobs) == 1:
            parallel, func = (list, _rfe_single_fit)
        else:
            parallel = Parallel(n_jobs=self.n_jobs)
            func = delayed(_rfe_single_fit)
        scores = parallel((func(rfe, self.estimator, X, y, train, test, scorer) for train, test in cv.split(X, y, groups)))
        scores = np.sum(scores, axis=0)
        scores_rev = scores[::-1]
        argmax_idx = len(scores) - np.argmax(scores_rev) - 1
        n_features_to_select = max(n_features - argmax_idx * step, self.min_features_to_select)
        rfe = RFE(estimator=self.estimator, n_features_to_select=n_features_to_select, step=self.step, verbose=self.verbose)
        rfe.fit(X, y)
        self.support_ = rfe.support_
        self.n_features_ = rfe.n_features_
        self.ranking_ = rfe.ranking_
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(self.transform(X), y)
        self.grid_scores_ = scores[::-1] / cv.get_n_splits(X, y, groups)
        return self