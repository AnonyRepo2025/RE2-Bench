import numpy as np
from ..base import BaseEstimator
from ._base import SelectorMixin
from ..utils import check_array
from ..utils.sparsefuncs import mean_variance_axis, min_max_axis
from ..utils.validation import check_is_fitted

class VarianceThreshold(SelectorMixin, BaseEstimator):

    def __init__(self, threshold=0.0):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = check_array(X, ('csr', 'csc'), dtype=np.float64, force_all_finite='allow-nan')
        if hasattr(X, 'toarray'):
            _, self.variances_ = mean_variance_axis(X, axis=0)
            if self.threshold == 0:
                mins, maxes = min_max_axis(X, axis=0)
                peak_to_peaks = maxes - mins
        else:
            self.variances_ = np.nanvar(X, axis=0)
            if self.threshold == 0:
                peak_to_peaks = np.ptp(X, axis=0)
        if self.threshold == 0:
            compare_arr = np.array([self.variances_, peak_to_peaks])
            self.variances_ = np.nanmin(compare_arr, axis=0)
        if np.all(~np.isfinite(self.variances_) | (self.variances_ <= self.threshold)):
            msg = 'No feature in X meets the variance threshold {0:.5f}'
            if X.shape[0] == 1:
                msg += ' (X contains only one sample)'
            raise ValueError(msg.format(self.threshold))
        return self

    def _get_support_mask(self):
        check_is_fitted(self)
        return self.variances_ > self.threshold

    def _more_tags(self):
        return {'allow_nan': True}