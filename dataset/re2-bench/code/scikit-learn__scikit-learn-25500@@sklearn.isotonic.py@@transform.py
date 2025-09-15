import numpy as np
from scipy import interpolate
from scipy.stats import spearmanr
from numbers import Real
import warnings
import math
from .base import BaseEstimator, TransformerMixin, RegressorMixin
from .utils import check_array, check_consistent_length
from .utils.validation import _check_sample_weight, check_is_fitted
from .utils._param_validation import Interval, StrOptions
from ._isotonic import _inplace_contiguous_isotonic_regression, _make_unique

__all__ = ["check_increasing", "isotonic_regression", "IsotonicRegression"]

class IsotonicRegression(RegressorMixin, TransformerMixin, BaseEstimator):
    _parameter_constraints: dict = {
        "y_min": [Interval(Real, None, None, closed="both"), None],
        "y_max": [Interval(Real, None, None, closed="both"), None],
        "increasing": ["boolean", StrOptions({"auto"})],
        "out_of_bounds": [StrOptions({"nan", "clip", "raise"})],
    }

    def __init__(self, *, y_min=None, y_max=None, increasing=True, out_of_bounds="nan"):
        self.y_min = y_min
        self.y_max = y_max
        self.increasing = increasing
        self.out_of_bounds = out_of_bounds

    def _check_input_data_shape(self, X):
        if not (X.ndim == 1 or (X.ndim == 2 and X.shape[1] == 1)):
            msg = (
                "Isotonic regression input X should be a 1d array or "
                "2d array with 1 feature"
            )
            raise ValueError(msg)

    def _transform(self, T):
        if hasattr(self, "X_thresholds_"):
            dtype = self.X_thresholds_.dtype
        else:
            dtype = np.float64

        T = check_array(T, dtype=dtype, ensure_2d=False)

        self._check_input_data_shape(T)
        T = T.reshape(-1)  # use 1d view

        if self.out_of_bounds == "clip":
            T = np.clip(T, self.X_min_, self.X_max_)

        res = self.f_(T)

        res = res.astype(T.dtype)

        return res

    def transform(self, T):
        return self._transform(T)
