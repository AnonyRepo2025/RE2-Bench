from numbers import Number
import numpy as np
import pandas as pd
from .algorithms import bootstrap
from .utils import _check_argument
from scipy.stats import gaussian_kde
from .external.kde import gaussian_kde



class EstimateAggregator:
    def __call__(self, data, var):
        """Aggregate over `var` column of `data` with estimate and error interval."""
        vals = data[var]
        if callable(self.estimator):
            # You would think we could pass to vals.agg, and yet:
            # https://github.com/mwaskom/seaborn/issues/2943
            estimate = self.estimator(vals)
        else:
            estimate = vals.agg(self.estimator)

        # Options that produce no error bars
        if self.error_method is None:
            err_min = err_max = np.nan
        elif len(data) <= 1:
            err_min = err_max = np.nan

        # Generic errorbars from user-supplied function
        elif callable(self.error_method):
            err_min, err_max = self.error_method(vals)

        # Parametric options
        elif self.error_method == "sd":
            half_interval = vals.std() * self.error_level
            err_min, err_max = estimate - half_interval, estimate + half_interval
        elif self.error_method == "se":
            half_interval = vals.sem() * self.error_level
            err_min, err_max = estimate - half_interval, estimate + half_interval

        # Nonparametric options
        elif self.error_method == "pi":
            err_min, err_max = _percentile_interval(vals, self.error_level)
        elif self.error_method == "ci":
            units = data.get("units", None)
            boots = bootstrap(vals, units=units, func=self.estimator, **self.boot_kws)
            err_min, err_max = _percentile_interval(boots, self.error_level)

        return pd.Series({var: estimate, f"{var}min": err_min, f"{var}max": err_max})