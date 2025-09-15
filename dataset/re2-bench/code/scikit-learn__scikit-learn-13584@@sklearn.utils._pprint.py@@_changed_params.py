from inspect import signature
import pprint
from collections import OrderedDict
from ..base import BaseEstimator
from .._config import get_config
from . import is_scalar_nan



def _changed_params(estimator):
    """Return dict (param_name: value) of parameters that were given to
    estimator with non-default values."""

    params = estimator.get_params(deep=False)
    filtered_params = {}
    init_func = getattr(estimator.__init__, 'deprecated_original',
                        estimator.__init__)
    init_params = signature(init_func).parameters
    init_params = {name: param.default for name, param in init_params.items()}
    for k, v in params.items():
        if (repr(v) != repr(init_params[k]) and
                not (is_scalar_nan(init_params[k]) and is_scalar_nan(v))):
            filtered_params[k] = v
    return filtered_params
