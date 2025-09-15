import warnings
import itertools
from copy import copy
from functools import partial
from collections import UserString
from collections.abc import Iterable, Sequence, Mapping
from numbers import Number
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib as mpl
from ._decorators import (
    share_init_params_with_map,
)
from .palettes import (
    QUAL_PALETTES,
    color_palette,
)
from .utils import (
    _check_argument,
    desaturate,
    get_color_cycle,
    remove_na,
)
from .axisgrid import FacetGrid



def variable_type(vector, boolean_type="numeric"):
    """
    Determine whether a vector contains numeric, categorical, or datetime data.

    This function differs from the pandas typing API in two ways:

    - Python sequences or object-typed PyData objects are considered numeric if
      all of their entries are numeric.
    - String or mixed-type data are considered categorical even if not
      explicitly represented as a :class:`pandas.api.types.CategoricalDtype`.

    Parameters
    ----------
    vector : :func:`pandas.Series`, :func:`numpy.ndarray`, or Python sequence
        Input data to test.
    boolean_type : 'numeric' or 'categorical'
        Type to use for vectors containing only 0s and 1s (and NAs).

    Returns
    -------
    var_type : 'numeric', 'categorical', or 'datetime'
        Name identifying the type of data in the vector.
    """
    vector = pd.Series(vector)

    # If a categorical dtype is set, infer categorical
    if isinstance(vector.dtype, pd.CategoricalDtype):
        return VariableType("categorical")

    # Special-case all-na data, which is always "numeric"
    if pd.isna(vector).all():
        return VariableType("numeric")

    # At this point, drop nans to simplify further type inference
    vector = vector.dropna()

    # Special-case binary/boolean data, allow caller to determine
    # This triggers a numpy warning when vector has strings/objects
    # https://github.com/numpy/numpy/issues/6784
    # Because we reduce with .all(), we are agnostic about whether the
    # comparison returns a scalar or vector, so we will ignore the warning.
    # It triggers a separate DeprecationWarning when the vector has datetimes:
    # https://github.com/numpy/numpy/issues/13548
    # This is considered a bug by numpy and will likely go away.
    with warnings.catch_warnings():
        warnings.simplefilter(
            action='ignore', category=(FutureWarning, DeprecationWarning)
        )
        if np.isin(vector, [0, 1]).all():
            return VariableType(boolean_type)

    # Defer to positive pandas tests
    if pd.api.types.is_numeric_dtype(vector):
        return VariableType("numeric")

    if pd.api.types.is_datetime64_dtype(vector):
        return VariableType("datetime")

    # --- If we get to here, we need to check the entries

    # Check for a collection where everything is a number

    def all_numeric(x):
        for x_i in x:
            if not isinstance(x_i, Number):
                return False
        return True

    if all_numeric(vector):
        return VariableType("numeric")

    # Check for a collection where everything is a datetime

    def all_datetime(x):
        for x_i in x:
            if not isinstance(x_i, (datetime, np.datetime64)):
                return False
        return True

    if all_datetime(vector):
        return VariableType("datetime")

    # Otherwise, our final fallback is to consider things categorical

    return VariableType("categorical")
