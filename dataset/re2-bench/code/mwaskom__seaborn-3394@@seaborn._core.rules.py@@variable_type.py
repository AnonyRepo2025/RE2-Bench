from __future__ import annotations
import warnings
from collections import UserString
from numbers import Number
from datetime import datetime
import numpy as np
import pandas as pd
from typing import TYPE_CHECKING
from typing import Literal
from pandas import Series



def variable_type(
    vector: Series,
    boolean_type: Literal["numeric", "categorical", "boolean"] = "numeric",
    strict_boolean: bool = False,
) -> VarType:
    """
    Determine whether a vector contains numeric, categorical, or datetime data.

    This function differs from the pandas typing API in a few ways:

    - Python sequences or object-typed PyData objects are considered numeric if
      all of their entries are numeric.
    - String or mixed-type data are considered categorical even if not
      explicitly represented as a :class:`pandas.api.types.CategoricalDtype`.
    - There is some flexibility about how to treat binary / boolean data.

    Parameters
    ----------
    vector : :func:`pandas.Series`, :func:`numpy.ndarray`, or Python sequence
        Input data to test.
    boolean_type : 'numeric', 'categorical', or 'boolean'
        Type to use for vectors containing only 0s and 1s (and NAs).
    strict_boolean : bool
        If True, only consider data to be boolean when the dtype is bool or Boolean.

    Returns
    -------
    var_type : 'numeric', 'categorical', or 'datetime'
        Name identifying the type of data in the vector.
    """

    # If a categorical dtype is set, infer categorical
    if isinstance(getattr(vector, 'dtype', None), pd.CategoricalDtype):
        return VarType("categorical")

    # Special-case all-na data, which is always "numeric"
    if pd.isna(vector).all():
        return VarType("numeric")

    # Now drop nulls to simplify further type inference
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
            action='ignore',
            category=(FutureWarning, DeprecationWarning)  # type: ignore  # mypy bug?
        )
        if strict_boolean:
            if isinstance(vector.dtype, pd.core.dtypes.base.ExtensionDtype):
                boolean_dtypes = ["bool", "boolean"]
            else:
                boolean_dtypes = ["bool"]
            boolean_vector = vector.dtype in boolean_dtypes
        else:
            boolean_vector = bool(np.isin(vector, [0, 1]).all())
        if boolean_vector:
            return VarType(boolean_type)

    # Defer to positive pandas tests
    if pd.api.types.is_numeric_dtype(vector):
        return VarType("numeric")

    if pd.api.types.is_datetime64_dtype(vector):
        return VarType("datetime")

    # --- If we get to here, we need to check the entries

    # Check for a collection where everything is a number

    def all_numeric(x):
        for x_i in x:
            if not isinstance(x_i, Number):
                return False
        return True

    if all_numeric(vector):
        return VarType("numeric")

    # Check for a collection where everything is a datetime

    def all_datetime(x):
        for x_i in x:
            if not isinstance(x_i, (datetime, np.datetime64)):
                return False
        return True

    if all_datetime(vector):
        return VarType("datetime")

    # Otherwise, our final fallback is to consider things categorical

    return VarType("categorical")
