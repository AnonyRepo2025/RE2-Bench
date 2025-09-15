

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

def variable_type(vector: Series, boolean_type: Literal['numeric', 'categorical', 'boolean']='numeric', strict_boolean: bool=False) -> VarType:
    if isinstance(getattr(vector, 'dtype', None), pd.CategoricalDtype):
        return VarType('categorical')
    if pd.isna(vector).all():
        return VarType('numeric')
    vector = vector.dropna()
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=(FutureWarning, DeprecationWarning))
        if strict_boolean:
            if isinstance(vector.dtype, pd.core.dtypes.base.ExtensionDtype):
                boolean_dtypes = ['bool', 'boolean']
            else:
                boolean_dtypes = ['bool']
            boolean_vector = vector.dtype in boolean_dtypes
        else:
            boolean_vector = bool(np.isin(vector, [0, 1]).all())
        if boolean_vector:
            return VarType(boolean_type)
    if pd.api.types.is_numeric_dtype(vector):
        return VarType('numeric')
    if pd.api.types.is_datetime64_dtype(vector):
        return VarType('datetime')

    def all_numeric(x):
        for x_i in x:
            if not isinstance(x_i, Number):
                return False
        return True
    if all_numeric(vector):
        return VarType('numeric')

    def all_datetime(x):
        for x_i in x:
            if not isinstance(x_i, (datetime, np.datetime64)):
                return False
        return True
    if all_datetime(vector):
        return VarType('datetime')
    return VarType('categorical')