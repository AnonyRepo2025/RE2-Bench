import datetime as dt
import warnings
from distutils.version import LooseVersion
from functools import partial
from numbers import Number
from typing import Any, Callable, Dict, Hashable, Sequence, Union
import numpy as np
import pandas as pd
from . import utils
from .common import _contains_datetime_like_objects, ones_like
from .computation import apply_ufunc
from .duck_array_ops import datetime_to_numeric, push, timedelta_to_numeric
from .options import _get_keep_attrs
from .pycompat import is_duck_dask_array
from .utils import OrderedSet, is_scalar
from .variable import Variable, broadcast_variables
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.coding.cftimeindex import CFTimeIndex
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
from scipy import interpolate
import dask.array as da



def ffill(arr, dim=None, limit=None):
    """forward fill missing values"""
    axis = arr.get_axis_num(dim)

    # work around for bottleneck 178
    _limit = limit if limit is not None else arr.shape[axis]

    return apply_ufunc(
        push,
        arr,
        dask="allowed",
        keep_attrs=True,
        output_dtypes=[arr.dtype],
        kwargs=dict(n=_limit, axis=axis),
    ).transpose(*arr.dims)
