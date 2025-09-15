import contextlib
import datetime
import inspect
import warnings
from distutils.version import LooseVersion
from functools import partial
import numpy as np
import pandas as pd
from . import dask_array_compat, dask_array_ops, dtypes, npcompat, nputils
from .nputils import nanfirst, nanlast
from .pycompat import (
    cupy_array_type,
    dask_array_type,
    is_duck_dask_array,
    sparse_array_type,
)
from .utils import is_duck_array
import dask.array as dask_array
from dask.base import tokenize
from . import nanops
import datetime as dt
from .common import _contains_cftime_datetimes
import sparse
import cupy as cp

moveaxis = npcompat.moveaxis
around = _dask_or_eager_func("around")
isclose = _dask_or_eager_func("isclose")
isnat = np.isnat
isnan = _dask_or_eager_func("isnan")
zeros_like = _dask_or_eager_func("zeros_like")
pandas_isnull = _dask_or_eager_func("isnull", eager_module=pd)
transpose = _dask_or_eager_func("transpose")
_where = _dask_or_eager_func("where", array_args=slice(3))
isin = _dask_or_eager_func("isin", array_args=slice(2))
take = _dask_or_eager_func("take")
broadcast_to = _dask_or_eager_func("broadcast_to")
pad = _dask_or_eager_func("pad", dask_module=dask_array_compat)
_concatenate = _dask_or_eager_func("concatenate", list_of_args=True)
_stack = _dask_or_eager_func("stack", list_of_args=True)
array_all = _dask_or_eager_func("all")
array_any = _dask_or_eager_func("any")
tensordot = _dask_or_eager_func("tensordot", array_args=slice(2))
einsum = _dask_or_eager_func("einsum", array_args=slice(1, None))
masked_invalid = _dask_or_eager_func(
    "masked_invalid", eager_module=np.ma, dask_module=getattr(dask_array, "ma", None)
)
argmax = _create_nan_agg_method("argmax", coerce_strings=True)
argmin = _create_nan_agg_method("argmin", coerce_strings=True)
max = _create_nan_agg_method("max", coerce_strings=True)
min = _create_nan_agg_method("min", coerce_strings=True)
sum = _create_nan_agg_method("sum")
sum.numeric_only = True
sum.available_min_count = True
std = _create_nan_agg_method("std")
std.numeric_only = True
var = _create_nan_agg_method("var")
var.numeric_only = True
median = _create_nan_agg_method("median", dask_module=dask_array_compat)
median.numeric_only = True
prod = _create_nan_agg_method("prod")
prod.numeric_only = True
prod.available_min_count = True
cumprod_1d = _create_nan_agg_method("cumprod")
cumprod_1d.numeric_only = True
cumsum_1d = _create_nan_agg_method("cumsum")
cumsum_1d.numeric_only = True
unravel_index = _dask_or_eager_func("unravel_index")
_mean = _create_nan_agg_method("mean")
mean.numeric_only = True  # type: ignore
_fail_on_dask_array_input_skipna = partial(
    fail_on_dask_array_input,
    msg="%r with skipna=True is not yet implemented on dask arrays",
)

def astype(data, dtype, **kwargs):
    try:
        import sparse
    except ImportError:
        sparse = None

    if (
        sparse is not None
        and isinstance(data, sparse_array_type)
        and LooseVersion(sparse.__version__) < LooseVersion("0.11.0")
        and "casting" in kwargs
    ):
        warnings.warn(
            "The current version of sparse does not support the 'casting' argument. It will be ignored in the call to astype().",
            RuntimeWarning,
            stacklevel=4,
        )
        kwargs.pop("casting")

    return data.astype(dtype, **kwargs)
