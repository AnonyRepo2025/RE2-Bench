from __future__ import annotations
import contextlib
import datetime
import inspect
import warnings
from functools import partial
import numpy as np
import pandas as pd
from numpy import all as array_all  # noqa
from numpy import any as array_any  # noqa
from numpy import zeros_like  # noqa
from numpy import around, broadcast_to  # noqa
from numpy import concatenate as _concatenate
from numpy import einsum, isclose, isin, isnan, isnat  # noqa
from numpy import stack as _stack
from numpy import take, tensordot, transpose, unravel_index  # noqa
from numpy import where as _where
from . import dask_array_compat, dask_array_ops, dtypes, npcompat, nputils
from .nputils import nanfirst, nanlast
from .pycompat import cupy_array_type, dask_array_type, is_duck_dask_array
from .utils import is_duck_array
import dask.array as dask_array
from dask.base import tokenize
from . import nanops
import datetime as dt
from .common import _contains_cftime_datetimes
from bottleneck import push
import cupy as cp

pandas_isnull = _dask_or_eager_func("isnull", eager_module=pd, dask_module=dask_array)
around.__doc__ = str.replace(
    around.__doc__ or "",
    "array([0.,  2.])",
    "array([0., 2.])",
)
around.__doc__ = str.replace(
    around.__doc__ or "",
    "array([0.,  2.])",
    "array([0., 2.])",
)
around.__doc__ = str.replace(
    around.__doc__ or "",
    "array([0.4,  1.6])",
    "array([0.4, 1.6])",
)
around.__doc__ = str.replace(
    around.__doc__ or "",
    "array([0.,  2.,  2.,  4.,  4.])",
    "array([0., 2., 2., 4., 4.])",
)
around.__doc__ = str.replace(
    around.__doc__ or "",
    (
        '    .. [2] "How Futile are Mindless Assessments of\n'
        '           Roundoff in Floating-Point Computation?", William Kahan,\n'
        "           https://people.eecs.berkeley.edu/~wkahan/Mindless.pdf\n"
    ),
    "",
)
masked_invalid = _dask_or_eager_func(
    "masked_invalid", eager_module=np.ma, dask_module=getattr(dask_array, "ma", None)
)
argmax = _create_nan_agg_method("argmax", coerce_strings=True)
argmin = _create_nan_agg_method("argmin", coerce_strings=True)
max = _create_nan_agg_method("max", coerce_strings=True, invariant_0d=True)
min = _create_nan_agg_method("min", coerce_strings=True, invariant_0d=True)
sum = _create_nan_agg_method("sum", invariant_0d=True)
sum.numeric_only = True
sum.available_min_count = True
std = _create_nan_agg_method("std")
std.numeric_only = True
var = _create_nan_agg_method("var")
var.numeric_only = True
median = _create_nan_agg_method("median", invariant_0d=True)
median.numeric_only = True
prod = _create_nan_agg_method("prod", invariant_0d=True)
prod.numeric_only = True
prod.available_min_count = True
cumprod_1d = _create_nan_agg_method("cumprod", invariant_0d=True)
cumprod_1d.numeric_only = True
cumsum_1d = _create_nan_agg_method("cumsum", invariant_0d=True)
cumsum_1d.numeric_only = True
_mean = _create_nan_agg_method("mean", invariant_0d=True)
mean.numeric_only = True  # type: ignore[attr-defined]
_fail_on_dask_array_input_skipna = partial(
    fail_on_dask_array_input,
    msg="%r with skipna=True is not yet implemented on dask arrays",
)

def _create_nan_agg_method(name, coerce_strings=False, invariant_0d=False):
    from . import nanops

    def f(values, axis=None, skipna=None, **kwargs):
        if kwargs.pop("out", None) is not None:
            raise TypeError(f"`out` is not valid for {name}")

        # The data is invariant in the case of 0d data, so do not
        # change the data (and dtype)
        # See https://github.com/pydata/xarray/issues/4885
        if invariant_0d and axis == ():
            return values

        values = asarray(values)

        if coerce_strings and values.dtype.kind in "SU":
            values = values.astype(object)

        func = None
        if skipna or (skipna is None and values.dtype.kind in "cfO"):
            nanname = "nan" + name
            func = getattr(nanops, nanname)
        else:
            if name in ["sum", "prod"]:
                kwargs.pop("min_count", None)

            if hasattr(values, "__array_namespace__"):
                xp = values.__array_namespace__()
                func = getattr(xp, name)
            else:
                func = getattr(np, name)

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "All-NaN slice encountered")
                return func(values, axis=axis, **kwargs)
        except AttributeError:
            if not is_duck_dask_array(values):
                raise
            try:  # dask/dask#3133 dask sometimes needs dtype argument
                # if func does not accept dtype, then raises TypeError
                return func(values, axis=axis, dtype=values.dtype, **kwargs)
            except (AttributeError, TypeError):
                raise NotImplementedError(
                    f"{name} is not yet implemented on dask arrays"
                )

    f.__name__ = name
    return f
