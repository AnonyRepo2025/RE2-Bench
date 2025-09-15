import functools
import warnings
from typing import Any, Callable, Dict
import numpy as np
from . import dtypes, duck_array_ops, utils
from .dask_array_ops import dask_rolling_wrapper
from .ops import inject_reduce_methods
from .options import _get_keep_attrs
from .pycompat import is_duck_dask_array
import bottleneck
from .dataarray import DataArray
from .dataarray import DataArray
from .dataset import Dataset
from .dataset import Dataset
from .dataarray import DataArray
from .dataset import Dataset

_ROLLING_REDUCE_DOCSTRING_TEMPLATE = """\
Reduce this object's data windows by applying `{name}` along its dimension.

Parameters
----------
keep_attrs : bool, default: None
    If True, the attributes (``attrs``) will be copied from the original
    object to the new one. If False, the new object will be returned
    without attributes. If None uses the global default.
**kwargs : dict
    Additional keyword arguments passed on to `{name}`.

Returns
-------
reduced : same type as caller
    New object with `{name}` applied along its rolling dimnension.
"""

class Rolling:
    __slots__ = ("obj", "window", "min_periods", "center", "dim", "keep_attrs")
    _attributes = ("window", "min_periods", "center", "dim", "keep_attrs")
    argmax = _reduce_method("argmax")
    argmin = _reduce_method("argmin")
    max = _reduce_method("max")
    min = _reduce_method("min")
    mean = _reduce_method("mean")
    prod = _reduce_method("prod")
    sum = _reduce_method("sum")
    std = _reduce_method("std")
    var = _reduce_method("var")
    median = _reduce_method("median")
    count.__doc__ = _ROLLING_REDUCE_DOCSTRING_TEMPLATE.format(name="count")
    def _get_keep_attrs(self, keep_attrs):

        if keep_attrs is None:
            # TODO: uncomment the next line and remove the others after the deprecation
            # keep_attrs = _get_keep_attrs(default=True)

            if self.keep_attrs is None:
                keep_attrs = _get_keep_attrs(default=True)
            else:
                keep_attrs = self.keep_attrs

        return keep_attrs