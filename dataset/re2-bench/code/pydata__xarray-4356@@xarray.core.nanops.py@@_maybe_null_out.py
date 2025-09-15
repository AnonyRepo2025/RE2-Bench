import numpy as np
from . import dtypes, nputils, utils
from .duck_array_ops import _dask_or_eager_func, count, fillna, isnull, where_method
from .pycompat import dask_array_type
import dask.array as dask_array
from . import dask_array_compat
from .duck_array_ops import _dask_or_eager_func, count, fillna, where_method



def _maybe_null_out(result, axis, mask, min_count=1):
    """
    xarray version of pandas.core.nanops._maybe_null_out
    """

    if axis is not None and getattr(result, "ndim", False):
        null_mask = (np.take(mask.shape, axis).prod() - mask.sum(axis) - min_count) < 0
        if null_mask.any():
            dtype, fill_value = dtypes.maybe_promote(result.dtype)
            result = result.astype(dtype)
            result[null_mask] = fill_value

    elif getattr(result, "dtype", None) not in dtypes.NAT_TYPES:
        null_mask = mask.size - mask.sum()
        if null_mask < min_count:
            result = np.nan

    return result
