from __future__ import annotations
import functools
import itertools
import operator
import warnings
from collections import Counter
from typing import (
    TYPE_CHECKING,
    AbstractSet,
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)
import numpy as np
from . import dtypes, duck_array_ops, utils
from .alignment import align, deep_align
from .merge import merge_attrs, merge_coordinates_without_align
from .options import OPTIONS, _get_keep_attrs
from .pycompat import is_duck_dask_array
from .utils import is_dict_like
from .variable import Variable
from .coordinates import Coordinates
from .dataset import Dataset
from .types import T_Xarray
from .dataarray import DataArray
from .dataset import Dataset
from .dataset import Dataset
from .groupby import _dummy_copy
from .groupby import GroupBy, peek_at
from .variable import Variable
from .variable import Variable, as_compatible_data
from .dataarray import DataArray
from .groupby import GroupBy
from .variable import Variable
from .dataarray import DataArray
from .dataarray import DataArray
from .dataarray import DataArray
from .variable import Variable
from .dataarray import DataArray
from .missing import get_clean_interp_index
from .dataarray import DataArray
from dask.array.core import unify_chunks
import dask.array
import dask.array as da

_NO_FILL_VALUE = utils.ReprObject("<no-fill-value>")
_DEFAULT_NAME = utils.ReprObject("<default-name>")
_JOINS_WITHOUT_FILL_VALUES = frozenset({"inner", "exact"})
SLICE_NONE = slice(None)

def _cov_corr(da_a, da_b, dim=None, ddof=0, method=None):
    """
    Internal method for xr.cov() and xr.corr() so only have to
    sanitize the input arrays once and we don't repeat code.
    """
    # 1. Broadcast the two arrays
    da_a, da_b = align(da_a, da_b, join="inner", copy=False)

    # 2. Ignore the nans
    valid_values = da_a.notnull() & da_b.notnull()
    valid_count = valid_values.sum(dim) - ddof

    def _get_valid_values(da, other):
        """
        Function to lazily mask da_a and da_b
        following a similar approach to
        https://github.com/pydata/xarray/pull/4559
        """
        missing_vals = np.logical_or(da.isnull(), other.isnull())
        if missing_vals.any():
            da = da.where(~missing_vals)
            return da
        else:
            # ensure consistent return dtype
            return da.astype(float)

    da_a = da_a.map_blocks(_get_valid_values, args=[da_b])
    da_b = da_b.map_blocks(_get_valid_values, args=[da_a])

    # 3. Detrend along the given dim
    demeaned_da_a = da_a - da_a.mean(dim=dim)
    demeaned_da_b = da_b - da_b.mean(dim=dim)

    # 4. Compute covariance along the given dim
    # N.B. `skipna=False` is required or there is a bug when computing
    # auto-covariance. E.g. Try xr.cov(da,da) for
    # da = xr.DataArray([[1, 2], [1, np.nan]], dims=["x", "time"])
    cov = (demeaned_da_a * demeaned_da_b).sum(dim=dim, skipna=True, min_count=1) / (
        valid_count
    )

    if method == "cov":
        return cov

    else:
        # compute std + corr
        da_a_std = da_a.std(dim=dim)
        da_b_std = da_b.std(dim=dim)
        corr = cov / (da_a_std * da_b_std)
        return corr
