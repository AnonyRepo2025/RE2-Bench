import warnings
from contextlib import suppress
from html import escape
from textwrap import dedent
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    Iterator,
    List,
    Mapping,
    Tuple,
    TypeVar,
    Union,
)
import numpy as np
import pandas as pd
from . import dtypes, duck_array_ops, formatting, formatting_html, ops
from .arithmetic import SupportsArithmetic
from .npcompat import DTypeLike
from .options import OPTIONS, _get_keep_attrs
from .pycompat import dask_array_type
from .rolling_exp import RollingExp
from .utils import Frozen, either_dict_or_kwargs, is_scalar
from .dataarray import DataArray
from .dataset import Dataset
from .variable import Variable
from .variable import Variable
from .dataarray import DataArray
from .resample import RESAMPLE_DIM
from ..coding.cftimeindex import CFTimeIndex
from .alignment import align
from .dataarray import DataArray
from .dataset import Dataset
from .computation import apply_ufunc
from .dataset import Dataset
from .dataarray import DataArray
from .variable import Variable
import dask.array
from cftime import datetime as cftime_datetime
from .resample_cftime import CFTimeGrouper

ALL_DIMS = ...
C = TypeVar("C")
T = TypeVar("T")

def full_like(other, fill_value, dtype: DTypeLike = None):
    """Return a new object with the same shape and type as a given object.

    Parameters
    ----------
    other : DataArray, Dataset, or Variable
        The reference object in input
    fill_value : scalar
        Value to fill the new object with before returning it.
    dtype : dtype, optional
        dtype of the new array. If omitted, it defaults to other.dtype.

    Returns
    -------
    out : same as object
        New object with the same shape and type as other, with the data
        filled with fill_value. Coords will be copied from other.
        If other is based on dask, the new one will be as well, and will be
        split in the same chunks.

    Examples
    --------

    >>> import numpy as np
    >>> import xarray as xr
    >>> x = xr.DataArray(
    ...     np.arange(6).reshape(2, 3),
    ...     dims=["lat", "lon"],
    ...     coords={"lat": [1, 2], "lon": [0, 1, 2]},
    ... )
    >>> x
    <xarray.DataArray (lat: 2, lon: 3)>
    array([[0, 1, 2],
           [3, 4, 5]])
    Coordinates:
    * lat      (lat) int64 1 2
    * lon      (lon) int64 0 1 2

    >>> xr.full_like(x, 1)
    <xarray.DataArray (lat: 2, lon: 3)>
    array([[1, 1, 1],
           [1, 1, 1]])
    Coordinates:
    * lat      (lat) int64 1 2
    * lon      (lon) int64 0 1 2

    >>> xr.full_like(x, 0.5)
    <xarray.DataArray (lat: 2, lon: 3)>
    array([[0, 0, 0],
           [0, 0, 0]])
    Coordinates:
    * lat      (lat) int64 1 2
    * lon      (lon) int64 0 1 2

    >>> xr.full_like(x, 0.5, dtype=np.double)
    <xarray.DataArray (lat: 2, lon: 3)>
    array([[0.5, 0.5, 0.5],
           [0.5, 0.5, 0.5]])
    Coordinates:
    * lat      (lat) int64 1 2
    * lon      (lon) int64 0 1 2

    >>> xr.full_like(x, np.nan, dtype=np.double)
    <xarray.DataArray (lat: 2, lon: 3)>
    array([[nan, nan, nan],
           [nan, nan, nan]])
    Coordinates:
    * lat      (lat) int64 1 2
    * lon      (lon) int64 0 1 2

    See also
    --------

    zeros_like
    ones_like

    """
    from .dataarray import DataArray
    from .dataset import Dataset
    from .variable import Variable

    if not is_scalar(fill_value):
        raise ValueError(f"fill_value must be scalar. Received {fill_value} instead.")

    if isinstance(other, Dataset):
        data_vars = {
            k: _full_like_variable(v, fill_value, dtype)
            for k, v in other.data_vars.items()
        }
        return Dataset(data_vars, coords=other.coords, attrs=other.attrs)
    elif isinstance(other, DataArray):
        return DataArray(
            _full_like_variable(other.variable, fill_value, dtype),
            dims=other.dims,
            coords=other.coords,
            attrs=other.attrs,
            name=other.name,
        )
    elif isinstance(other, Variable):
        return _full_like_variable(other, fill_value, dtype)
    else:
        raise TypeError("Expected DataArray, Dataset, or Variable")
