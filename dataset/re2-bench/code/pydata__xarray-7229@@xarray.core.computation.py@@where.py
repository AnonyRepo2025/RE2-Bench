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
    Hashable,
    Iterable,
    Mapping,
    Sequence,
    TypeVar,
    Union,
    overload,
)
import numpy as np
from . import dtypes, duck_array_ops, utils
from .alignment import align, deep_align
from .common import zeros_like
from .duck_array_ops import datetime_to_numeric
from .indexes import Index, filter_indexes_from_coords
from .merge import merge_attrs, merge_coordinates_without_align
from .options import OPTIONS, _get_keep_attrs
from .pycompat import is_duck_dask_array
from .types import T_DataArray
from .utils import is_dict_like, is_scalar
from .variable import Variable
from .coordinates import Coordinates
from .dataarray import DataArray
from .dataset import Dataset
from .types import CombineAttrsOptions, JoinOptions
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
from .dataset import Dataset
from .dataset import Dataset
from .dataarray import DataArray
from dask.array.core import unify_chunks
import dask.array
import dask.array as da

_NO_FILL_VALUE = utils.ReprObject("<no-fill-value>")
_DEFAULT_NAME = utils.ReprObject("<default-name>")
_JOINS_WITHOUT_FILL_VALUES = frozenset({"inner", "exact"})
SLICE_NONE = slice(None)
_T = TypeVar("_T", bound=Union["Dataset", "DataArray"])
_U = TypeVar("_U", bound=Union["Dataset", "DataArray"])
_V = TypeVar("_V", bound=Union["Dataset", "DataArray"])

def where(cond, x, y, keep_attrs=None):
    """Return elements from `x` or `y` depending on `cond`.

    Performs xarray-like broadcasting across input arguments.

    All dimension coordinates on `x` and `y`  must be aligned with each
    other and with `cond`.

    Parameters
    ----------
    cond : scalar, array, Variable, DataArray or Dataset
        When True, return values from `x`, otherwise returns values from `y`.
    x : scalar, array, Variable, DataArray or Dataset
        values to choose from where `cond` is True
    y : scalar, array, Variable, DataArray or Dataset
        values to choose from where `cond` is False
    keep_attrs : bool or str or callable, optional
        How to treat attrs. If True, keep the attrs of `x`.

    Returns
    -------
    Dataset, DataArray, Variable or array
        In priority order: Dataset, DataArray, Variable or array, whichever
        type appears as an input argument.

    Examples
    --------
    >>> x = xr.DataArray(
    ...     0.1 * np.arange(10),
    ...     dims=["lat"],
    ...     coords={"lat": np.arange(10)},
    ...     name="sst",
    ... )
    >>> x
    <xarray.DataArray 'sst' (lat: 10)>
    array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    Coordinates:
      * lat      (lat) int64 0 1 2 3 4 5 6 7 8 9

    >>> xr.where(x < 0.5, x, x * 100)
    <xarray.DataArray 'sst' (lat: 10)>
    array([ 0. ,  0.1,  0.2,  0.3,  0.4, 50. , 60. , 70. , 80. , 90. ])
    Coordinates:
      * lat      (lat) int64 0 1 2 3 4 5 6 7 8 9

    >>> y = xr.DataArray(
    ...     0.1 * np.arange(9).reshape(3, 3),
    ...     dims=["lat", "lon"],
    ...     coords={"lat": np.arange(3), "lon": 10 + np.arange(3)},
    ...     name="sst",
    ... )
    >>> y
    <xarray.DataArray 'sst' (lat: 3, lon: 3)>
    array([[0. , 0.1, 0.2],
           [0.3, 0.4, 0.5],
           [0.6, 0.7, 0.8]])
    Coordinates:
      * lat      (lat) int64 0 1 2
      * lon      (lon) int64 10 11 12

    >>> xr.where(y.lat < 1, y, -1)
    <xarray.DataArray (lat: 3, lon: 3)>
    array([[ 0. ,  0.1,  0.2],
           [-1. , -1. , -1. ],
           [-1. , -1. , -1. ]])
    Coordinates:
      * lat      (lat) int64 0 1 2
      * lon      (lon) int64 10 11 12

    >>> cond = xr.DataArray([True, False], dims=["x"])
    >>> x = xr.DataArray([1, 2], dims=["y"])
    >>> xr.where(cond, x, 0)
    <xarray.DataArray (x: 2, y: 2)>
    array([[1, 2],
           [0, 0]])
    Dimensions without coordinates: x, y

    See Also
    --------
    numpy.where : corresponding numpy function
    Dataset.where, DataArray.where :
        equivalent methods
    """
    from .dataset import Dataset

    if keep_attrs is None:
        keep_attrs = _get_keep_attrs(default=False)

    # alignment for three arguments is complicated, so don't support it yet
    result = apply_ufunc(
        duck_array_ops.where,
        cond,
        x,
        y,
        join="exact",
        dataset_join="exact",
        dask="allowed",
        keep_attrs=keep_attrs,
    )

    # keep the attributes of x, the second parameter, by default to
    # be consistent with the `where` method of `DataArray` and `Dataset`
    # rebuild the attrs from x at each level of the output, which could be
    # Dataset, DataArray, or Variable, and also handle coords
    if keep_attrs is True:
        if isinstance(y, Dataset) and not isinstance(x, Dataset):
            # handle special case where x gets promoted to Dataset
            result.attrs = {}
            if getattr(x, "name", None) in result.data_vars:
                result[x.name].attrs = getattr(x, "attrs", {})
        else:
            # otherwise, fill in global attrs and variable attrs (if they exist)
            result.attrs = getattr(x, "attrs", {})
            for v in getattr(result, "data_vars", []):
                result[v].attrs = getattr(getattr(x, v, None), "attrs", {})
        for c in getattr(result, "coords", []):
            # always fill coord attrs of x
            result[c].attrs = getattr(getattr(x, c, None), "attrs", {})

    return result
