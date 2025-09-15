from __future__ import annotations
import copy
import itertools
import math
import numbers
import warnings
from datetime import timedelta
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Hashable,
    Iterable,
    Literal,
    Mapping,
    Sequence,
)
import numpy as np
import pandas as pd
from packaging.version import Version
import xarray as xr  # only for Dataset and DataArray
from . import common, dtypes, duck_array_ops, indexing, nputils, ops, utils
from .arithmetic import VariableArithmetic
from .common import AbstractArray
from .indexing import (
    BasicIndexer,
    OuterIndexer,
    PandasIndexingAdapter,
    VectorizedIndexer,
    as_indexable,
)
from .npcompat import QUANTILE_METHODS, ArrayLike
from .options import OPTIONS, _get_keep_attrs
from .pycompat import (
    DuckArrayModule,
    cupy_array_type,
    dask_array_type,
    integer_types,
    is_duck_dask_array,
    sparse_array_type,
)
from .utils import (
    Frozen,
    NdimSizeLenMixin,
    OrderedSet,
    _default,
    decode_numpy_dict_values,
    drop_dims_from_indexers,
    either_dict_or_kwargs,
    ensure_us_time_resolution,
    infix_dims,
    is_duck_array,
    maybe_coerce_to_str,
)
from .types import (
        ErrorOptionsWithWarn,
        PadModeOptions,
        PadReflectOptions,
        T_Variable,
    )
from .dataarray import DataArray
from .dataarray import DataArray
from .computation import apply_ufunc
from dask.base import normalize_token
import dask.array as da
import sparse
from .computation import apply_ufunc
from .merge import merge_attrs
from .computation import apply_ufunc
import bottleneck as bn
from .computation import apply_ufunc
from .computation import apply_ufunc
from dask.base import normalize_token
from .merge import merge_attrs
from sparse import COO

NON_NUMPY_SUPPORTED_ARRAY_TYPES = (
    (
        indexing.ExplicitlyIndexed,
        pd.Index,
    )
    + dask_array_type
    + cupy_array_type
)
BASIC_INDEXING_TYPES = integer_types + (slice,)
Coordinate = utils.alias(IndexVariable, "Coordinate")

def as_compatible_data(data, fastpath=False):
    """Prepare and wrap data to put in a Variable.

    - If data does not have the necessary attributes, convert it to ndarray.
    - If data has dtype=datetime64, ensure that it has ns precision. If it's a
      pandas.Timestamp, convert it to datetime64.
    - If data is already a pandas or xarray object (other than an Index), just
      use the values.

    Finally, wrap it up with an adapter if necessary.
    """
    from .dataarray import DataArray

    if fastpath and getattr(data, "ndim", 0) > 0:
        # can't use fastpath (yet) for scalars
        return _maybe_wrap_data(data)

    if isinstance(data, (Variable, DataArray)):
        return data.data

    if isinstance(data, NON_NUMPY_SUPPORTED_ARRAY_TYPES):
        return _maybe_wrap_data(data)

    if isinstance(data, tuple):
        data = utils.to_0d_object_array(data)

    if isinstance(data, pd.Timestamp):
        # TODO: convert, handle datetime objects, too
        data = np.datetime64(data.value, "ns")

    if isinstance(data, timedelta):
        data = np.timedelta64(getattr(data, "value", data), "ns")

    # we don't want nested self-described arrays
    if isinstance(data, (pd.Series, pd.Index, pd.DataFrame)):
        data = data.values

    if isinstance(data, np.ma.MaskedArray):
        mask = np.ma.getmaskarray(data)
        if mask.any():
            dtype, fill_value = dtypes.maybe_promote(data.dtype)
            data = np.asarray(data, dtype=dtype)
            data[mask] = fill_value
        else:
            data = np.asarray(data)

    if not isinstance(data, np.ndarray) and (
        hasattr(data, "__array_function__") or hasattr(data, "__array_namespace__")
    ):
        return data

    # validate whether the data is valid data types.
    data = np.asarray(data)

    if isinstance(data, np.ndarray) and data.dtype.kind in "OMm":
        data = _possibly_convert_objects(data)
    return _maybe_wrap_data(data)
