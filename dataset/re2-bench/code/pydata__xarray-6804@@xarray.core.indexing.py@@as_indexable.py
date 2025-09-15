from __future__ import annotations
import enum
import functools
import operator
from collections import Counter, defaultdict
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import timedelta
from html import escape
from typing import TYPE_CHECKING, Any, Callable, Hashable, Iterable, Mapping
import numpy as np
import pandas as pd
from packaging.version import Version
from . import duck_array_ops
from .npcompat import DTypeLike
from .nputils import NumpyVIndexAdapter
from .options import OPTIONS
from .pycompat import dask_version, integer_types, is_duck_dask_array, sparse_array_type
from .types import T_Xarray
from .utils import (
    NDArrayMixin,
    either_dict_or_kwargs,
    get_valid_numpy_dtype,
    safe_cast_to_index,
    to_0d_array,
)
from .indexes import Index
from .variable import Variable
from .dataarray import DataArray
import dask.array as da
from .formatting import format_array_flat
from .formatting import short_numpy_repr
import sparse

LazilyOuterIndexedArray = LazilyIndexedArray

def as_indexable(array):
    """
    This function always returns a ExplicitlyIndexed subclass,
    so that the vectorized indexing is always possible with the returned
    object.
    """
    if isinstance(array, ExplicitlyIndexed):
        return array
    if isinstance(array, np.ndarray):
        return NumpyIndexingAdapter(array)
    if isinstance(array, pd.Index):
        return PandasIndexingAdapter(array)
    if is_duck_dask_array(array):
        return DaskIndexingAdapter(array)
    if hasattr(array, "__array_function__"):
        return NdArrayLikeIndexingAdapter(array)
    if hasattr(array, "__array_namespace__"):
        return ArrayApiIndexingAdapter(array)

    raise TypeError(f"Invalid array type: {type(array)}")
