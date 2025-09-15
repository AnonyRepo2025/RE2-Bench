from __future__ import annotations
import datetime
import warnings
from collections.abc import Hashable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Callable, Generic, Literal, TypeVar, Union, cast
import numpy as np
import pandas as pd
from xarray.core import dtypes, duck_array_ops, nputils, ops
from xarray.core._aggregations import (
    DataArrayGroupByAggregations,
    DatasetGroupByAggregations,
)
from xarray.core.alignment import align
from xarray.core.arithmetic import DataArrayGroupbyArithmetic, DatasetGroupbyArithmetic
from xarray.core.common import ImplementsArrayReduce, ImplementsDatasetReduce
from xarray.core.concat import concat
from xarray.core.formatting import format_array_flat
from xarray.core.indexes import (
    create_default_index_implicit,
    filter_indexes_from_coords,
    safe_cast_to_index,
)
from xarray.core.options import _get_keep_attrs
from xarray.core.pycompat import integer_types
from xarray.core.types import Dims, QuantileMethods, T_Xarray
from xarray.core.utils import (
    either_dict_or_kwargs,
    hashable,
    is_scalar,
    maybe_wrap_array,
    peek_at,
)
from xarray.core.variable import IndexVariable, Variable
from numpy.typing import ArrayLike
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset
from xarray.core.types import DatetimeLike, SideOptions
from xarray.core.utils import Frozen
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset
from xarray.core.dataarray import DataArray
from xarray.core.dataarray import DataArray
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset
from flox.xarray import xarray_reduce
from xarray.core.dataset import Dataset
from xarray import CFTimeIndex
from xarray.core.resample_cftime import CFTimeGrouper

T_Group = TypeVar("T_Group", bound=Union["DataArray", "IndexVariable", _DummyGroup])

class GroupBy(Generic[T_Xarray]
):
    __slots__ = (
        "_full_index",
        "_inserted_dims",
        "_group",
        "_group_dim",
        "_group_indices",
        "_groups",
        "_obj",
        "_restore_coord_dims",
        "_stacked_dim",
        "_unique_coord",
        "_dims",
        "_sizes",
        "_squeeze",
        # Save unstacked object for flox
        "_original_obj",
        "_original_group",
        "_bins",
    )
    def _get_index_and_items(self, index, grouper):
        first_items = grouper.first_items(index)
        full_index = first_items.index
        if first_items.isnull().any():
            first_items = first_items.dropna()
        return full_index, first_items