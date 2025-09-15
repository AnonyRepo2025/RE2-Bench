from __future__ import annotations
import copy
import datetime
import inspect
import itertools
import math
import sys
import warnings
from collections import defaultdict
from html import escape
from numbers import Number
from operator import methodcaller
from os import PathLike
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    Callable,
    Collection,
    Generic,
    Hashable,
    Iterable,
    Iterator,
    Literal,
    Mapping,
    MutableMapping,
    Sequence,
    cast,
    overload,
)
import numpy as np
import pandas as pd
from ..coding.calendar_ops import convert_calendar, interp_calendar
from ..coding.cftimeindex import CFTimeIndex, _parse_array_of_cftime_strings
from ..plot.dataset_plot import _Dataset_PlotMethods
from . import alignment
from . import dtypes as xrdtypes
from . import duck_array_ops, formatting, formatting_html, ops, utils
from ._reductions import DatasetReductions
from .alignment import _broadcast_helper, _get_broadcast_dims_map_common_coords, align
from .arithmetic import DatasetArithmetic
from .common import DataWithCoords, _contains_datetime_like_objects, get_chunksizes
from .computation import unify_chunks
from .coordinates import DatasetCoordinates, assert_coordinate_consistent
from .duck_array_ops import datetime_to_numeric
from .indexes import (
    Index,
    Indexes,
    PandasIndex,
    PandasMultiIndex,
    assert_no_index_corrupted,
    create_default_index_implicit,
    filter_indexes_from_coords,
    isel_indexes,
    remove_unused_levels_categories,
    roll_indexes,
)
from .indexing import is_fancy_indexer, map_index_queries
from .merge import (
    dataset_merge_method,
    dataset_update_method,
    merge_coordinates_without_align,
    merge_data_and_coords,
)
from .missing import get_clean_interp_index
from .npcompat import QUANTILE_METHODS, ArrayLike
from .options import OPTIONS, _get_keep_attrs
from .pycompat import is_duck_dask_array, sparse_array_type
from .types import T_Dataset
from .utils import (
    Default,
    Frozen,
    HybridMappingProxy,
    OrderedSet,
    _default,
    decode_numpy_dict_values,
    drop_dims_from_indexers,
    either_dict_or_kwargs,
    infix_dims,
    is_dict_like,
    is_scalar,
    maybe_wrap_array,
)
from .variable import (
    IndexVariable,
    Variable,
    as_variable,
    broadcast_variables,
    calculate_dimensions,
)
from ..backends import AbstractDataStore, ZarrStore
from ..backends.api import T_NetcdfEngine, T_NetcdfTypes
from .coordinates import Coordinates
from .dataarray import DataArray
from .groupby import DatasetGroupBy
from .merge import CoercibleMapping
from .resample import DatasetResample
from .rolling import DatasetCoarsen, DatasetRolling
from .types import (
        CFCalendar,
        CoarsenBoundaryOptions,
        CombineAttrsOptions,
        CompatOptions,
        DatetimeUnitOptions,
        Dims,
        ErrorOptions,
        ErrorOptionsWithWarn,
        InterpOptions,
        JoinOptions,
        PadModeOptions,
        PadReflectOptions,
        QueryEngineOptions,
        QueryParserOptions,
        ReindexMethodOptions,
        SideOptions,
        T_Xarray,
    )
from .weighted import DatasetWeighted
from .dataarray import DataArray
import dask.array as da
from dask.base import tokenize
from dask.delayed import Delayed
from dask.dataframe import DataFrame as DaskDataFrame
from dask.base import normalize_token
import dask
import dask
import dask.array as da
import dask.array as da
import dask
from dask import is_dask_collection
from dask.highlevelgraph import HighLevelGraph
from dask.optimization import cull
from .dataarray import DataArray
from .dataarray import DataArray
from .alignment import align
from .dataarray import DataArray
from ..backends.api import dump_to_store
from ..backends.api import to_netcdf
from ..backends.api import to_zarr
from ..coding.cftimeindex import CFTimeIndex
from .dataarray import DataArray
from .dataarray import DataArray
from . import missing
from .concat import concat
from .dataarray import DataArray
from .missing import _apply_over_vars_with_dim, interp_na
from .missing import _apply_over_vars_with_dim, ffill
from .missing import _apply_over_vars_with_dim, bfill
from .dataarray import DataArray
from sparse import COO
import dask.array as da
import dask.dataframe as dd
from .dataarray import DataArray
from .groupby import GroupBy
from .dataarray import DataArray
from .groupby import GroupBy
from .dataarray import DataArray
from .variable import Variable
from .variable import Variable
from .parallel import map_blocks
from .dataarray import DataArray
from scipy.optimize import curve_fit
from .alignment import broadcast
from .computation import apply_ufunc
from .dataarray import _THIS_ARRAY, DataArray
from .groupby import DatasetGroupBy
from .groupby import DatasetGroupBy
from .weighted import DatasetWeighted
from .rolling import DatasetRolling
from .rolling import DatasetCoarsen
from .resample import DatasetResample
import dask.array as da
import dask
import itertools
from dask.highlevelgraph import HighLevelGraph
from dask import sharedict
from dask.base import flatten, replace_name_in_key

_DATETIMEINDEX_COMPONENTS = [
    "year",
    "month",
    "day",
    "hour",
    "minute",
    "second",
    "microsecond",
    "nanosecond",
    "date",
    "time",
    "dayofyear",
    "weekofyear",
    "dayofweek",
    "quarter",
]

class Dataset(DataWithCoords, DatasetReductions, DatasetArithmetic, Mapping[(Hashable, 'DataArray')]
):
    __slots__ = (
        "_attrs",
        "_cache",
        "_coord_names",
        "_dims",
        "_encoding",
        "_close",
        "_indexes",
        "_variables",
        "__weakref__",
    )
    __hash__ = None  # type: ignore[assignment]
    plot = utils.UncachedAccessor(_Dataset_PlotMethods)
    def copy(
        self: T_Dataset, deep: bool = False, data: Mapping[Any, ArrayLike] | None = None
    ) -> T_Dataset:
        """Returns a copy of this dataset.

        If `deep=True`, a deep copy is made of each of the component variables.
        Otherwise, a shallow copy of each of the component variable is made, so
        that the underlying memory region of the new dataset is the same as in
        the original dataset.

        Use `data` to create a new object with the same structure as
        original but entirely new data.

        Parameters
        ----------
        deep : bool, default: False
            Whether each component variable is loaded into memory and copied onto
            the new object. Default is False.
        data : dict-like or None, optional
            Data to use in the new object. Each item in `data` must have same
            shape as corresponding data variable in original. When `data` is
            used, `deep` is ignored for the data variables and only used for
            coords.

        Returns
        -------
        object : Dataset
            New object with dimensions, attributes, coordinates, name, encoding,
            and optionally data copied from original.

        Examples
        --------
        Shallow copy versus deep copy

        >>> da = xr.DataArray(np.random.randn(2, 3))
        >>> ds = xr.Dataset(
        ...     {"foo": da, "bar": ("x", [-1, 2])},
        ...     coords={"x": ["one", "two"]},
        ... )
        >>> ds.copy()
        <xarray.Dataset>
        Dimensions:  (dim_0: 2, dim_1: 3, x: 2)
        Coordinates:
          * x        (x) <U3 'one' 'two'
        Dimensions without coordinates: dim_0, dim_1
        Data variables:
            foo      (dim_0, dim_1) float64 1.764 0.4002 0.9787 2.241 1.868 -0.9773
            bar      (x) int64 -1 2

        >>> ds_0 = ds.copy(deep=False)
        >>> ds_0["foo"][0, 0] = 7
        >>> ds_0
        <xarray.Dataset>
        Dimensions:  (dim_0: 2, dim_1: 3, x: 2)
        Coordinates:
          * x        (x) <U3 'one' 'two'
        Dimensions without coordinates: dim_0, dim_1
        Data variables:
            foo      (dim_0, dim_1) float64 7.0 0.4002 0.9787 2.241 1.868 -0.9773
            bar      (x) int64 -1 2

        >>> ds
        <xarray.Dataset>
        Dimensions:  (dim_0: 2, dim_1: 3, x: 2)
        Coordinates:
          * x        (x) <U3 'one' 'two'
        Dimensions without coordinates: dim_0, dim_1
        Data variables:
            foo      (dim_0, dim_1) float64 7.0 0.4002 0.9787 2.241 1.868 -0.9773
            bar      (x) int64 -1 2

        Changing the data using the ``data`` argument maintains the
        structure of the original object, but with the new data. Original
        object is unaffected.

        >>> ds.copy(data={"foo": np.arange(6).reshape(2, 3), "bar": ["a", "b"]})
        <xarray.Dataset>
        Dimensions:  (dim_0: 2, dim_1: 3, x: 2)
        Coordinates:
          * x        (x) <U3 'one' 'two'
        Dimensions without coordinates: dim_0, dim_1
        Data variables:
            foo      (dim_0, dim_1) int64 0 1 2 3 4 5
            bar      (x) <U1 'a' 'b'

        >>> ds
        <xarray.Dataset>
        Dimensions:  (dim_0: 2, dim_1: 3, x: 2)
        Coordinates:
          * x        (x) <U3 'one' 'two'
        Dimensions without coordinates: dim_0, dim_1
        Data variables:
            foo      (dim_0, dim_1) float64 7.0 0.4002 0.9787 2.241 1.868 -0.9773
            bar      (x) int64 -1 2

        See Also
        --------
        pandas.DataFrame.copy
        """
        if data is None:
            data = {}
        elif not utils.is_dict_like(data):
            raise ValueError("Data must be dict-like")

        if data:
            var_keys = set(self.data_vars.keys())
            data_keys = set(data.keys())
            keys_not_in_vars = data_keys - var_keys
            if keys_not_in_vars:
                raise ValueError(
                    "Data must only contain variables in original "
                    "dataset. Extra variables: {}".format(keys_not_in_vars)
                )
            keys_missing_from_data = var_keys - data_keys
            if keys_missing_from_data:
                raise ValueError(
                    "Data must contain all variables in original "
                    "dataset. Data is missing {}".format(keys_missing_from_data)
                )

        indexes, index_vars = self.xindexes.copy_indexes(deep=deep)

        variables = {}
        for k, v in self._variables.items():
            if k in index_vars:
                variables[k] = index_vars[k]
            else:
                variables[k] = v.copy(deep=deep, data=data.get(k))

        attrs = copy.deepcopy(self._attrs) if deep else copy.copy(self._attrs)
        encoding = copy.deepcopy(self._encoding) if deep else copy.copy(self._encoding)

        return self._replace(variables, indexes=indexes, attrs=attrs, encoding=encoding)
    @property
    def xindexes(self) -> Indexes[Index]:
        """Mapping of xarray Index objects used for label based indexing."""
        return Indexes(self._indexes, {k: self._variables[k] for k in self._indexes})