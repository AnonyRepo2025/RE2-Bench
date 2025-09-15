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
        Ellipsis,
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
    @property
    def xindexes(self) -> Indexes[Index]:
        """Mapping of xarray Index objects used for label based indexing."""
        return Indexes(self._indexes, {k: self._variables[k] for k in self._indexes})
    def reset_index(
        self: T_Dataset,
        dims_or_levels: Hashable | Sequence[Hashable],
        drop: bool = False,
    ) -> T_Dataset:
        """Reset the specified index(es) or multi-index level(s).

        This legacy method is specific to pandas (multi-)indexes and
        1-dimensional "dimension" coordinates. See the more generic
        :py:meth:`~Dataset.drop_indexes` and :py:meth:`~Dataset.set_xindex`
        method to respectively drop and set pandas or custom indexes for
        arbitrary coordinates.

        Parameters
        ----------
        dims_or_levels : Hashable or Sequence of Hashable
            Name(s) of the dimension(s) and/or multi-index level(s) that will
            be reset.
        drop : bool, default: False
            If True, remove the specified indexes and/or multi-index levels
            instead of extracting them as new coordinates (default: False).

        Returns
        -------
        obj : Dataset
            Another dataset, with this dataset's data but replaced coordinates.

        See Also
        --------
        Dataset.set_index
        Dataset.set_xindex
        Dataset.drop_indexes
        """
        if isinstance(dims_or_levels, str) or not isinstance(dims_or_levels, Sequence):
            dims_or_levels = [dims_or_levels]

        invalid_coords = set(dims_or_levels) - set(self._indexes)
        if invalid_coords:
            raise ValueError(
                f"{tuple(invalid_coords)} are not coordinates with an index"
            )

        drop_indexes: set[Hashable] = set()
        drop_variables: set[Hashable] = set()
        seen: set[Index] = set()
        new_indexes: dict[Hashable, Index] = {}
        new_variables: dict[Hashable, Variable] = {}

        def drop_or_convert(var_names):
            if drop:
                drop_variables.update(var_names)
            else:
                base_vars = {
                    k: self._variables[k].to_base_variable() for k in var_names
                }
                new_variables.update(base_vars)

        for name in dims_or_levels:
            index = self._indexes[name]

            if index in seen:
                continue
            seen.add(index)

            idx_var_names = set(self.xindexes.get_all_coords(name))
            drop_indexes.update(idx_var_names)

            if isinstance(index, PandasMultiIndex):
                # special case for pd.MultiIndex
                level_names = index.index.names
                keep_level_vars = {
                    k: self._variables[k]
                    for k in level_names
                    if k not in dims_or_levels
                }

                if index.dim not in dims_or_levels and keep_level_vars:
                    # do not drop the multi-index completely
                    # instead replace it by a new (multi-)index with dropped level(s)
                    idx = index.keep_levels(keep_level_vars)
                    idx_vars = idx.create_variables(keep_level_vars)
                    new_indexes.update({k: idx for k in idx_vars})
                    new_variables.update(idx_vars)
                    if not isinstance(idx, PandasMultiIndex):
                        # multi-index reduced to single index
                        # backward compatibility: unique level coordinate renamed to dimension
                        drop_variables.update(keep_level_vars)
                    drop_or_convert(
                        [k for k in level_names if k not in keep_level_vars]
                    )
                else:
                    # always drop the multi-index dimension variable
                    drop_variables.add(index.dim)
                    drop_or_convert(level_names)
            else:
                drop_or_convert(idx_var_names)

        indexes = {k: v for k, v in self._indexes.items() if k not in drop_indexes}
        indexes.update(new_indexes)

        variables = {
            k: v for k, v in self._variables.items() if k not in drop_variables
        }
        variables.update(new_variables)

        coord_names = self._coord_names - drop_variables

        return self._replace_with_new_dims(
            variables, coord_names=coord_names, indexes=indexes
        )