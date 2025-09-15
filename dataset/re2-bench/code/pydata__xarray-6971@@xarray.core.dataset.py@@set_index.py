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
    def set_index(
        self,
        indexes: Mapping[Any, Hashable | Sequence[Hashable]] | None = None,
        append: bool = False,
        **indexes_kwargs: Hashable | Sequence[Hashable],
    ) -> Dataset:
        """Set Dataset (multi-)indexes using one or more existing coordinates
        or variables.

        This legacy method is limited to pandas (multi-)indexes and
        1-dimensional "dimension" coordinates. See
        :py:meth:`~Dataset.set_xindex` for setting a pandas or a custom
        Xarray-compatible index from one or more arbitrary coordinates.

        Parameters
        ----------
        indexes : {dim: index, ...}
            Mapping from names matching dimensions and values given
            by (lists of) the names of existing coordinates or variables to set
            as new (multi-)index.
        append : bool, default: False
            If True, append the supplied index(es) to the existing index(es).
            Otherwise replace the existing index(es) (default).
        **indexes_kwargs : optional
            The keyword arguments form of ``indexes``.
            One of indexes or indexes_kwargs must be provided.

        Returns
        -------
        obj : Dataset
            Another dataset, with this dataset's data but replaced coordinates.

        Examples
        --------
        >>> arr = xr.DataArray(
        ...     data=np.ones((2, 3)),
        ...     dims=["x", "y"],
        ...     coords={"x": range(2), "y": range(3), "a": ("x", [3, 4])},
        ... )
        >>> ds = xr.Dataset({"v": arr})
        >>> ds
        <xarray.Dataset>
        Dimensions:  (x: 2, y: 3)
        Coordinates:
          * x        (x) int64 0 1
          * y        (y) int64 0 1 2
            a        (x) int64 3 4
        Data variables:
            v        (x, y) float64 1.0 1.0 1.0 1.0 1.0 1.0
        >>> ds.set_index(x="a")
        <xarray.Dataset>
        Dimensions:  (x: 2, y: 3)
        Coordinates:
          * x        (x) int64 3 4
          * y        (y) int64 0 1 2
        Data variables:
            v        (x, y) float64 1.0 1.0 1.0 1.0 1.0 1.0

        See Also
        --------
        Dataset.reset_index
        Dataset.set_xindex
        Dataset.swap_dims
        """
        dim_coords = either_dict_or_kwargs(indexes, indexes_kwargs, "set_index")

        new_indexes: dict[Hashable, Index] = {}
        new_variables: dict[Hashable, Variable] = {}
        drop_indexes: set[Hashable] = set()
        drop_variables: set[Hashable] = set()
        replace_dims: dict[Hashable, Hashable] = {}
        all_var_names: set[Hashable] = set()

        for dim, _var_names in dim_coords.items():
            if isinstance(_var_names, str) or not isinstance(_var_names, Sequence):
                var_names = [_var_names]
            else:
                var_names = list(_var_names)

            invalid_vars = set(var_names) - set(self._variables)
            if invalid_vars:
                raise ValueError(
                    ", ".join([str(v) for v in invalid_vars])
                    + " variable(s) do not exist"
                )

            all_var_names.update(var_names)
            drop_variables.update(var_names)

            # drop any pre-existing index involved and its corresponding coordinates
            index_coord_names = self.xindexes.get_all_coords(dim, errors="ignore")
            all_index_coord_names = set(index_coord_names)
            for k in var_names:
                all_index_coord_names.update(
                    self.xindexes.get_all_coords(k, errors="ignore")
                )

            drop_indexes.update(all_index_coord_names)
            drop_variables.update(all_index_coord_names)

            if len(var_names) == 1 and (not append or dim not in self._indexes):
                var_name = var_names[0]
                var = self._variables[var_name]
                if var.dims != (dim,):
                    raise ValueError(
                        f"dimension mismatch: try setting an index for dimension {dim!r} with "
                        f"variable {var_name!r} that has dimensions {var.dims}"
                    )
                idx = PandasIndex.from_variables({dim: var}, options={})
                idx_vars = idx.create_variables({var_name: var})

                # trick to preserve coordinate order in this case
                if dim in self._coord_names:
                    drop_variables.remove(dim)
            else:
                if append:
                    current_variables = {
                        k: self._variables[k] for k in index_coord_names
                    }
                else:
                    current_variables = {}
                idx, idx_vars = PandasMultiIndex.from_variables_maybe_expand(
                    dim,
                    current_variables,
                    {k: self._variables[k] for k in var_names},
                )
                for n in idx.index.names:
                    replace_dims[n] = dim

            new_indexes.update({k: idx for k in idx_vars})
            new_variables.update(idx_vars)

        # re-add deindexed coordinates (convert to base variables)
        for k in drop_variables:
            if (
                k not in new_variables
                and k not in all_var_names
                and k in self._coord_names
            ):
                new_variables[k] = self._variables[k].to_base_variable()

        indexes_: dict[Any, Index] = {
            k: v for k, v in self._indexes.items() if k not in drop_indexes
        }
        indexes_.update(new_indexes)

        variables = {
            k: v for k, v in self._variables.items() if k not in drop_variables
        }
        variables.update(new_variables)

        # update dimensions if necessary, GH: 3512
        for k, v in variables.items():
            if any(d in replace_dims for d in v.dims):
                new_dims = [replace_dims.get(d, d) for d in v.dims]
                variables[k] = v._replace(dims=new_dims)

        coord_names = self._coord_names - drop_variables | set(new_variables)

        return self._replace_with_new_dims(
            variables, coord_names=coord_names, indexes=indexes_
        )