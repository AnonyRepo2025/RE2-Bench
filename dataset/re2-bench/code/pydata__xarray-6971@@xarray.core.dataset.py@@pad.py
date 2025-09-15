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
    def variables(self) -> Frozen[Hashable, Variable]:
        """Low level interface to Dataset contents as dict of Variable objects.

        This ordered dictionary is frozen to prevent mutation that could
        violate Dataset invariants. It contains all variable objects
        constituting the Dataset, including both data variables and
        coordinates.
        """
        return Frozen(self._variables)
    @property
    def xindexes(self) -> Indexes[Index]:
        """Mapping of xarray Index objects used for label based indexing."""
        return Indexes(self._indexes, {k: self._variables[k] for k in self._indexes})
    @property
    def data_vars(self) -> DataVariables:
        """Dictionary of DataArray objects corresponding to data variables"""
        return DataVariables(self)
    def pad(
        self: T_Dataset,
        pad_width: Mapping[Any, int | tuple[int, int]] = None,
        mode: PadModeOptions = "constant",
        stat_length: int
        | tuple[int, int]
        | Mapping[Any, tuple[int, int]]
        | None = None,
        constant_values: (
            float | tuple[float, float] | Mapping[Any, tuple[float, float]] | None
        ) = None,
        end_values: int | tuple[int, int] | Mapping[Any, tuple[int, int]] | None = None,
        reflect_type: PadReflectOptions = None,
        **pad_width_kwargs: Any,
    ) -> T_Dataset:
        """Pad this dataset along one or more dimensions.

        .. warning::
            This function is experimental and its behaviour is likely to change
            especially regarding padding of dimension coordinates (or IndexVariables).

        When using one of the modes ("edge", "reflect", "symmetric", "wrap"),
        coordinates will be padded with the same mode, otherwise coordinates
        are padded using the "constant" mode with fill_value dtypes.NA.

        Parameters
        ----------
        pad_width : mapping of hashable to tuple of int
            Mapping with the form of {dim: (pad_before, pad_after)}
            describing the number of values padded along each dimension.
            {dim: pad} is a shortcut for pad_before = pad_after = pad
        mode : {"constant", "edge", "linear_ramp", "maximum", "mean", "median", \
            "minimum", "reflect", "symmetric", "wrap"}, default: "constant"
            How to pad the DataArray (taken from numpy docs):

            - "constant": Pads with a constant value.
            - "edge": Pads with the edge values of array.
            - "linear_ramp": Pads with the linear ramp between end_value and the
              array edge value.
            - "maximum": Pads with the maximum value of all or part of the
              vector along each axis.
            - "mean": Pads with the mean value of all or part of the
              vector along each axis.
            - "median": Pads with the median value of all or part of the
              vector along each axis.
            - "minimum": Pads with the minimum value of all or part of the
              vector along each axis.
            - "reflect": Pads with the reflection of the vector mirrored on
              the first and last values of the vector along each axis.
            - "symmetric": Pads with the reflection of the vector mirrored
              along the edge of the array.
            - "wrap": Pads with the wrap of the vector along the axis.
              The first values are used to pad the end and the
              end values are used to pad the beginning.

        stat_length : int, tuple or mapping of hashable to tuple, default: None
            Used in 'maximum', 'mean', 'median', and 'minimum'.  Number of
            values at edge of each axis used to calculate the statistic value.
            {dim_1: (before_1, after_1), ... dim_N: (before_N, after_N)} unique
            statistic lengths along each dimension.
            ((before, after),) yields same before and after statistic lengths
            for each dimension.
            (stat_length,) or int is a shortcut for before = after = statistic
            length for all axes.
            Default is ``None``, to use the entire axis.
        constant_values : scalar, tuple or mapping of hashable to tuple, default: 0
            Used in 'constant'.  The values to set the padded values for each
            axis.
            ``{dim_1: (before_1, after_1), ... dim_N: (before_N, after_N)}`` unique
            pad constants along each dimension.
            ``((before, after),)`` yields same before and after constants for each
            dimension.
            ``(constant,)`` or ``constant`` is a shortcut for ``before = after = constant`` for
            all dimensions.
            Default is 0.
        end_values : scalar, tuple or mapping of hashable to tuple, default: 0
            Used in 'linear_ramp'.  The values used for the ending value of the
            linear_ramp and that will form the edge of the padded array.
            ``{dim_1: (before_1, after_1), ... dim_N: (before_N, after_N)}`` unique
            end values along each dimension.
            ``((before, after),)`` yields same before and after end values for each
            axis.
            ``(constant,)`` or ``constant`` is a shortcut for ``before = after = constant`` for
            all axes.
            Default is 0.
        reflect_type : {"even", "odd", None}, optional
            Used in "reflect", and "symmetric".  The "even" style is the
            default with an unaltered reflection around the edge value.  For
            the "odd" style, the extended part of the array is created by
            subtracting the reflected values from two times the edge value.
        **pad_width_kwargs
            The keyword arguments form of ``pad_width``.
            One of ``pad_width`` or ``pad_width_kwargs`` must be provided.

        Returns
        -------
        padded : Dataset
            Dataset with the padded coordinates and data.

        See Also
        --------
        Dataset.shift, Dataset.roll, Dataset.bfill, Dataset.ffill, numpy.pad, dask.array.pad

        Notes
        -----
        By default when ``mode="constant"`` and ``constant_values=None``, integer types will be
        promoted to ``float`` and padded with ``np.nan``. To avoid type promotion
        specify ``constant_values=np.nan``

        Padding coordinates will drop their corresponding index (if any) and will reset default
        indexes for dimension coordinates.

        Examples
        --------
        >>> ds = xr.Dataset({"foo": ("x", range(5))})
        >>> ds.pad(x=(1, 2))
        <xarray.Dataset>
        Dimensions:  (x: 8)
        Dimensions without coordinates: x
        Data variables:
            foo      (x) float64 nan 0.0 1.0 2.0 3.0 4.0 nan nan
        """
        pad_width = either_dict_or_kwargs(pad_width, pad_width_kwargs, "pad")

        if mode in ("edge", "reflect", "symmetric", "wrap"):
            coord_pad_mode = mode
            coord_pad_options = {
                "stat_length": stat_length,
                "constant_values": constant_values,
                "end_values": end_values,
                "reflect_type": reflect_type,
            }
        else:
            coord_pad_mode = "constant"
            coord_pad_options = {}

        variables = {}

        # keep indexes that won't be affected by pad and drop all other indexes
        xindexes = self.xindexes
        pad_dims = set(pad_width)
        indexes = {}
        for k, idx in xindexes.items():
            if not pad_dims.intersection(xindexes.get_all_dims(k)):
                indexes[k] = idx

        for name, var in self.variables.items():
            var_pad_width = {k: v for k, v in pad_width.items() if k in var.dims}
            if not var_pad_width:
                variables[name] = var
            elif name in self.data_vars:
                variables[name] = var.pad(
                    pad_width=var_pad_width,
                    mode=mode,
                    stat_length=stat_length,
                    constant_values=constant_values,
                    end_values=end_values,
                    reflect_type=reflect_type,
                )
            else:
                variables[name] = var.pad(
                    pad_width=var_pad_width,
                    mode=coord_pad_mode,
                    **coord_pad_options,  # type: ignore[arg-type]
                )
                # reset default index of dimension coordinates
                if (name,) == var.dims:
                    dim_var = {name: variables[name]}
                    index = PandasIndex.from_variables(dim_var, options={})
                    index_vars = index.create_variables(dim_var)
                    indexes[name] = index
                    variables[name] = index_vars[name]

        return self._replace_with_new_dims(variables, indexes=indexes)