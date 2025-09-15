import copy
import datetime
import functools
import sys
import warnings
from collections import defaultdict
from distutils.version import LooseVersion
from html import escape
from numbers import Number
from operator import methodcaller
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    DefaultDict,
    Dict,
    Hashable,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
    cast,
    overload,
)
import numpy as np
import pandas as pd
import xarray as xr
from ..coding.cftimeindex import _parse_array_of_cftime_strings
from ..plot.dataset_plot import _Dataset_PlotMethods
from . import (
    alignment,
    dtypes,
    duck_array_ops,
    formatting,
    formatting_html,
    groupby,
    ops,
    resample,
    rolling,
    utils,
    weighted,
)
from .alignment import _broadcast_helper, _get_broadcast_dims_map_common_coords, align
from .common import (
    DataWithCoords,
    ImplementsDatasetReduce,
    _contains_datetime_like_objects,
)
from .coordinates import (
    DatasetCoordinates,
    assert_coordinate_consistent,
    remap_label_indexers,
)
from .duck_array_ops import datetime_to_numeric
from .indexes import (
    Indexes,
    default_indexes,
    isel_variable_and_index,
    propagate_indexes,
    remove_unused_levels_categories,
    roll_index,
)
from .indexing import is_fancy_indexer
from .merge import (
    dataset_merge_method,
    dataset_update_method,
    merge_coordinates_without_align,
    merge_data_and_coords,
)
from .missing import get_clean_interp_index
from .options import OPTIONS, _get_keep_attrs
from .pycompat import is_duck_dask_array, sparse_array_type
from .utils import (
    Default,
    Frozen,
    HybridMappingProxy,
    SortedKeysDict,
    _default,
    decode_numpy_dict_values,
    drop_dims_from_indexers,
    either_dict_or_kwargs,
    hashable,
    infix_dims,
    is_dict_like,
    is_scalar,
    maybe_wrap_array,
)
from .variable import (
    IndexVariable,
    Variable,
    as_variable,
    assert_unique_multiindex_level_names,
    broadcast_variables,
)
from ..backends import AbstractDataStore, ZarrStore
from .dataarray import DataArray
from .merge import CoercibleMapping
import dask.array as da
from dask.base import tokenize
from dask.delayed import Delayed
from dask.base import normalize_token
import dask
import dask
import dask.array as da
import dask.array as da
import dask
import dask
from dask.optimization import cull
from .dataarray import DataArray
from ..backends.api import dump_to_store
from ..backends.api import to_netcdf
from ..backends.api import to_zarr
from .dataarray import DataArray
from .dataarray import DataArray
from . import missing
from .missing import _apply_over_vars_with_dim, interp_na
from .missing import _apply_over_vars_with_dim, ffill
from .missing import _apply_over_vars_with_dim, bfill
from .dataarray import DataArray
from sparse import COO
import dask.array as da
import dask.dataframe as dd
from .dataarray import DataArray
from .variable import Variable
from .variable import Variable
import dask.array
from .parallel import map_blocks
import dask.array as da
import dask
import itertools
from .dataarray import DataArray
from .dataarray import DataArray
from dask.highlevelgraph import HighLevelGraph
from dask import sharedict

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

class Dataset(Mapping, ImplementsDatasetReduce, DataWithCoords):
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
    _groupby_cls = groupby.DatasetGroupBy
    _rolling_cls = rolling.DatasetRolling
    _coarsen_cls = rolling.DatasetCoarsen
    _resample_cls = resample.DatasetResample
    _weighted_cls = weighted.DatasetWeighted
    __hash__ = None  # type: ignore
    plot = utils.UncachedAccessor(_Dataset_PlotMethods)
    @property
    def variables(self) -> Mapping[Hashable, Variable]:
        """Low level interface to Dataset contents as dict of Variable objects.

        This ordered dictionary is frozen to prevent mutation that could
        violate Dataset invariants. It contains all variable objects
        constituting the Dataset, including both data variables and
        coordinates.
        """
        return Frozen(self._variables)
    @attrs.setter
    def attrs(self, value: Mapping[Hashable, Any]) -> None:
        self._attrs = dict(value)
    @property
    def dims(self) -> Mapping[Hashable, int]:
        """Mapping from dimension names to lengths.

        Cannot be modified directly, but is updated when adding new variables.

        Note that type of this object differs from `DataArray.dims`.
        See `Dataset.sizes` and `DataArray.sizes` for consistently named
        properties.
        """
        return Frozen(SortedKeysDict(self._dims))
    @property
    def indexes(self) -> Indexes:
        """Mapping of pandas.Index objects used for label based indexing"""
        if self._indexes is None:
            self._indexes = default_indexes(self._variables, self._dims)
        return Indexes(self._indexes)
    @property
    def coords(self) -> DatasetCoordinates:
        """Dictionary of xarray.DataArray objects corresponding to coordinate
        variables
        """
        return DatasetCoordinates(self)
    def reduce(
        self,
        func: Callable,
        dim: Union[Hashable, Iterable[Hashable]] = None,
        keep_attrs: bool = None,
        keepdims: bool = False,
        numeric_only: bool = False,
        **kwargs: Any,
    ) -> "Dataset":
        """Reduce this dataset by applying `func` along some dimension(s).

        Parameters
        ----------
        func : callable
            Function which can be called in the form
            `f(x, axis=axis, **kwargs)` to return the result of reducing an
            np.ndarray over an integer valued axis.
        dim : str or sequence of str, optional
            Dimension(s) over which to apply `func`.  By default `func` is
            applied over all dimensions.
        keep_attrs : bool, optional
            If True, the dataset's attributes (`attrs`) will be copied from
            the original object to the new one.  If False (default), the new
            object will be returned without attributes.
        keepdims : bool, default: False
            If True, the dimensions which are reduced are left in the result
            as dimensions of size one. Coordinates that use these dimensions
            are removed.
        numeric_only : bool, optional
            If True, only apply ``func`` to variables with a numeric dtype.
        **kwargs : Any
            Additional keyword arguments passed on to ``func``.

        Returns
        -------
        reduced : Dataset
            Dataset with this object's DataArrays replaced with new DataArrays
            of summarized data and the indicated dimension(s) removed.
        """
        if "axis" in kwargs:
            raise ValueError(
                "passing 'axis' to Dataset reduce methods is ambiguous."
                " Please use 'dim' instead."
            )

        if dim is None or dim is ...:
            dims = set(self.dims)
        elif isinstance(dim, str) or not isinstance(dim, Iterable):
            dims = {dim}
        else:
            dims = set(dim)

        missing_dimensions = [d for d in dims if d not in self.dims]
        if missing_dimensions:
            raise ValueError(
                "Dataset does not contain the dimensions: %s" % missing_dimensions
            )

        if keep_attrs is None:
            keep_attrs = _get_keep_attrs(default=False)

        variables: Dict[Hashable, Variable] = {}
        for name, var in self._variables.items():
            reduce_dims = [d for d in var.dims if d in dims]
            if name in self.coords:
                if not reduce_dims:
                    variables[name] = var
            else:
                if (
                    not numeric_only
                    or np.issubdtype(var.dtype, np.number)
                    or (var.dtype == np.bool_)
                ):
                    if len(reduce_dims) == 1:
                        # unpack dimensions for the benefit of functions
                        # like np.argmin which can't handle tuple arguments
                        (reduce_dims,) = reduce_dims
                    elif len(reduce_dims) == var.ndim:
                        # prefer to aggregate over axis=None rather than
                        # axis=(0, 1) if they will be equivalent, because
                        # the former is often more efficient
                        reduce_dims = None  # type: ignore
                    variables[name] = var.reduce(
                        func,
                        dim=reduce_dims,
                        keep_attrs=keep_attrs,
                        keepdims=keepdims,
                        **kwargs,
                    )

        coord_names = {k for k in self.coords if k in variables}
        indexes = {k: v for k, v in self.indexes.items() if k in variables}
        attrs = self.attrs if keep_attrs else None
        return self._replace_with_new_dims(
            variables, coord_names=coord_names, attrs=attrs, indexes=indexes
        )