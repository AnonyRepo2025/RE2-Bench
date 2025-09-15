from __future__ import annotations
import datetime
import warnings
from os import PathLike
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Hashable,
    Iterable,
    Literal,
    Mapping,
    NoReturn,
    Sequence,
    cast,
    overload,
)
import numpy as np
import pandas as pd
from ..coding.calendar_ops import convert_calendar, interp_calendar
from ..coding.cftimeindex import CFTimeIndex
from ..plot.plot import _PlotMethods
from ..plot.utils import _get_units_from_attrs
from . import alignment, computation, dtypes, indexing, ops, utils
from ._reductions import DataArrayReductions
from .accessor_dt import CombinedDatetimelikeAccessor
from .accessor_str import StringAccessor
from .alignment import _broadcast_helper, _get_broadcast_dims_map_common_coords, align
from .arithmetic import DataArrayArithmetic
from .common import AbstractArray, DataWithCoords, get_chunksizes
from .computation import unify_chunks
from .coordinates import DataArrayCoordinates, assert_coordinate_consistent
from .dataset import Dataset
from .formatting import format_item
from .indexes import (
    Index,
    Indexes,
    PandasMultiIndex,
    filter_indexes_from_coords,
    isel_indexes,
)
from .indexing import is_fancy_indexer, map_index_queries
from .merge import PANDAS_TYPES, MergeError, _create_indexes_from_coords
from .npcompat import QUANTILE_METHODS, ArrayLike
from .options import OPTIONS, _get_keep_attrs
from .utils import (
    Default,
    HybridMappingProxy,
    ReprObject,
    _default,
    either_dict_or_kwargs,
)
from .variable import IndexVariable, Variable, as_compatible_data, as_variable
from typing import TypeVar, Union
from ..backends.api import T_NetcdfEngine, T_NetcdfTypes
from .groupby import DataArrayGroupBy
from .resample import DataArrayResample
from .rolling import DataArrayCoarsen, DataArrayRolling
from .types import (
        CoarsenBoundaryOptions,
        DatetimeUnitOptions,
        Ellipsis,
        ErrorOptions,
        ErrorOptionsWithWarn,
        InterpOptions,
        PadModeOptions,
        PadReflectOptions,
        QueryEngineOptions,
        QueryParserOptions,
        ReindexMethodOptions,
        SideOptions,
        T_DataArray,
        T_Xarray,
    )
from .weighted import DataArrayWeighted
from dask.delayed import Delayed
from cdms2 import Variable as cdms2_Variable
from iris.cube import Cube as iris_Cube
from .dataset import _get_virtual_variable
from dask.base import normalize_token
from .missing import interp_na
from .missing import ffill
from .missing import bfill
from ..backends.api import DATAARRAY_NAME, DATAARRAY_VARIABLE, to_netcdf
from ..convert import to_cdms2
from ..convert import from_cdms2
from ..convert import to_iris
from ..convert import from_iris
from .groupby import GroupBy
from .groupby import GroupBy
from .parallel import map_blocks
from .groupby import DataArrayGroupBy
from .groupby import DataArrayGroupBy
from .weighted import DataArrayWeighted
from .rolling import DataArrayRolling
from .rolling import DataArrayCoarsen
from .resample import DataArrayResample

_THIS_ARRAY = ReprObject("<this-array>")

class DataArray(AbstractArray, DataWithCoords, DataArrayArithmetic, DataArrayReductions):
    __slots__ = (
        "_cache",
        "_coords",
        "_close",
        "_indexes",
        "_name",
        "_variable",
        "__weakref__",
    )
    dt = utils.UncachedAccessor(CombinedDatetimelikeAccessor["DataArray"])
    __hash__ = None  # type: ignore[assignment]
    plot = utils.UncachedAccessor(_PlotMethods)
    str = utils.UncachedAccessor(StringAccessor["DataArray"])
    def _to_temp_dataset(self) -> Dataset:
        return self._to_dataset_whole(name=_THIS_ARRAY, shallow_copy=False)
    def _to_dataset_whole(
        self, name: Hashable = None, shallow_copy: bool = True
    ) -> Dataset:
        if name is None:
            name = self.name
        if name is None:
            raise ValueError(
                "unable to convert unnamed DataArray to a "
                "Dataset without providing an explicit name"
            )
        if name in self.coords:
            raise ValueError(
                "cannot create a Dataset from a DataArray with "
                "the same name as one of its coordinates"
            )
        # use private APIs for speed: this is called by _to_temp_dataset(),
        # which is used in the guts of a lot of operations (e.g., reindex)
        variables = self._coords.copy()
        variables[name] = self.variable
        if shallow_copy:
            for k in variables:
                variables[k] = variables[k].copy(deep=False)
        indexes = self._indexes

        coord_names = set(self._coords)
        return Dataset._construct_direct(variables, coord_names, indexes=indexes)
    @property
    def variable(self) -> Variable:
        """Low level interface to the Variable object for this DataArray."""
        return self._variable
    @property
    def coords(self) -> DataArrayCoordinates:
        """Dictionary-like container of coordinate arrays."""
        return DataArrayCoordinates(self)
    def set_index(
        self,
        indexes: Mapping[Any, Hashable | Sequence[Hashable]] = None,
        append: bool = False,
        **indexes_kwargs: Hashable | Sequence[Hashable],
    ) -> DataArray:
        """Set DataArray (multi-)indexes using one or more existing
        coordinates.

        This legacy method is limited to pandas (multi-)indexes and
        1-dimensional "dimension" coordinates. See
        :py:meth:`~DataArray.set_xindex` for setting a pandas or a custom
        Xarray-compatible index from one or more arbitrary coordinates.

        Parameters
        ----------
        indexes : {dim: index, ...}
            Mapping from names matching dimensions and values given
            by (lists of) the names of existing coordinates or variables to set
            as new (multi-)index.
        append : bool, default: False
            If True, append the supplied index(es) to the existing index(es).
            Otherwise replace the existing index(es).
        **indexes_kwargs : optional
            The keyword arguments form of ``indexes``.
            One of indexes or indexes_kwargs must be provided.

        Returns
        -------
        obj : DataArray
            Another DataArray, with this data but replaced coordinates.

        Examples
        --------
        >>> arr = xr.DataArray(
        ...     data=np.ones((2, 3)),
        ...     dims=["x", "y"],
        ...     coords={"x": range(2), "y": range(3), "a": ("x", [3, 4])},
        ... )
        >>> arr
        <xarray.DataArray (x: 2, y: 3)>
        array([[1., 1., 1.],
               [1., 1., 1.]])
        Coordinates:
          * x        (x) int64 0 1
          * y        (y) int64 0 1 2
            a        (x) int64 3 4
        >>> arr.set_index(x="a")
        <xarray.DataArray (x: 2, y: 3)>
        array([[1., 1., 1.],
               [1., 1., 1.]])
        Coordinates:
          * x        (x) int64 3 4
          * y        (y) int64 0 1 2

        See Also
        --------
        DataArray.reset_index
        DataArray.set_xindex
        """
        ds = self._to_temp_dataset().set_index(indexes, append=append, **indexes_kwargs)
        return self._from_temp_dataset(ds)