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
        Dims,
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
    def copy(self: T_DataArray, deep: bool = True, data: Any = None) -> T_DataArray:
        """Returns a copy of this array.

        If `deep=True`, a deep copy is made of the data array.
        Otherwise, a shallow copy is made, and the returned data array's
        values are a new view of this data array's values.

        Use `data` to create a new object with the same structure as
        original but entirely new data.

        Parameters
        ----------
        deep : bool, optional
            Whether the data array and its coordinates are loaded into memory
            and copied onto the new object. Default is True.
        data : array_like, optional
            Data to use in the new object. Must have same shape as original.
            When `data` is used, `deep` is ignored for all data variables,
            and only used for coords.

        Returns
        -------
        copy : DataArray
            New object with dimensions, attributes, coordinates, name,
            encoding, and optionally data copied from original.

        Examples
        --------
        Shallow versus deep copy

        >>> array = xr.DataArray([1, 2, 3], dims="x", coords={"x": ["a", "b", "c"]})
        >>> array.copy()
        <xarray.DataArray (x: 3)>
        array([1, 2, 3])
        Coordinates:
          * x        (x) <U1 'a' 'b' 'c'
        >>> array_0 = array.copy(deep=False)
        >>> array_0[0] = 7
        >>> array_0
        <xarray.DataArray (x: 3)>
        array([7, 2, 3])
        Coordinates:
          * x        (x) <U1 'a' 'b' 'c'
        >>> array
        <xarray.DataArray (x: 3)>
        array([7, 2, 3])
        Coordinates:
          * x        (x) <U1 'a' 'b' 'c'

        Changing the data using the ``data`` argument maintains the
        structure of the original object, but with the new data. Original
        object is unaffected.

        >>> array.copy(data=[0.1, 0.2, 0.3])
        <xarray.DataArray (x: 3)>
        array([0.1, 0.2, 0.3])
        Coordinates:
          * x        (x) <U1 'a' 'b' 'c'
        >>> array
        <xarray.DataArray (x: 3)>
        array([7, 2, 3])
        Coordinates:
          * x        (x) <U1 'a' 'b' 'c'

        See Also
        --------
        pandas.DataFrame.copy
        """
        return self._copy(deep=deep, data=data)