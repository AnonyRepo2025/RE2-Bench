import datetime
import warnings
from numbers import Number
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
)
import numpy as np
import pandas as pd
from ..plot.plot import _PlotMethods
from . import (
    computation,
    dtypes,
    groupby,
    indexing,
    ops,
    pdcompat,
    resample,
    rolling,
    utils,
    weighted,
)
from .accessor_dt import CombinedDatetimelikeAccessor
from .accessor_str import StringAccessor
from .alignment import (
    _broadcast_helper,
    _get_broadcast_dims_map_common_coords,
    align,
    reindex_like_indexers,
)
from .arithmetic import DataArrayArithmetic
from .common import AbstractArray, DataWithCoords
from .coordinates import (
    DataArrayCoordinates,
    assert_coordinate_consistent,
    remap_label_indexers,
)
from .dataset import Dataset, split_indexes
from .formatting import format_item
from .indexes import Indexes, default_indexes, propagate_indexes
from .indexing import is_fancy_indexer
from .merge import PANDAS_TYPES, MergeError, _extract_indexes_from_coords
from .options import OPTIONS, _get_keep_attrs
from .utils import (
    Default,
    HybridMappingProxy,
    ReprObject,
    _default,
    either_dict_or_kwargs,
)
from .variable import (
    IndexVariable,
    Variable,
    as_compatible_data,
    as_variable,
    assert_unique_multiindex_level_names,
)
from dask.delayed import Delayed
from cdms2 import Variable as cdms2_Variable
from iris.cube import Cube as iris_Cube
from .dataset import _get_virtual_variable
from dask.base import normalize_token
from .missing import interp_na
from .missing import ffill
from .missing import bfill
from ..backends.api import DATAARRAY_NAME, DATAARRAY_VARIABLE
from ..convert import to_cdms2
from ..convert import from_cdms2
from ..convert import to_iris
from ..convert import from_iris
from .parallel import map_blocks

T_DataArray = TypeVar("T_DataArray", bound="DataArray")
T_DSorDA = TypeVar("T_DSorDA", "DataArray", Dataset)
_THIS_ARRAY = ReprObject("<this-array>")

class DataArray(AbstractArray, DataWithCoords, DataArrayArithmetic):
    __slots__ = (
        "_cache",
        "_coords",
        "_close",
        "_indexes",
        "_name",
        "_variable",
        "__weakref__",
    )
    _groupby_cls = groupby.DataArrayGroupBy
    _rolling_cls = rolling.DataArrayRolling
    _coarsen_cls = rolling.DataArrayCoarsen
    _resample_cls = resample.DataArrayResample
    _weighted_cls = weighted.DataArrayWeighted
    dt = utils.UncachedAccessor(CombinedDatetimelikeAccessor)
    __hash__ = None  # type: ignore[assignment]
    plot = utils.UncachedAccessor(_PlotMethods)
    str = utils.UncachedAccessor(StringAccessor)
    def ffill(self, dim: Hashable, limit: int = None) -> "DataArray":
        """Fill NaN values by propogating values forward

        *Requires bottleneck.*

        Parameters
        ----------
        dim : hashable
            Specifies the dimension along which to propagate values when
            filling.
        limit : int, default: None
            The maximum number of consecutive NaN values to forward fill. In
            other words, if there is a gap with more than this number of
            consecutive NaNs, it will only be partially filled. Must be greater
            than 0 or None for no limit. Must be None or greater than or equal
            to axis length if filling along chunked axes (dimensions).

        Returns
        -------
        DataArray
        """
        from .missing import ffill

        return ffill(self, dim, limit=limit)