from __future__ import annotations
import copy
import itertools
import math
import numbers
import warnings
from datetime import timedelta
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
)
import numpy as np
import pandas as pd
from packaging.version import Version
import xarray as xr  # only for Dataset and DataArray
from . import common, dtypes, duck_array_ops, indexing, nputils, ops, utils
from .arithmetic import VariableArithmetic
from .common import AbstractArray
from .indexing import (
    BasicIndexer,
    OuterIndexer,
    PandasIndexingAdapter,
    VectorizedIndexer,
    as_indexable,
)
from .npcompat import QUANTILE_METHODS, ArrayLike
from .options import OPTIONS, _get_keep_attrs
from .pycompat import (
    DuckArrayModule,
    cupy_array_type,
    integer_types,
    is_duck_dask_array,
    sparse_array_type,
)
from .utils import (
    Frozen,
    NdimSizeLenMixin,
    OrderedSet,
    _default,
    decode_numpy_dict_values,
    drop_dims_from_indexers,
    either_dict_or_kwargs,
    ensure_us_time_resolution,
    infix_dims,
    is_duck_array,
    maybe_coerce_to_str,
)
from .types import (
        Dims,
        ErrorOptionsWithWarn,
        PadModeOptions,
        PadReflectOptions,
        T_Variable,
    )
from .dataarray import DataArray
from .dataarray import DataArray
from .computation import apply_ufunc
from dask.base import normalize_token
import dask.array as da
import sparse
from .computation import apply_ufunc
from .merge import merge_attrs
from .computation import apply_ufunc
import bottleneck as bn
from .computation import apply_ufunc
from .computation import apply_ufunc
from dask.base import normalize_token
from .merge import merge_attrs
from sparse import COO

NON_NUMPY_SUPPORTED_ARRAY_TYPES = (
    indexing.ExplicitlyIndexed,
    pd.Index,
)
BASIC_INDEXING_TYPES = integer_types + (slice,)
Coordinate = utils.alias(IndexVariable, "Coordinate")

class Variable(AbstractArray, NdimSizeLenMixin, VariableArithmetic):
    __slots__ = ("_dims", "_data", "_attrs", "_encoding")
    to_variable = utils.alias(to_base_variable, "to_variable")
    to_coord = utils.alias(to_index_variable, "to_coord")
    __hash__ = None  # type: ignore[assignment]
    _array_counter = itertools.count()
    @property
    def shape(self):
        """
        Tuple of array dimensions.

        See Also
        --------
        numpy.ndarray.shape
        """
        return self._data.shape
    def _to_index(self) -> pd.Index:
        return self.to_index_variable()._to_index()
    @dims.setter
    def dims(self, value: str | Iterable[Hashable]) -> None:
        self._dims = self._parse_dimensions(value)