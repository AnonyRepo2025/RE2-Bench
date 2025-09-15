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

class IndexVariable(Variable):
    __slots__ = ()
    to_coord = utils.alias(to_index_variable, "to_coord")
    def _to_index(self) -> pd.Index:
        # n.b. creating a new pandas.Index from an old pandas.Index is
        # basically free as pandas.Index objects are immutable.
        # n.b.2. this method returns the multi-index instance for
        # a pandas multi-index level variable.
        assert self.ndim == 1
        index = self._data.array
        if isinstance(index, pd.MultiIndex):
            # set default names for multi-index unnamed levels so that
            # we can safely rename dimension / coordinate later
            valid_level_names = [
                name or f"{self.dims[0]}_level_{i}"
                for i, name in enumerate(index.names)
            ]
            index = index.set_names(valid_level_names)
        else:
            index = index.set_names(self.name)
        return index
    def to_index(self) -> pd.Index:
        """Convert this variable to a pandas.Index"""
        index = self._to_index()
        level = getattr(self._data, "level", None)
        if level is not None:
            # return multi-index level converted to a single index
            return index.get_level_values(level)
        else:
            return index
    @name.setter
    def name(self, value) -> NoReturn:
        raise AttributeError("cannot modify name of IndexVariable in-place")