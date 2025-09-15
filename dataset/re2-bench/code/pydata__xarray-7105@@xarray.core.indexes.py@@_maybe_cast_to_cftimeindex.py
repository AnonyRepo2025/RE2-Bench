from __future__ import annotations
import collections.abc
import copy
from collections import defaultdict
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generic,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
    TypeVar,
    cast,
)
import numpy as np
import pandas as pd
from . import formatting, nputils, utils
from .indexing import IndexSelResult, PandasIndexingAdapter, PandasMultiIndexingAdapter
from .utils import Frozen, get_valid_numpy_dtype, is_dict_like, is_scalar
from .types import ErrorOptions, T_Index
from .variable import Variable
from ..coding.cftimeindex import CFTimeIndex
from .dataarray import DataArray
from .variable import Variable
from .dataarray import DataArray
from .variable import Variable
from .variable import IndexVariable
from .variable import Variable
from .dataarray import DataArray
from .variable import Variable
from .variable import IndexVariable
from .dataarray import DataArray
from .variable import Variable
from .variable import calculate_dimensions
from .variable import calculate_dimensions

IndexVars = Dict[Any, "Variable"]
T_PandasOrXarrayIndex = TypeVar("T_PandasOrXarrayIndex", Index, pd.Index)

def _maybe_cast_to_cftimeindex(index: pd.Index) -> pd.Index:
    from ..coding.cftimeindex import CFTimeIndex

    if len(index) > 0 and index.dtype == "O":
        try:
            return CFTimeIndex(index)
        except (ImportError, TypeError):
            return index
    else:
        return index
