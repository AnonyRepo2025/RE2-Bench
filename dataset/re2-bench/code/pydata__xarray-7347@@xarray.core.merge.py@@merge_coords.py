from __future__ import annotations
from collections import defaultdict
from typing import (
    TYPE_CHECKING,
    AbstractSet,
    Any,
    Hashable,
    Iterable,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)
import pandas as pd
from . import dtypes
from .alignment import deep_align
from .duck_array_ops import lazy_array_equiv
from .indexes import (
    Index,
    Indexes,
    create_default_index_implicit,
    filter_indexes_from_coords,
    indexes_equal,
)
from .utils import Frozen, compat_dict_union, dict_equiv, equivalent
from .variable import Variable, as_variable, calculate_dimensions
from .coordinates import Coordinates
from .dataarray import DataArray
from .dataset import Dataset
from .types import CombineAttrsOptions, CompatOptions, JoinOptions
from .dataarray import DataArray
from .dataset import Dataset
from .dataarray import DataArray
from .dataset import Dataset
from .dataarray import DataArray
from .dataset import Dataset
from .dataarray import DataArray
from .dataset import Dataset
from .dataarray import DataArray
from .dataset import Dataset
from .dataarray import DataArray
from .dataset import Dataset

PANDAS_TYPES = (pd.Series, pd.DataFrame)
_VALID_COMPAT = Frozen(
    {
        "identical": 0,
        "equals": 1,
        "broadcast_equals": 2,
        "minimal": 3,
        "no_conflicts": 4,
        "override": 5,
    }
)
MergeElement = Tuple[Variable, Optional[Index]]

def merge_coords(
    objects: Iterable[CoercibleMapping],
    compat: CompatOptions = "minimal",
    join: JoinOptions = "outer",
    priority_arg: int | None = None,
    indexes: Mapping[Any, Index] | None = None,
    fill_value: object = dtypes.NA,
) -> tuple[dict[Hashable, Variable], dict[Hashable, Index]]:
    """Merge coordinate variables.

    See merge_core below for argument descriptions. This works similarly to
    merge_core, except everything we don't worry about whether variables are
    coordinates or not.
    """
    _assert_compat_valid(compat)
    coerced = coerce_pandas_values(objects)
    aligned = deep_align(
        coerced, join=join, copy=False, indexes=indexes, fill_value=fill_value
    )
    collected = collect_variables_and_indexes(aligned, indexes=indexes)
    prioritized = _get_priority_vars_and_indexes(aligned, priority_arg, compat=compat)
    variables, out_indexes = merge_collected(collected, prioritized, compat=compat)
    return variables, out_indexes
