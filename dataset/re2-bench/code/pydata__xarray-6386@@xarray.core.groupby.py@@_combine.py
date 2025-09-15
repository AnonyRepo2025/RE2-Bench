from __future__ import annotations
import datetime
import warnings
from typing import Any, Callable, Hashable, Sequence
import numpy as np
import pandas as pd
from . import dtypes, duck_array_ops, nputils, ops
from ._reductions import DataArrayGroupByReductions, DatasetGroupByReductions
from .arithmetic import DataArrayGroupbyArithmetic, DatasetGroupbyArithmetic
from .concat import concat
from .formatting import format_array_flat
from .indexes import create_default_index_implicit, filter_indexes_from_coords
from .options import _get_keep_attrs
from .pycompat import integer_types
from .utils import (
    either_dict_or_kwargs,
    hashable,
    is_scalar,
    maybe_wrap_array,
    peek_at,
    safe_cast_to_index,
)
from .variable import IndexVariable, Variable
from .dataarray import DataArray
from .dataset import Dataset
from .dataarray import DataArray
from .resample_cftime import CFTimeGrouper



class DatasetGroupByBase(GroupBy, DatasetGroupbyArithmetic):
    __slots__ = ()
    def _combine(self, applied):
        """Recombine the applied objects like the original."""
        applied_example, applied = peek_at(applied)
        coord, dim, positions = self._infer_concat_args(applied_example)
        combined = concat(applied, dim)
        combined = _maybe_reorder(combined, dim, positions)
        # assign coord when the applied function does not return that coord
        if coord is not None and dim not in applied_example.dims:
            index, index_vars = create_default_index_implicit(coord)
            indexes = {k: index for k in index_vars}
            combined = combined._overwrite_indexes(indexes, index_vars)
        combined = self._maybe_restore_empty_groups(combined)
        combined = self._maybe_unstack(combined)
        return combined