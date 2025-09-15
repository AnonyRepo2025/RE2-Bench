import datetime
import functools
import warnings
import numpy as np
import pandas as pd
from . import dtypes, duck_array_ops, nputils, ops
from .arithmetic import SupportsArithmetic
from .common import ALL_DIMS, ImplementsArrayReduce, ImplementsDatasetReduce
from .concat import concat
from .formatting import format_array_flat
from .options import _get_keep_attrs
from .pycompat import integer_types
from .utils import (
    either_dict_or_kwargs,
    hashable,
    maybe_wrap_array,
    peek_at,
    safe_cast_to_index,
)
from .variable import IndexVariable, Variable, as_variable
from .dataset import Dataset
from .dataarray import DataArray
from .dataarray import DataArray
from .resample_cftime import CFTimeGrouper



class GroupBy(SupportsArithmetic):
    __slots__ = (
        "_full_index",
        "_inserted_dims",
        "_group",
        "_group_dim",
        "_group_indices",
        "_groups",
        "_obj",
        "_restore_coord_dims",
        "_stacked_dim",
        "_unique_coord",
        "_dims",
    )
    def __repr__(self):
        return "%s, grouped over %r \n%r groups with labels %s." % (
            self.__class__.__name__,
            self._unique_coord.name,
            self._unique_coord.size,
            ", ".join(format_array_flat(self._unique_coord, 30).split()),
        )