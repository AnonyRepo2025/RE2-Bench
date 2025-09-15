from __future__ import annotations
import contextlib
import functools
import math
from collections import defaultdict
from datetime import datetime, timedelta
from itertools import chain, zip_longest
from reprlib import recursive_repr
from typing import Collection, Hashable
import numpy as np
import pandas as pd
from pandas.errors import OutOfBoundsDatetime
from .duck_array_ops import array_equiv
from .indexing import MemoryCachedArray
from .options import OPTIONS, _get_boolean_with_default
from .pycompat import dask_array_type, sparse_array_type
from .utils import is_duck_array
import sparse
from .indexes import PandasIndex, PandasMultiIndex
from .variable import Variable

_KNOWN_TYPE_REPRS = {np.ndarray: "np.ndarray"}
EMPTY_REPR = "    *empty*"
data_vars_repr = functools.partial(
    _mapping_repr,
    title="Data variables",
    summarizer=summarize_variable,
    expand_option_name="display_expand_data_vars",
)
attrs_repr = functools.partial(
    _mapping_repr,
    title="Attributes",
    summarizer=summarize_attr,
    expand_option_name="display_expand_attrs",
)
diff_data_vars_repr = functools.partial(
    _diff_mapping_repr, title="Data variables", summarizer=summarize_variable
)
diff_attrs_repr = functools.partial(
    _diff_mapping_repr, title="Attributes", summarizer=summarize_attr
)

def short_data_repr(array):
    """Format "data" for DataArray and Variable."""
    internal_data = getattr(array, "variable", array)._data
    if isinstance(array, np.ndarray):
        return short_numpy_repr(array)
    elif is_duck_array(internal_data):
        return limit_lines(repr(array.data), limit=40)
    elif array._in_memory:
        return short_numpy_repr(array)
    else:
        # internal xarray array type
        return f"[{array.size} values with dtype={array.dtype}]"
