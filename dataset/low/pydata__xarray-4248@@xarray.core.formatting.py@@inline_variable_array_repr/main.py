import contextlib
import functools
from datetime import datetime, timedelta
from itertools import chain, zip_longest
from typing import Hashable
import numpy as np
import pandas as pd
from pandas.errors import OutOfBoundsDatetime
from .duck_array_ops import array_equiv
from .options import OPTIONS
from .pycompat import dask_array_type, sparse_array_type
import sparse
from .variable import IndexVariable

_KNOWN_TYPE_REPRS = {np.ndarray: "np.ndarray"}
EMPTY_REPR = "    *empty*"
data_vars_repr = functools.partial(
    _mapping_repr, title="Data variables", summarizer=summarize_datavar
)
attrs_repr = functools.partial(
    _mapping_repr, title="Attributes", summarizer=summarize_attr
)
diff_coords_repr = functools.partial(
    _diff_mapping_repr, title="Coordinates", summarizer=summarize_coord
)
diff_data_vars_repr = functools.partial(
    _diff_mapping_repr, title="Data variables", summarizer=summarize_datavar
)
diff_attrs_repr = functools.partial(
    _diff_mapping_repr, title="Attributes", summarizer=summarize_attr
)

def inline_variable_array_repr(var, max_width):
    """Build a one-line summary of a variable's data."""
    if var._in_memory:
        return format_array_flat(var, max_width)
    elif isinstance(var._data, dask_array_type):
        return inline_dask_repr(var.data)
    elif isinstance(var._data, sparse_array_type):
        return inline_sparse_repr(var.data)
    elif hasattr(var._data, "_repr_inline_"):
        return var._data._repr_inline_(max_width)
    elif hasattr(var._data, "__array_function__"):
        return maybe_truncate(repr(var._data).replace("\n", " "), max_width)
    else:
        # internal xarray array type
        return "..."
