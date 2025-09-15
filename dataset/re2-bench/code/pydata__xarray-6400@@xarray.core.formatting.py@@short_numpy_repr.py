import contextlib
import functools
from collections import defaultdict
from datetime import datetime, timedelta
from itertools import chain, zip_longest
from typing import Collection, Hashable, Optional
import numpy as np
import pandas as pd
from pandas.errors import OutOfBoundsDatetime
from .duck_array_ops import array_equiv
from .indexing import MemoryCachedArray
from .options import OPTIONS, _get_boolean_with_default
from .pycompat import dask_array_type, sparse_array_type
from .utils import is_duck_array
import sparse
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

def short_numpy_repr(array):
    array = np.asarray(array)

    # default to lower precision so a full (abbreviated) line can fit on
    # one line with the default display_width
    options = {
        "precision": 6,
        "linewidth": OPTIONS["display_width"],
        "threshold": OPTIONS["display_values_threshold"],
    }
    if array.ndim < 3:
        edgeitems = 3
    elif array.ndim == 3:
        edgeitems = 2
    else:
        edgeitems = 1
    options["edgeitems"] = edgeitems
    with set_numpy_options(**options):
        return repr(array)
