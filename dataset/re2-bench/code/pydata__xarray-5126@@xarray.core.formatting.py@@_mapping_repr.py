import contextlib
import functools
from datetime import datetime, timedelta
from itertools import chain, zip_longest
from typing import Hashable
import numpy as np
import pandas as pd
from pandas.errors import OutOfBoundsDatetime
from .duck_array_ops import array_equiv
from .options import OPTIONS, _get_boolean_with_default
from .pycompat import dask_array_type, sparse_array_type
from .utils import is_duck_array
import sparse
from .variable import IndexVariable

_KNOWN_TYPE_REPRS = {np.ndarray: "np.ndarray"}
EMPTY_REPR = "    *empty*"
data_vars_repr = functools.partial(
    _mapping_repr,
    title="Data variables",
    summarizer=summarize_datavar,
    expand_option_name="display_expand_data_vars",
)
attrs_repr = functools.partial(
    _mapping_repr,
    title="Attributes",
    summarizer=summarize_attr,
    expand_option_name="display_expand_attrs",
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

def _mapping_repr(
    mapping, title, summarizer, expand_option_name, col_width=None, max_rows=None
):
    if col_width is None:
        col_width = _calculate_col_width(mapping)
    if max_rows is None:
        max_rows = OPTIONS["display_max_rows"]
    summary = [f"{title}:"]
    if mapping:
        len_mapping = len(mapping)
        if not _get_boolean_with_default(expand_option_name, default=True):
            summary = [f"{summary[0]} ({len_mapping})"]
        elif len_mapping > max_rows:
            summary = [f"{summary[0]} ({max_rows}/{len_mapping})"]
            first_rows = max_rows // 2 + max_rows % 2
            items = list(mapping.items())
            summary += [summarizer(k, v, col_width) for k, v in items[:first_rows]]
            if max_rows > 1:
                last_rows = max_rows // 2
                summary += [pretty_print("    ...", col_width) + " ..."]
                summary += [summarizer(k, v, col_width) for k, v in items[-last_rows:]]
        else:
            summary += [summarizer(k, v, col_width) for k, v in mapping.items()]
    else:
        summary += [EMPTY_REPR]
    return "\n".join(summary)
