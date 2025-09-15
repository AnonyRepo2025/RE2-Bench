import contextlib
import functools
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
from .variable import IndexVariable
from .variable import Variable

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

def _element_formatter(
    elements: Collection[Hashable],
    col_width: int,
    max_rows: Optional[int] = None,
    delimiter: str = ", ",
) -> str:
    """
    Formats elements for better readability.

    Once it becomes wider than the display width it will create a newline and
    continue indented to col_width.
    Once there are more rows than the maximum displayed rows it will start
    removing rows.

    Parameters
    ----------
    elements : Collection of hashable
        Elements to join together.
    col_width : int
        The width to indent to if a newline has been made.
    max_rows : int, optional
        The maximum number of allowed rows. The default is None.
    delimiter : str, optional
        Delimiter to use between each element. The default is ", ".
    """
    elements_len = len(elements)
    out = [""]
    length_row = 0
    for i, v in enumerate(elements):
        delim = delimiter if i < elements_len - 1 else ""
        v_delim = f"{v}{delim}"
        length_element = len(v_delim)
        length_row += length_element

        # Create a new row if the next elements makes the print wider than
        # the maximum display width:
        if col_width + length_row > OPTIONS["display_width"]:
            out[-1] = out[-1].rstrip()  # Remove trailing whitespace.
            out.append("\n" + pretty_print("", col_width) + v_delim)
            length_row = length_element
        else:
            out[-1] += v_delim

    # If there are too many rows of dimensions trim some away:
    if max_rows and (len(out) > max_rows):
        first_rows = calc_max_rows_first(max_rows)
        last_rows = calc_max_rows_last(max_rows)
        out = (
            out[:first_rows]
            + ["\n" + pretty_print("", col_width) + "..."]
            + (out[-last_rows:] if max_rows > 1 else [])
        )
    return "".join(out)
