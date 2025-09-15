from __future__ import annotations
import warnings
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Hashable, Iterator, Mapping, Sequence, cast
import numpy as np
import pandas as pd
from . import formatting
from .indexes import Index, Indexes, PandasMultiIndex, assert_no_index_corrupted
from .merge import merge_coordinates_without_align, merge_coords
from .utils import Frozen, ReprObject
from .variable import Variable, calculate_dimensions
from .dataarray import DataArray
from .dataset import Dataset
from .dataset import Dataset
from .dataset import Dataset

_THIS_ARRAY = ReprObject("<this-array>")

def drop_coords(
    coords_to_drop: set[Hashable], variables, indexes: Indexes
) -> tuple[dict, dict]:
    """Drop index variables associated with variables in coords_to_drop."""
    # Only warn when we're dropping the dimension with the multi-indexed coordinate
    # If asked to drop a subset of the levels in a multi-index, we raise an error
    # later but skip the warning here.
    new_variables = dict(variables.copy())
    new_indexes = dict(indexes.copy())
    for key in coords_to_drop & set(indexes):
        maybe_midx = indexes[key]
        idx_coord_names = set(indexes.get_all_coords(key))
        if (
            isinstance(maybe_midx, PandasMultiIndex)
            and key == maybe_midx.dim
            and (idx_coord_names - coords_to_drop)
        ):
            warnings.warn(
                f"Updating MultiIndexed coordinate {key!r} would corrupt indices for "
                f"other variables: {list(maybe_midx.index.names)!r}. "
                f"This will raise an error in the future. Use `.drop_vars({idx_coord_names!r})` before "
                "assigning new coordinate values.",
                FutureWarning,
                stacklevel=4,
            )
            for k in idx_coord_names:
                del new_variables[k]
                del new_indexes[k]
    return new_variables, new_indexes
