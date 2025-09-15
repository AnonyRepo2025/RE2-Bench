import itertools
import warnings
from collections import Counter
from textwrap import dedent
import pandas as pd
from . import dtypes
from .concat import concat
from .dataarray import DataArray
from .dataset import Dataset
from .merge import merge

_CONCAT_DIM_DEFAULT = "__infer_concat_dim__"

def _check_dimension_depth_tile_ids(combined_tile_ids):
    """
    Check all tuples are the same length, i.e. check that all lists are
    nested to the same depth.
    """
    tile_ids = combined_tile_ids.keys()
    nesting_depths = [len(tile_id) for tile_id in tile_ids]
    if not nesting_depths:
        nesting_depths = [0]
    if not set(nesting_depths) == {nesting_depths[0]}:
        raise ValueError(
            "The supplied objects do not form a hypercube because"
            " sub-lists do not have consistent depths"
        )
    # return these just to be reused in _check_shape_tile_ids
    return tile_ids, nesting_depths
