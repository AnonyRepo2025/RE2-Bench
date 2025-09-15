import itertools
import warnings
from collections import Counter, OrderedDict
from textwrap import dedent
import pandas as pd
from . import dtypes
from .concat import concat
from .dataarray import DataArray
from .dataset import Dataset
from .merge import merge

_CONCAT_DIM_DEFAULT = "__infer_concat_dim__"

def _combine_nd(
    combined_ids,
    concat_dims,
    data_vars="all",
    coords="different",
    compat="no_conflicts",
    fill_value=dtypes.NA,
    join="outer",
):
    """
    Combines an N-dimensional structure of datasets into one by applying a
    series of either concat and merge operations along each dimension.

    No checks are performed on the consistency of the datasets, concat_dims or
    tile_IDs, because it is assumed that this has already been done.

    Parameters
    ----------
    combined_ids : Dict[Tuple[int, ...]], xarray.Dataset]
        Structure containing all datasets to be concatenated with "tile_IDs" as
        keys, which specify position within the desired final combined result.
    concat_dims : sequence of str
        The dimensions along which the datasets should be concatenated. Must be
        in order, and the length must match the length of the tuples used as
        keys in combined_ids. If the string is a dimension name then concat
        along that dimension, if it is None then merge.

    Returns
    -------
    combined_ds : xarray.Dataset
    """

    example_tile_id = next(iter(combined_ids.keys()))

    n_dims = len(example_tile_id)
    if len(concat_dims) != n_dims:
        raise ValueError(
            "concat_dims has length {} but the datasets "
            "passed are nested in a {}-dimensional structure".format(
                len(concat_dims), n_dims
            )
        )

    # Each iteration of this loop reduces the length of the tile_ids tuples
    # by one. It always combines along the first dimension, removing the first
    # element of the tuple
    for concat_dim in concat_dims:
        combined_ids = _combine_all_along_first_dim(
            combined_ids,
            dim=concat_dim,
            data_vars=data_vars,
            coords=coords,
            compat=compat,
            fill_value=fill_value,
            join=join,
        )
    (combined_ds,) = combined_ids.values()
    return combined_ds
