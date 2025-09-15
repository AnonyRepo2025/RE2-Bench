import itertools
import warnings
from collections import Counter, OrderedDict
from textwrap import dedent
import pandas as pd
from .dataarray import DataArray
from .dataset import Dataset
from .concat import concat
from . import dtypes
from .merge import merge

_CONCAT_DIM_DEFAULT = '__infer_concat_dim__'

def combine_by_coords(datasets, compat='no_conflicts', data_vars='all',
                      coords='different', fill_value=dtypes.NA):
    """
    Attempt to auto-magically combine the given datasets into one by using
    dimension coordinates.

    This method attempts to combine a group of datasets along any number of
    dimensions into a single entity by inspecting coords and metadata and using
    a combination of concat and merge.

    Will attempt to order the datasets such that the values in their dimension
    coordinates are monotonic along all dimensions. If it cannot determine the
    order in which to concatenate the datasets, it will raise a ValueError.
    Non-coordinate dimensions will be ignored, as will any coordinate
    dimensions which do not vary between each dataset.

    Aligns coordinates, but different variables on datasets can cause it
    to fail under some scenarios. In complex cases, you may need to clean up
    your data and use concat/merge explicitly (also see `manual_combine`).

    Works well if, for example, you have N years of data and M data variables,
    and each combination of a distinct time period and set of data variables is
    saved as its own dataset. Also useful for if you have a simulation which is
    parallelized in multiple dimensions, but has global coordinates saved in
    each file specifying the positions of points within the global domain.

    Parameters
    ----------
    datasets : sequence of xarray.Dataset
        Dataset objects to combine.
    compat : {'identical', 'equals', 'broadcast_equals',
              'no_conflicts'}, optional
        String indicating how to compare variables of the same name for
        potential conflicts:

        - 'broadcast_equals': all values must be equal when variables are
          broadcast against each other to ensure common dimensions.
        - 'equals': all values and dimensions must be the same.
        - 'identical': all values, dimensions and attributes must be the
          same.
        - 'no_conflicts': only values which are not null in both datasets
          must be equal. The returned dataset then contains the combination
          of all non-null values.
    data_vars : {'minimal', 'different', 'all' or list of str}, optional
        Details are in the documentation of concat
    coords : {'minimal', 'different', 'all' or list of str}, optional
        Details are in the documentation of concat
    fill_value : scalar, optional
        Value to use for newly missing values

    Returns
    -------
    combined : xarray.Dataset

    See also
    --------
    concat
    merge
    combine_nested

    Examples
    --------

    Combining two datasets using their common dimension coordinates. Notice
    they are concatenated based on the values in their dimension coordinates,
    not on their position in the list passed to `combine_by_coords`.

    >>> x1
    <xarray.Dataset>
    Dimensions:         (x: 3)
    Coords:
      * position        (x) int64   0 1 2
    Data variables:
        temperature     (x) float64 11.04 23.57 20.77 ...

    >>> x2
    <xarray.Dataset>
    Dimensions:         (x: 3)
    Coords:
      * position        (x) int64   3 4 5
    Data variables:
        temperature     (x) float64 6.97 8.13 7.42 ...

    >>> combined = xr.combine_by_coords([x2, x1])
    <xarray.Dataset>
    Dimensions:         (x: 6)
    Coords:
      * position        (x) int64   0 1 2 3 4 5
    Data variables:
        temperature     (x) float64 11.04 23.57 20.77 ...
    """

    # Group by data vars
    sorted_datasets = sorted(datasets, key=vars_as_keys)
    grouped_by_vars = itertools.groupby(sorted_datasets, key=vars_as_keys)

    # Perform the multidimensional combine on each group of data variables
    # before merging back together
    concatenated_grouped_by_data_vars = []
    for vars, datasets_with_same_vars in grouped_by_vars:
        combined_ids, concat_dims = _infer_concat_order_from_coords(
            list(datasets_with_same_vars))

        _check_shape_tile_ids(combined_ids)

        # Concatenate along all of concat_dims one by one to create single ds
        concatenated = _combine_nd(combined_ids, concat_dims=concat_dims,
                                   data_vars=data_vars, coords=coords,
                                   fill_value=fill_value)

        # Check the overall coordinates are monotonically increasing
        for dim in concat_dims:
            indexes = concatenated.indexes.get(dim)
            if not (indexes.is_monotonic_increasing
                    or indexes.is_monotonic_decreasing):
                raise ValueError("Resulting object does not have monotonic"
                                 " global indexes along dimension {}"
                                 .format(dim))
        concatenated_grouped_by_data_vars.append(concatenated)

    return merge(concatenated_grouped_by_data_vars, compat=compat,
                 fill_value=fill_value)
