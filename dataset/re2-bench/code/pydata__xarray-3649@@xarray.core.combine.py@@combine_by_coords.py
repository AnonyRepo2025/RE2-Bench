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

def combine_by_coords(
    datasets,
    compat="no_conflicts",
    data_vars="all",
    coords="different",
    fill_value=dtypes.NA,
    join="outer",
):
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
    compat : {'identical', 'equals', 'broadcast_equals', 'no_conflicts', 'override'}, optional
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
        - 'override': skip comparing and pick variable from first dataset
    data_vars : {'minimal', 'different', 'all' or list of str}, optional
        These data variables will be concatenated together:

        * 'minimal': Only data variables in which the dimension already
          appears are included.
        * 'different': Data variables which are not equal (ignoring
          attributes) across all datasets are also concatenated (as well as
          all for which dimension already appears). Beware: this option may
          load the data payload of data variables into memory if they are not
          already loaded.
        * 'all': All data variables will be concatenated.
        * list of str: The listed data variables will be concatenated, in
          addition to the 'minimal' data variables.

        If objects are DataArrays, `data_vars` must be 'all'.
    coords : {'minimal', 'different', 'all' or list of str}, optional
        As per the 'data_vars' kwarg, but for coordinate variables.
    fill_value : scalar, optional
        Value to use for newly missing values. If None, raises a ValueError if
        the passed Datasets do not create a complete hypercube.
    join : {'outer', 'inner', 'left', 'right', 'exact'}, optional
        String indicating how to combine differing indexes
        (excluding concat_dim) in objects

        - 'outer': use the union of object indexes
        - 'inner': use the intersection of object indexes
        - 'left': use indexes from the first object with each dimension
        - 'right': use indexes from the last object with each dimension
        - 'exact': instead of aligning, raise `ValueError` when indexes to be
          aligned are not equal
        - 'override': if indexes are of same size, rewrite indexes to be
          those of the first object with that dimension. Indexes for the same
          dimension must have the same size in all objects.

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

    >>> import numpy as np
    >>> import xarray as xr

    >>> x1 = xr.Dataset(
    ...     {
    ...         "temperature": (("y", "x"), 20 * np.random.rand(6).reshape(2, 3)),
    ...         "precipitation": (("y", "x"), np.random.rand(6).reshape(2, 3)),
    ...     },
    ...     coords={"y": [0, 1], "x": [10, 20, 30]},
    ... )
    >>> x2 = xr.Dataset(
    ...     {
    ...         "temperature": (("y", "x"), 20 * np.random.rand(6).reshape(2, 3)),
    ...         "precipitation": (("y", "x"), np.random.rand(6).reshape(2, 3)),
    ...     },
    ...     coords={"y": [2, 3], "x": [10, 20, 30]},
    ... )
    >>> x3 = xr.Dataset(
    ...     {
    ...         "temperature": (("y", "x"), 20 * np.random.rand(6).reshape(2, 3)),
    ...         "precipitation": (("y", "x"), np.random.rand(6).reshape(2, 3)),
    ...     },
    ...     coords={"y": [2, 3], "x": [40, 50, 60]},
    ... )

    >>> x1
    <xarray.Dataset>
    Dimensions:        (x: 3, y: 2)
    Coordinates:
    * y              (y) int64 0 1
    * x              (x) int64 10 20 30
    Data variables:
        temperature    (y, x) float64 1.654 10.63 7.015 2.543 13.93 9.436
        precipitation  (y, x) float64 0.2136 0.9974 0.7603 0.4679 0.3115 0.945

    >>> x2
    <xarray.Dataset>
    Dimensions:        (x: 3, y: 2)
    Coordinates:
    * y              (y) int64 2 3
    * x              (x) int64 10 20 30
    Data variables:
        temperature    (y, x) float64 9.341 0.1251 6.269 7.709 8.82 2.316
        precipitation  (y, x) float64 0.1728 0.1178 0.03018 0.6509 0.06938 0.3792

    >>> x3
    <xarray.Dataset>
    Dimensions:        (x: 3, y: 2)
    Coordinates:
    * y              (y) int64 2 3
    * x              (x) int64 40 50 60
    Data variables:
        temperature    (y, x) float64 2.789 2.446 6.551 12.46 2.22 15.96
        precipitation  (y, x) float64 0.4804 0.1902 0.2457 0.6125 0.4654 0.5953

    >>> xr.combine_by_coords([x2, x1])
    <xarray.Dataset>
    Dimensions:        (x: 3, y: 4)
    Coordinates:
    * x              (x) int64 10 20 30
    * y              (y) int64 0 1 2 3
    Data variables:
        temperature    (y, x) float64 1.654 10.63 7.015 2.543 ... 7.709 8.82 2.316
        precipitation  (y, x) float64 0.2136 0.9974 0.7603 ... 0.6509 0.06938 0.3792

    >>> xr.combine_by_coords([x3, x1])
    <xarray.Dataset>
    Dimensions:        (x: 6, y: 4)
    Coordinates:
    * x              (x) int64 10 20 30 40 50 60
    * y              (y) int64 0 1 2 3
    Data variables:
        temperature    (y, x) float64 1.654 10.63 7.015 nan ... nan 12.46 2.22 15.96
        precipitation  (y, x) float64 0.2136 0.9974 0.7603 ... 0.6125 0.4654 0.5953

    >>> xr.combine_by_coords([x3, x1], join='override')
    <xarray.Dataset>
    Dimensions:        (x: 3, y: 4)
    Coordinates:
    * x              (x) int64 10 20 30
    * y              (y) int64 0 1 2 3
    Data variables:
    temperature    (y, x) float64 1.654 10.63 7.015 2.543 ... 12.46 2.22 15.96
    precipitation  (y, x) float64 0.2136 0.9974 0.7603 ... 0.6125 0.4654 0.5953

    >>> xr.combine_by_coords([x1, x2, x3])
    <xarray.Dataset>
    Dimensions:        (x: 6, y: 4)
    Coordinates:
    * x              (x) int64 10 20 30 40 50 60
    * y              (y) int64 0 1 2 3
    Data variables:
    temperature    (y, x) float64 1.654 10.63 7.015 nan ... 12.46 2.22 15.96
    precipitation  (y, x) float64 0.2136 0.9974 0.7603 ... 0.6125 0.4654 0.5953
    """

    # Group by data vars
    sorted_datasets = sorted(datasets, key=vars_as_keys)
    grouped_by_vars = itertools.groupby(sorted_datasets, key=vars_as_keys)

    # Perform the multidimensional combine on each group of data variables
    # before merging back together
    concatenated_grouped_by_data_vars = []
    for vars, datasets_with_same_vars in grouped_by_vars:
        combined_ids, concat_dims = _infer_concat_order_from_coords(
            list(datasets_with_same_vars)
        )

        if fill_value is None:
            # check that datasets form complete hypercube
            _check_shape_tile_ids(combined_ids)
        else:
            # check only that all datasets have same dimension depth for these
            # vars
            _check_dimension_depth_tile_ids(combined_ids)

        # Concatenate along all of concat_dims one by one to create single ds
        concatenated = _combine_nd(
            combined_ids,
            concat_dims=concat_dims,
            data_vars=data_vars,
            coords=coords,
            compat=compat,
            fill_value=fill_value,
            join=join,
        )

        # Check the overall coordinates are monotonically increasing
        for dim in concat_dims:
            indexes = concatenated.indexes.get(dim)
            if not (indexes.is_monotonic_increasing or indexes.is_monotonic_decreasing):
                raise ValueError(
                    "Resulting object does not have monotonic"
                    " global indexes along dimension {}".format(dim)
                )
        concatenated_grouped_by_data_vars.append(concatenated)

    return merge(
        concatenated_grouped_by_data_vars,
        compat=compat,
        fill_value=fill_value,
        join=join,
    )
