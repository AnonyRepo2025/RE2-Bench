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

def combine_nested(
    datasets,
    concat_dim,
    compat="no_conflicts",
    data_vars="all",
    coords="different",
    fill_value=dtypes.NA,
    join="outer",
):
    """
    Explicitly combine an N-dimensional grid of datasets into one by using a
    succession of concat and merge operations along each dimension of the grid.

    Does not sort the supplied datasets under any circumstances, so the
    datasets must be passed in the order you wish them to be concatenated. It
    does align coordinates, but different variables on datasets can cause it to
    fail under some scenarios. In complex cases, you may need to clean up your
    data and use concat/merge explicitly.

    To concatenate along multiple dimensions the datasets must be passed as a
    nested list-of-lists, with a depth equal to the length of ``concat_dims``.
    ``manual_combine`` will concatenate along the top-level list first.

    Useful for combining datasets from a set of nested directories, or for
    collecting the output of a simulation parallelized along multiple
    dimensions.

    Parameters
    ----------
    datasets : list or nested list of xarray.Dataset objects.
        Dataset objects to combine.
        If concatenation or merging along more than one dimension is desired,
        then datasets must be supplied in a nested list-of-lists.
    concat_dim : str, or list of str, DataArray, Index or None
        Dimensions along which to concatenate variables, as used by
        :py:func:`xarray.concat`.
        Set ``concat_dim=[..., None, ...]`` explicitly to disable concatenation
        and merge instead along a particular dimension.
        The position of ``None`` in the list specifies the dimension of the
        nested-list input along which to merge.
        Must be the same length as the depth of the list passed to
        ``datasets``.
    compat : {'identical', 'equals', 'broadcast_equals',
              'no_conflicts', 'override'}, optional
        String indicating how to compare variables of the same name for
        potential merge conflicts:

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
        Details are in the documentation of concat
    coords : {'minimal', 'different', 'all' or list of str}, optional
        Details are in the documentation of concat
    fill_value : scalar, optional
        Value to use for newly missing values
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

    Examples
    --------

    A common task is collecting data from a parallelized simulation in which
    each processor wrote out to a separate file. A domain which was decomposed
    into 4 parts, 2 each along both the x and y axes, requires organising the
    datasets into a doubly-nested list, e.g:

    >>> x1y1
    <xarray.Dataset>
    Dimensions:         (x: 2, y: 2)
    Dimensions without coordinates: x, y
    Data variables:
      temperature       (x, y) float64 11.04 23.57 20.77 ...
      precipitation     (x, y) float64 5.904 2.453 3.404 ...

    >>> ds_grid = [[x1y1, x1y2], [x2y1, x2y2]]
    >>> combined = xr.combine_nested(ds_grid, concat_dim=['x', 'y'])
    <xarray.Dataset>
    Dimensions:         (x: 4, y: 4)
    Dimensions without coordinates: x, y
    Data variables:
      temperature       (x, y) float64 11.04 23.57 20.77 ...
      precipitation     (x, y) float64 5.904 2.453 3.404 ...

    ``manual_combine`` can also be used to explicitly merge datasets with
    different variables. For example if we have 4 datasets, which are divided
    along two times, and contain two different variables, we can pass ``None``
    to ``concat_dim`` to specify the dimension of the nested list over which
    we wish to use ``merge`` instead of ``concat``:

    >>> t1temp
    <xarray.Dataset>
    Dimensions:         (t: 5)
    Dimensions without coordinates: t
    Data variables:
      temperature       (t) float64 11.04 23.57 20.77 ...

    >>> t1precip
    <xarray.Dataset>
    Dimensions:         (t: 5)
    Dimensions without coordinates: t
    Data variables:
      precipitation     (t) float64 5.904 2.453 3.404 ...

    >>> ds_grid = [[t1temp, t1precip], [t2temp, t2precip]]
    >>> combined = xr.combine_nested(ds_grid, concat_dim=['t', None])
    <xarray.Dataset>
    Dimensions:         (t: 10)
    Dimensions without coordinates: t
    Data variables:
      temperature       (t) float64 11.04 23.57 20.77 ...
      precipitation     (t) float64 5.904 2.453 3.404 ...

    See also
    --------
    concat
    merge
    auto_combine
    """
    if isinstance(concat_dim, (str, DataArray)) or concat_dim is None:
        concat_dim = [concat_dim]

    # The IDs argument tells _manual_combine that datasets aren't yet sorted
    return _nested_combine(
        datasets,
        concat_dims=concat_dim,
        compat=compat,
        data_vars=data_vars,
        coords=coords,
        ids=False,
        fill_value=fill_value,
        join=join,
    )
