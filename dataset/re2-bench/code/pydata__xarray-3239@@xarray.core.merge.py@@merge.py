from collections import OrderedDict
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Hashable,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Sequence,
    Set,
    Tuple,
    Union,
)
import pandas as pd
from . import dtypes, pdcompat
from .alignment import deep_align
from .utils import Frozen
from .variable import Variable, as_variable, assert_unique_multiindex_level_names
from .dataarray import DataArray
from .dataset import Dataset
from .dataarray import DataArray  # noqa: F811
from .dataset import Dataset
from .dataarray import DataArray  # noqa: F811
from .dataset import Dataset
from .dataarray import DataArray  # noqa: F811
from .dataset import Dataset
from .dataset import calculate_dimensions
from .dataarray import DataArray  # noqa: F811
from .dataset import Dataset
from .dataarray import DataArray  # noqa: F811
from .dataset import Dataset

PANDAS_TYPES = (pd.Series, pd.DataFrame, pdcompat.Panel)
_VALID_COMPAT = Frozen(
    {
        "identical": 0,
        "equals": 1,
        "broadcast_equals": 2,
        "minimal": 3,
        "no_conflicts": 4,
        "override": 5,
    }
)

def merge(objects, compat="no_conflicts", join="outer", fill_value=dtypes.NA):
    """Merge any number of xarray objects into a single Dataset as variables.

    Parameters
    ----------
    objects : Iterable[Union[xarray.Dataset, xarray.DataArray, dict]]
        Merge together all variables from these objects. If any of them are
        DataArray objects, they must have a name.
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
    join : {'outer', 'inner', 'left', 'right', 'exact'}, optional
        String indicating how to combine differing indexes in objects.

        - 'outer': use the union of object indexes
        - 'inner': use the intersection of object indexes
        - 'left': use indexes from the first object with each dimension
        - 'right': use indexes from the last object with each dimension
        - 'exact': instead of aligning, raise `ValueError` when indexes to be
          aligned are not equal
        - 'override': if indexes are of same size, rewrite indexes to be
          those of the first object with that dimension. Indexes for the same
          dimension must have the same size in all objects.
    fill_value : scalar, optional
        Value to use for newly missing values

    Returns
    -------
    Dataset
        Dataset with combined variables from each object.

    Examples
    --------
    >>> arrays = [xr.DataArray(n, name='var%d' % n) for n in range(5)]
    >>> xr.merge(arrays)
    <xarray.Dataset>
    Dimensions:  ()
    Coordinates:
        *empty*
    Data variables:
        var0     int64 0
        var1     int64 1
        var2     int64 2
        var3     int64 3
        var4     int64 4

    Raises
    ------
    xarray.MergeError
        If any variables with the same name have conflicting values.

    See also
    --------
    concat
    """  # noqa
    from .dataarray import DataArray  # noqa: F811
    from .dataset import Dataset

    dict_like_objects = list()
    for obj in objects:
        if not (isinstance(obj, (DataArray, Dataset, dict))):
            raise TypeError(
                "objects must be an iterable containing only "
                "Dataset(s), DataArray(s), and dictionaries."
            )

        obj = obj.to_dataset() if isinstance(obj, DataArray) else obj
        dict_like_objects.append(obj)

    variables, coord_names, dims = merge_core(
        dict_like_objects, compat, join, fill_value=fill_value
    )
    # TODO: don't always recompute indexes
    merged = Dataset._construct_direct(variables, coord_names, dims, indexes=None)

    return merged
