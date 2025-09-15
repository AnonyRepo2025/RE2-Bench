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

def merge_variables(
    list_of_variables_dicts: List[Mapping[Any, Variable]],
    priority_vars: Mapping[Any, Variable] = None,
    compat: str = "minimal",
) -> "OrderedDict[Any, Variable]":
    """Merge dicts of variables, while resolving conflicts appropriately.

    Parameters
    ----------
    lists_of_variables_dicts : list of mappings with Variable values
        List of mappings for which each value is a xarray.Variable object.
    priority_vars : mapping with Variable or None values, optional
        If provided, variables are always taken from this dict in preference to
        the input variable dictionaries, without checking for conflicts.
    compat : {'identical', 'equals', 'broadcast_equals', 'minimal', 'no_conflicts', 'override'}, optional
        Type of equality check to use when checking for conflicts.

    Returns
    -------
    OrderedDict with keys taken by the union of keys on list_of_variable_dicts,
    and Variable values corresponding to those that should be found on the
    merged result.
    """  # noqa
    if priority_vars is None:
        priority_vars = {}

    _assert_compat_valid(compat)
    dim_compat = min(compat, "equals", key=_VALID_COMPAT.get)

    lookup = OrderedDefaultDict(list)
    for variables in list_of_variables_dicts:
        for name, var in variables.items():
            lookup[name].append(var)

    # n.b. it's important to fill up merged in the original order in which
    # variables appear
    merged = OrderedDict()  # type: OrderedDict[Any, Variable]

    for name, var_list in lookup.items():
        if name in priority_vars:
            # one of these arguments (e.g., the first for in-place arithmetic
            # or the second for Dataset.update) takes priority
            merged[name] = priority_vars[name]
        else:
            dim_variables = [var for var in var_list if (name,) == var.dims]
            if dim_variables:
                # if there are dimension coordinates, these must be equal (or
                # identical), and they take priority over non-dimension
                # coordinates
                merged[name] = unique_variable(name, dim_variables, dim_compat)
            else:
                try:
                    merged[name] = unique_variable(name, var_list, compat)
                except MergeError:
                    if compat != "minimal":
                        # we need more than "minimal" compatibility (for which
                        # we drop conflicting coordinates)
                        raise

    return merged
