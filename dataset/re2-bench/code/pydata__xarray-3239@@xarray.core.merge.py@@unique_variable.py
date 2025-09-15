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

def unique_variable(name, variables, compat="broadcast_equals", equals=None):
    # type: (Any, List[Variable], str, bool) -> Variable
    """Return the unique variable from a list of variables or raise MergeError.

    Parameters
    ----------
    name : hashable
        Name for this variable.
    variables : list of xarray.Variable
        List of Variable objects, all of which go by the same name in different
        inputs.
    compat : {'identical', 'equals', 'broadcast_equals', 'no_conflicts', 'override'}, optional
        Type of equality check to use.
    equals: None or bool,
        corresponding to result of compat test

    Returns
    -------
    Variable to use in the result.

    Raises
    ------
    MergeError: if any of the variables are not equal.
    """  # noqa
    out = variables[0]

    if len(variables) == 1 or compat == "override":
        return out

    combine_method = None

    if compat == "minimal":
        compat = "broadcast_equals"

    if compat == "broadcast_equals":
        dim_lengths = broadcast_dimension_size(variables)
        out = out.set_dims(dim_lengths)

    if compat == "no_conflicts":
        combine_method = "fillna"

    if equals is None:
        out = out.compute()
        for var in variables[1:]:
            equals = getattr(out, compat)(var)
            if not equals:
                break

    if not equals:
        raise MergeError(
            "conflicting values for variable %r on objects to be combined. You can skip this check by specifying compat='override'."
            % (name)
        )

    if combine_method:
        for var in variables[1:]:
            out = getattr(out, combine_method)(var)

    return out
