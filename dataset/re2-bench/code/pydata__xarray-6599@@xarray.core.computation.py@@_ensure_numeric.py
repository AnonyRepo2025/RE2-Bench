from __future__ import annotations
import functools
import itertools
import operator
import warnings
from collections import Counter
from typing import (
    TYPE_CHECKING,
    AbstractSet,
    Any,
    Callable,
    Hashable,
    Iterable,
    Mapping,
    Sequence,
    overload,
)
import numpy as np
from . import dtypes, duck_array_ops, utils
from .alignment import align, deep_align
from .common import zeros_like
from .duck_array_ops import datetime_to_numeric
from .indexes import Index, filter_indexes_from_coords
from .merge import merge_attrs, merge_coordinates_without_align
from .options import OPTIONS, _get_keep_attrs
from .pycompat import is_duck_dask_array
from .utils import is_dict_like
from .variable import Variable
from .coordinates import Coordinates
from .dataarray import DataArray
from .dataset import Dataset
from .types import T_Xarray
from .dataarray import DataArray
from .dataset import Dataset
from .dataset import Dataset
from .groupby import _dummy_copy
from .groupby import GroupBy, peek_at
from .variable import Variable
from .variable import Variable, as_compatible_data
from .dataarray import DataArray
from .groupby import GroupBy
from .variable import Variable
from .dataarray import DataArray
from .dataarray import DataArray
from .dataarray import DataArray
from .variable import Variable
from .dataset import Dataset
from .dataarray import DataArray
from dask.array.core import unify_chunks
import dask.array
import dask.array as da

_NO_FILL_VALUE = utils.ReprObject("<no-fill-value>")
_DEFAULT_NAME = utils.ReprObject("<default-name>")
_JOINS_WITHOUT_FILL_VALUES = frozenset({"inner", "exact"})
SLICE_NONE = slice(None)

def _ensure_numeric(data: T_Xarray) -> T_Xarray:
    """Converts all datetime64 variables to float64

    Parameters
    ----------
    data : DataArray or Dataset
        Variables with possible datetime dtypes.

    Returns
    -------
    DataArray or Dataset
        Variables with datetime64 dtypes converted to float64.
    """
    from .dataset import Dataset

    def to_floatable(x: DataArray) -> DataArray:
        if x.dtype.kind == "M":
            # datetimes
            return x.copy(
                data=datetime_to_numeric(
                    x.data,
                    offset=np.datetime64("1970-01-01"),
                    datetime_unit="ns",
                ),
            )
        elif x.dtype.kind == "m":
            # timedeltas
            return x.astype(float)
        return x

    if isinstance(data, Dataset):
        return data.map(to_floatable)
    else:
        return to_floatable(data)
