import copy
import functools
import itertools
import numbers
import warnings
from collections import defaultdict
from datetime import timedelta
from distutils.version import LooseVersion
from typing import (
    Any,
    Dict,
    Hashable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)
import numpy as np
import pandas as pd
import xarray as xr  # only for Dataset and DataArray
from . import arithmetic, common, dtypes, duck_array_ops, indexing, nputils, ops, utils
from .indexing import (
    BasicIndexer,
    OuterIndexer,
    PandasIndexAdapter,
    VectorizedIndexer,
    as_indexable,
)
from .npcompat import IS_NEP18_ACTIVE
from .options import _get_keep_attrs
from .pycompat import (
    cupy_array_type,
    dask_array_type,
    integer_types,
    is_duck_dask_array,
)
from .utils import (
    OrderedSet,
    _default,
    decode_numpy_dict_values,
    drop_dims_from_indexers,
    either_dict_or_kwargs,
    ensure_us_time_resolution,
    infix_dims,
    is_duck_array,
    maybe_coerce_to_str,
)
from .dataarray import DataArray
from .computation import apply_ufunc
from dask.base import normalize_token
import dask
import dask.array as da
import sparse
from .computation import apply_ufunc
import bottleneck as bn
from .computation import apply_ufunc
from .computation import apply_ufunc
from dask.base import normalize_token

NON_NUMPY_SUPPORTED_ARRAY_TYPES = (
    (
        indexing.ExplicitlyIndexed,
        pd.Index,
    )
    + dask_array_type
    + cupy_array_type
)
BASIC_INDEXING_TYPES = integer_types + (slice,)  # type: ignore
VariableType = TypeVar("VariableType", bound="Variable")
Coordinate = utils.alias(IndexVariable, "Coordinate")

def as_variable(obj, name=None) -> "Union[Variable, IndexVariable]":
    """Convert an object into a Variable.

    Parameters
    ----------
    obj : object
        Object to convert into a Variable.

        - If the object is already a Variable, return a shallow copy.
        - Otherwise, if the object has 'dims' and 'data' attributes, convert
          it into a new Variable.
        - If all else fails, attempt to convert the object into a Variable by
          unpacking it into the arguments for creating a new Variable.
    name : str, optional
        If provided:

        - `obj` can be a 1D array, which is assumed to label coordinate values
          along a dimension of this given name.
        - Variables with name matching one of their dimensions are converted
          into `IndexVariable` objects.

    Returns
    -------
    var : Variable
        The newly created variable.

    """
    from .dataarray import DataArray

    # TODO: consider extending this method to automatically handle Iris and
    if isinstance(obj, DataArray):
        # extract the primary Variable from DataArrays
        obj = obj.variable

    if isinstance(obj, Variable):
        obj = obj.copy(deep=False)
    elif isinstance(obj, tuple):
        if isinstance(obj[1], DataArray):
            # TODO: change into TypeError
            warnings.warn(
                (
                    "Using a DataArray object to construct a variable is"
                    " ambiguous, please extract the data using the .data property."
                    " This will raise a TypeError in 0.19.0."
                ),
                DeprecationWarning,
            )
        try:
            obj = Variable(*obj)
        except (TypeError, ValueError) as error:
            # use .format() instead of % because it handles tuples consistently
            raise error.__class__(
                "Could not convert tuple of form "
                "(dims, data[, attrs, encoding]): "
                "{} to Variable.".format(obj)
            )
    elif utils.is_scalar(obj):
        obj = Variable([], obj)
    elif isinstance(obj, (pd.Index, IndexVariable)) and obj.name is not None:
        obj = Variable(obj.name, obj)
    elif isinstance(obj, (set, dict)):
        raise TypeError("variable {!r} has invalid type {!r}".format(name, type(obj)))
    elif name is not None:
        data = as_compatible_data(obj)
        if data.ndim != 1:
            raise MissingDimensionsError(
                "cannot set variable %r with %r-dimensional data "
                "without explicit dimension names. Pass a tuple of "
                "(dims, data) instead." % (name, data.ndim)
            )
        obj = Variable(name, data, fastpath=True)
    else:
        raise TypeError(
            "unable to convert object into a variable without an "
            "explicit list of dimensions: %r" % obj
        )

    if name is not None and name in obj.dims:
        # convert the Variable into an Index
        if obj.ndim != 1:
            raise MissingDimensionsError(
                "%r has more than 1-dimension and the same name as one of its "
                "dimensions %r. xarray disallows such variables because they "
                "conflict with the coordinates used to label "
                "dimensions." % (name, obj.dims)
            )
        obj = obj.to_index_variable()

    return obj
