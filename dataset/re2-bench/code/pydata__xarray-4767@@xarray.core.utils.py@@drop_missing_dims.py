import contextlib
import functools
import io
import itertools
import os.path
import re
import warnings
from enum import Enum
from typing import (
    AbstractSet,
    Any,
    Callable,
    Collection,
    Container,
    Dict,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSet,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
)
import numpy as np
import pandas as pd
from ..coding.cftimeindex import CFTimeIndex
from . import duck_array_ops
from .variable import NON_NUMPY_SUPPORTED_ARRAY_TYPES
from dask.base import normalize_token

K = TypeVar("K")
V = TypeVar("V")
T = TypeVar("T")
_default = Default.token

def drop_missing_dims(
    supplied_dims: Collection, dims: Collection, missing_dims: str
) -> Collection:
    """Depending on the setting of missing_dims, drop any dimensions from supplied_dims that
    are not present in dims.

    Parameters
    ----------
    supplied_dims : dict
    dims : sequence
    missing_dims : {"raise", "warn", "ignore"}
    """

    if missing_dims == "raise":
        supplied_dims_set = set(val for val in supplied_dims if val is not ...)
        invalid = supplied_dims_set - set(dims)
        if invalid:
            raise ValueError(
                f"Dimensions {invalid} do not exist. Expected one or more of {dims}"
            )

        return supplied_dims

    elif missing_dims == "warn":

        invalid = set(supplied_dims) - set(dims)
        if invalid:
            warnings.warn(
                f"Dimensions {invalid} do not exist. Expected one or more of {dims}"
            )

        return [val for val in supplied_dims if val in dims or val is ...]

    elif missing_dims == "ignore":
        return [val for val in supplied_dims if val in dims or val is ...]

    else:
        raise ValueError(
            f"Unrecognised option {missing_dims} for missing_dims argument"
        )
