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

def drop_dims_from_indexers(
    indexers: Mapping[Hashable, Any],
    dims: Union[list, Mapping[Hashable, int]],
    missing_dims: str,
) -> Mapping[Hashable, Any]:
    """Depending on the setting of missing_dims, drop any dimensions from indexers that
    are not present in dims.

    Parameters
    ----------
    indexers : dict
    dims : sequence
    missing_dims : {"raise", "warn", "ignore"}
    """

    if missing_dims == "raise":
        invalid = indexers.keys() - set(dims)
        if invalid:
            raise ValueError(
                f"Dimensions {invalid} do not exist. Expected one or more of {dims}"
            )

        return indexers

    elif missing_dims == "warn":

        # don't modify input
        indexers = dict(indexers)

        invalid = indexers.keys() - set(dims)
        if invalid:
            warnings.warn(
                f"Dimensions {invalid} do not exist. Expected one or more of {dims}"
            )
        for key in invalid:
            indexers.pop(key)

        return indexers

    elif missing_dims == "ignore":
        return {key: val for key, val in indexers.items() if key in dims}

    else:
        raise ValueError(
            f"Unrecognised option {missing_dims} for missing_dims argument"
        )
