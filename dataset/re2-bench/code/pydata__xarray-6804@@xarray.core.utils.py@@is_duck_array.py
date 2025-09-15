from __future__ import annotations
import contextlib
import functools
import io
import itertools
import math
import os
import re
import sys
import warnings
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Collection,
    Container,
    Generic,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSet,
    TypeVar,
    cast,
    overload,
)
import numpy as np
import pandas as pd
from .types import ErrorOptionsWithWarn
from ..coding.cftimeindex import CFTimeIndex
from . import dtypes
from . import duck_array_ops
from .variable import NON_NUMPY_SUPPORTED_ARRAY_TYPES
from .dataarray import DataArray
from .pycompat import is_duck_dask_array
from typing import TypeGuard
from typing_extensions import TypeGuard
from dask.base import normalize_token

K = TypeVar("K")
V = TypeVar("V")
T = TypeVar("T")
_Accessor = TypeVar("_Accessor")
_default = Default.token

def is_duck_array(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        return True
    return (
        hasattr(value, "ndim")
        and hasattr(value, "shape")
        and hasattr(value, "dtype")
        and (
            (hasattr(value, "__array_function__") and hasattr(value, "__array_ufunc__"))
            or hasattr(value, "__array_namespace__")
        )
    )
