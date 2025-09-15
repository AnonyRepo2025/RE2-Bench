from __future__ import annotations
import warnings
from contextlib import suppress
from html import escape
from textwrap import dedent
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    TypeVar,
    Union,
    overload,
)
import numpy as np
import pandas as pd
from . import dtypes, duck_array_ops, formatting, formatting_html, ops
from .npcompat import DTypeLike, DTypeLikeSave
from .options import OPTIONS, _get_keep_attrs
from .pycompat import is_duck_dask_array
from .utils import Frozen, either_dict_or_kwargs, is_scalar
import cftime
import datetime
from .dataarray import DataArray
from .dataset import Dataset
from .indexes import Index
from .resample import Resample
from .rolling_exp import RollingExp
from .types import ScalarOrArray, SideOptions, T_DataWithCoords
from .variable import Variable
from .dataarray import DataArray
from .dataset import Dataset
from .variable import Variable
from .variable import Variable
from .computation import apply_ufunc
from . import rolling_exp
from ..coding.cftimeindex import CFTimeIndex
from .dataarray import DataArray
from .resample import RESAMPLE_DIM
from .alignment import align
from .dataarray import DataArray
from .dataset import Dataset
from .computation import apply_ufunc
from .computation import apply_ufunc
from .computation import apply_ufunc
from .dataarray import DataArray
from .dataset import Dataset
from .variable import Variable
from .computation import apply_ufunc
import dask.array
from .resample_cftime import CFTimeGrouper

ALL_DIMS = ...
T_Resample = TypeVar("T_Resample", bound="Resample")
C = TypeVar("C")
T = TypeVar("T")
DTypeMaybeMapping = Union[DTypeLikeSave, Mapping[Any, DTypeLikeSave]]

class AbstractArray:
    __slots__ = ()
    def __format__(self: Any, format_spec: str = "") -> str:
        if format_spec != "":
            if self.shape == ():
                # Scalar values might be ok use format_spec with instead of repr:
                return self.data.__format__(format_spec)
            else:
                # TODO: If it's an array the formatting.array_repr(self) should
                # take format_spec as an input. If we'd only use self.data we
                # lose all the information about coords for example which is
                # important information:
                raise NotImplementedError(
                    "Using format_spec is only supported"
                    f" when shape is (). Got shape = {self.shape}."
                )
        else:
            return self.__repr__()