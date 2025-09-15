import warnings
from contextlib import suppress
from html import escape
from textwrap import dedent
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    Iterator,
    List,
    Mapping,
    Tuple,
    TypeVar,
    Union,
)
import numpy as np
import pandas as pd
from . import dtypes, duck_array_ops, formatting, formatting_html, ops
from .arithmetic import SupportsArithmetic
from .npcompat import DTypeLike
from .options import OPTIONS, _get_keep_attrs
from .pycompat import is_duck_dask_array
from .rolling_exp import RollingExp
from .utils import Frozen, either_dict_or_kwargs, is_scalar
from .dataarray import DataArray
from .dataset import Dataset
from .variable import Variable
from .variable import Variable
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
from cftime import datetime as cftime_datetime
from .resample_cftime import CFTimeGrouper

ALL_DIMS = ...
C = TypeVar("C")
T = TypeVar("T")

class DataWithCoords(SupportsArithmetic, AttrAccessMixin):
    __slots__ = ()
    _rolling_exp_cls = RollingExp
    def astype(
        self: T,
        dtype,
        *,
        order=None,
        casting=None,
        subok=None,
        copy=None,
        keep_attrs=True,
    ) -> T:
        """
        Copy of the xarray object, with data cast to a specified type.
        Leaves coordinate dtype unchanged.

        Parameters
        ----------
        dtype : str or dtype
            Typecode or data-type to which the array is cast.
        order : {'C', 'F', 'A', 'K'}, optional
            Controls the memory layout order of the result. ‘C’ means C order,
            ‘F’ means Fortran order, ‘A’ means ‘F’ order if all the arrays are
            Fortran contiguous, ‘C’ order otherwise, and ‘K’ means as close to
            the order the array elements appear in memory as possible.
        casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
            Controls what kind of data casting may occur.

            * 'no' means the data types should not be cast at all.
            * 'equiv' means only byte-order changes are allowed.
            * 'safe' means only casts which can preserve values are allowed.
            * 'same_kind' means only safe casts or casts within a kind,
              like float64 to float32, are allowed.
            * 'unsafe' means any data conversions may be done.

        subok : bool, optional
            If True, then sub-classes will be passed-through, otherwise the
            returned array will be forced to be a base-class array.
        copy : bool, optional
            By default, astype always returns a newly allocated array. If this
            is set to False and the `dtype` requirement is satisfied, the input
            array is returned instead of a copy.
        keep_attrs : bool, optional
            By default, astype keeps attributes. Set to False to remove
            attributes in the returned object.

        Returns
        -------
        out : same as object
            New object with data cast to the specified type.

        Notes
        -----
        The ``order``, ``casting``, ``subok`` and ``copy`` arguments are only passed
        through to the ``astype`` method of the underlying array when a value
        different than ``None`` is supplied.
        Make sure to only supply these arguments if the underlying array class
        supports them.

        See also
        --------
        numpy.ndarray.astype
        dask.array.Array.astype
        sparse.COO.astype
        """
        from .computation import apply_ufunc

        kwargs = dict(order=order, casting=casting, subok=subok, copy=copy)
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        return apply_ufunc(
            duck_array_ops.astype,
            self,
            dtype,
            kwargs=kwargs,
            keep_attrs=keep_attrs,
            dask="allowed",
        )