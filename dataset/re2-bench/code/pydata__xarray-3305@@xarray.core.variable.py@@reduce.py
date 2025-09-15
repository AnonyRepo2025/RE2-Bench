import functools
import itertools
from collections import OrderedDict, defaultdict
from datetime import timedelta
from distutils.version import LooseVersion
from typing import Any, Hashable, Mapping, Union
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
from .pycompat import dask_array_type, integer_types
from .utils import (
    OrderedSet,
    decode_numpy_dict_values,
    either_dict_or_kwargs,
    ensure_us_time_resolution,
)
import dask.array as da
from .dataarray import DataArray
import dask
import dask.array as da
import bottleneck as bn

NON_NUMPY_SUPPORTED_ARRAY_TYPES = (
    indexing.ExplicitlyIndexed,
    pd.Index,
) + dask_array_type
BASIC_INDEXING_TYPES = integer_types + (slice,)  # type: ignore
Coordinate = utils.alias(IndexVariable, "Coordinate")

class Variable(AbstractArray, SupportsArithmetic, NdimSizeLenMixin):
    __slots__ = ("_dims", "_data", "_attrs", "_encoding")
    to_variable = utils.alias(to_base_variable, "to_variable")
    to_coord = utils.alias(to_index_variable, "to_coord")
    __hash__ = None  # type: ignore
    _array_counter = itertools.count()
    @property
    def shape(self):
        return self._data.shape
    @data.setter
    def data(self, data):
        data = as_compatible_data(data)
        if data.shape != self.shape:
            raise ValueError("replacement data must match the Variable's shape")
        self._data = data
    @values.setter
    def values(self, values):
        self.data = values
    @dims.setter
    def dims(self, value):
        self._dims = self._parse_dimensions(value)
    def reduce(
        self,
        func,
        dim=None,
        axis=None,
        keep_attrs=None,
        keepdims=False,
        allow_lazy=False,
        **kwargs
    ):
        """Reduce this array by applying `func` along some dimension(s).

        Parameters
        ----------
        func : function
            Function which can be called in the form
            `func(x, axis=axis, **kwargs)` to return the result of reducing an
            np.ndarray over an integer valued axis.
        dim : str or sequence of str, optional
            Dimension(s) over which to apply `func`.
        axis : int or sequence of int, optional
            Axis(es) over which to apply `func`. Only one of the 'dim'
            and 'axis' arguments can be supplied. If neither are supplied, then
            the reduction is calculated over the flattened array (by calling
            `func(x)` without an axis argument).
        keep_attrs : bool, optional
            If True, the variable's attributes (`attrs`) will be copied from
            the original object to the new one.  If False (default), the new
            object will be returned without attributes.
        keepdims : bool, default False
            If True, the dimensions which are reduced are left in the result
            as dimensions of size one
        **kwargs : dict
            Additional keyword arguments passed on to `func`.

        Returns
        -------
        reduced : Array
            Array with summarized data and the indicated dimension(s)
            removed.
        """
        if dim is common.ALL_DIMS:
            dim = None
        if dim is not None and axis is not None:
            raise ValueError("cannot supply both 'axis' and 'dim' arguments")

        if dim is not None:
            axis = self.get_axis_num(dim)
        input_data = self.data if allow_lazy else self.values
        if axis is not None:
            data = func(input_data, axis=axis, **kwargs)
        else:
            data = func(input_data, **kwargs)

        if getattr(data, "shape", ()) == self.shape:
            dims = self.dims
        else:
            removed_axes = (
                range(self.ndim) if axis is None else np.atleast_1d(axis) % self.ndim
            )
            if keepdims:
                # Insert np.newaxis for removed dims
                slices = tuple(
                    np.newaxis if i in removed_axes else slice(None, None)
                    for i in range(self.ndim)
                )
                if getattr(data, "shape", None) is None:
                    # Reduce has produced a scalar value, not an array-like
                    data = np.asanyarray(data)[slices]
                else:
                    data = data[slices]
                dims = self.dims
            else:
                dims = [
                    adim for n, adim in enumerate(self.dims) if n not in removed_axes
                ]

        if keep_attrs is None:
            keep_attrs = _get_keep_attrs(default=False)
        attrs = self._attrs if keep_attrs else None

        return Variable(dims, data, attrs=attrs)