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
    @data.setter
    def data(self, data):
        data = as_compatible_data(data)
        if data.shape != self.shape:
            raise ValueError("replacement data must match the Variable's shape")
        self._data = data
    @dims.setter
    def dims(self, value):
        self._dims = self._parse_dimensions(value)
    def quantile(self, q, dim=None, interpolation="linear", keep_attrs=None):
        """Compute the qth quantile of the data along the specified dimension.

        Returns the qth quantiles(s) of the array elements.

        Parameters
        ----------
        q : float in range of [0,1] (or sequence of floats)
            Quantile to compute, which must be between 0 and 1
            inclusive.
        dim : str or sequence of str, optional
            Dimension(s) over which to apply quantile.
        interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
            This optional parameter specifies the interpolation method to
            use when the desired quantile lies between two data points
            ``i < j``:
                * linear: ``i + (j - i) * fraction``, where ``fraction`` is
                  the fractional part of the index surrounded by ``i`` and
                  ``j``.
                * lower: ``i``.
                * higher: ``j``.
                * nearest: ``i`` or ``j``, whichever is nearest.
                * midpoint: ``(i + j) / 2``.
        keep_attrs : bool, optional
            If True, the variable's attributes (`attrs`) will be copied from
            the original object to the new one.  If False (default), the new
            object will be returned without attributes.

        Returns
        -------
        quantiles : Variable
            If `q` is a single quantile, then the result
            is a scalar. If multiple percentiles are given, first axis of
            the result corresponds to the quantile and a quantile dimension
            is added to the return array. The other dimensions are the
            dimensions that remain after the reduction of the array.

        See Also
        --------
        numpy.nanpercentile, pandas.Series.quantile, Dataset.quantile,
        DataArray.quantile
        """
        if isinstance(self.data, dask_array_type):
            raise TypeError(
                "quantile does not work for arrays stored as dask "
                "arrays. Load the data via .compute() or .load() "
                "prior to calling this method."
            )

        q = np.asarray(q, dtype=np.float64)

        new_dims = list(self.dims)
        if dim is not None:
            axis = self.get_axis_num(dim)
            if utils.is_scalar(dim):
                new_dims.remove(dim)
            else:
                for d in dim:
                    new_dims.remove(d)
        else:
            axis = None
            new_dims = []

        # Only add the quantile dimension if q is array-like
        if q.ndim != 0:
            new_dims = ["quantile"] + new_dims

        qs = np.nanpercentile(
            self.data, q * 100.0, axis=axis, interpolation=interpolation
        )

        if keep_attrs is None:
            keep_attrs = _get_keep_attrs(default=False)
        attrs = self._attrs if keep_attrs else None

        return Variable(new_dims, qs, attrs)