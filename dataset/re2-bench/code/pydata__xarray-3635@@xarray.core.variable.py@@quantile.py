import copy
import functools
import itertools
import warnings
from collections import defaultdict
from datetime import timedelta
from distutils.version import LooseVersion
from typing import Any, Dict, Hashable, Mapping, TypeVar, Union
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
    _default,
    decode_numpy_dict_values,
    either_dict_or_kwargs,
    ensure_us_time_resolution,
    infix_dims,
)
import dask.array as da
from .dataarray import DataArray
from dask.base import normalize_token
import dask
import dask.array as da
import sparse
from .computation import apply_ufunc
import bottleneck as bn
from dask.base import normalize_token

NON_NUMPY_SUPPORTED_ARRAY_TYPES = (
    indexing.ExplicitlyIndexed,
    pd.Index,
) + dask_array_type
BASIC_INDEXING_TYPES = integer_types + (slice,)  # type: ignore
VariableType = TypeVar("VariableType", bound="Variable")
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
            raise ValueError(
                f"replacement data must match the Variable's shape. "
                f"replacement data has shape {data.shape}; Variable has shape {self.shape}"
            )
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

        from .computation import apply_ufunc

        if keep_attrs is None:
            keep_attrs = _get_keep_attrs(default=False)

        scalar = utils.is_scalar(q)
        q = np.atleast_1d(np.asarray(q, dtype=np.float64))

        # TODO: remove once numpy >= 1.15.0 is the minimum requirement
        if np.count_nonzero(q < 0.0) or np.count_nonzero(q > 1.0):
            raise ValueError("Quantiles must be in the range [0, 1]")

        if dim is None:
            dim = self.dims

        if utils.is_scalar(dim):
            dim = [dim]

        def _wrapper(npa, **kwargs):
            # move quantile axis to end. required for apply_ufunc

            # TODO: use np.nanquantile once numpy >= 1.15.0 is the minimum requirement
            return np.moveaxis(np.nanpercentile(npa, **kwargs), 0, -1)

        axis = np.arange(-1, -1 * len(dim) - 1, -1)
        result = apply_ufunc(
            _wrapper,
            self,
            input_core_dims=[dim],
            exclude_dims=set(dim),
            output_core_dims=[["quantile"]],
            output_dtypes=[np.float64],
            output_sizes={"quantile": len(q)},
            dask="parallelized",
            kwargs={"q": q * 100, "axis": axis, "interpolation": interpolation},
        )

        # for backward compatibility
        result = result.transpose("quantile", ...)
        if scalar:
            result = result.squeeze("quantile")
        if keep_attrs:
            result.attrs = self._attrs
        return result