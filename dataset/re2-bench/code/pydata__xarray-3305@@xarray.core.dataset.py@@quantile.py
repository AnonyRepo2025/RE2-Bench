import copy
import functools
import sys
import warnings
from collections import OrderedDict, defaultdict
from distutils.version import LooseVersion
from numbers import Number
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    DefaultDict,
    Dict,
    Hashable,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
    overload,
)
import numpy as np
import pandas as pd
import xarray as xr
from ..coding.cftimeindex import _parse_array_of_cftime_strings
from ..plot.dataset_plot import _Dataset_PlotMethods
from . import (
    alignment,
    dtypes,
    duck_array_ops,
    formatting,
    groupby,
    ops,
    pdcompat,
    resample,
    rolling,
    utils,
)
from .alignment import _broadcast_helper, _get_broadcast_dims_map_common_coords, align
from .common import (
    ALL_DIMS,
    DataWithCoords,
    ImplementsDatasetReduce,
    _contains_datetime_like_objects,
)
from .coordinates import (
    DatasetCoordinates,
    LevelCoordinatesSource,
    assert_coordinate_consistent,
    remap_label_indexers,
)
from .duck_array_ops import datetime_to_numeric
from .indexes import Indexes, default_indexes, isel_variable_and_index, roll_index
from .merge import (
    dataset_merge_method,
    dataset_update_method,
    merge_data_and_coords,
    merge_variables,
)
from .options import OPTIONS, _get_keep_attrs
from .pycompat import dask_array_type
from .utils import (
    Frozen,
    SortedKeysDict,
    _check_inplace,
    decode_numpy_dict_values,
    either_dict_or_kwargs,
    hashable,
    maybe_wrap_array,
    is_dict_like,
    is_list_like,
)
from .variable import IndexVariable, Variable, as_variable, broadcast_variables
from ..backends import AbstractDataStore, ZarrStore
from .dataarray import DataArray
from .merge import DatasetLike
from dask.delayed import Delayed
import dask
import dask
import dask.array as da
import dask.array as da
import dask
import dask
from .dataarray import DataArray
from ..backends.api import dump_to_store
from ..backends.api import to_netcdf
from ..backends.api import to_zarr
from .dataarray import DataArray
from .dataarray import DataArray
from . import missing
from .missing import interp_na, _apply_over_vars_with_dim
from .missing import ffill, _apply_over_vars_with_dim
from .missing import bfill, _apply_over_vars_with_dim
from .dataarray import DataArray
from sparse import COO
import dask.array as da
import dask.dataframe as dd
from .dataarray import DataArray
from .variable import Variable
from .variable import Variable
import dask.array as da
import dask
from dask.base import tokenize
import itertools
from .dataarray import DataArray
from .dataarray import DataArray
from dask.highlevelgraph import HighLevelGraph
import dask  # noqa
from dask import sharedict

_DATETIMEINDEX_COMPONENTS = [
    "year",
    "month",
    "day",
    "hour",
    "minute",
    "second",
    "microsecond",
    "nanosecond",
    "date",
    "time",
    "dayofyear",
    "weekofyear",
    "dayofweek",
    "quarter",
]

class Dataset(Mapping, ImplementsDatasetReduce, DataWithCoords):
    __slots__ = (
        "_accessors",
        "_attrs",
        "_coord_names",
        "_dims",
        "_encoding",
        "_file_obj",
        "_indexes",
        "_variables",
    )
    _groupby_cls = groupby.DatasetGroupBy
    _rolling_cls = rolling.DatasetRolling
    _coarsen_cls = rolling.DatasetCoarsen
    _resample_cls = resample.DatasetResample
    __default = object()
    __hash__ = None  # type: ignore
    @property
    def variables(self) -> Mapping[Hashable, Variable]:
        """Low level interface to Dataset contents as dict of Variable objects.

        This ordered dictionary is frozen to prevent mutation that could
        violate Dataset invariants. It contains all variable objects
        constituting the Dataset, including both data variables and
        coordinates.
        """
        return Frozen(self._variables)
    @attrs.setter
    def attrs(self, value: Mapping[Hashable, Any]) -> None:
        self._attrs = OrderedDict(value)
    @property
    def dims(self) -> Mapping[Hashable, int]:
        """Mapping from dimension names to lengths.

        Cannot be modified directly, but is updated when adding new variables.

        Note that type of this object differs from `DataArray.dims`.
        See `Dataset.sizes` and `DataArray.sizes` for consistently named
        properties.
        """
        return Frozen(SortedKeysDict(self._dims))
    @classmethod
    def _construct_direct(
        cls,
        variables,
        coord_names,
        dims,
        attrs=None,
        indexes=None,
        encoding=None,
        file_obj=None,
    ):
        """Shortcut around __init__ for internal use when we want to skip
        costly validation
        """
        obj = object.__new__(cls)
        obj._variables = variables
        obj._coord_names = coord_names
        obj._dims = dims
        obj._indexes = indexes
        obj._attrs = attrs
        obj._file_obj = file_obj
        obj._encoding = encoding
        obj._accessors = None
        return obj
    def _replace(  # type: ignore
        self,
        variables: "OrderedDict[Any, Variable]" = None,
        coord_names: Set[Hashable] = None,
        dims: Dict[Any, int] = None,
        attrs: "Optional[OrderedDict]" = __default,
        indexes: "Optional[OrderedDict[Any, pd.Index]]" = __default,
        encoding: Optional[dict] = __default,
        inplace: bool = False,
    ) -> "Dataset":
        """Fastpath constructor for internal use.

        Returns an object with optionally with replaced attributes.

        Explicitly passed arguments are *not* copied when placed on the new
        dataset. It is up to the caller to ensure that they have the right type
        and are not used elsewhere.
        """
        if inplace:
            if variables is not None:
                self._variables = variables
            if coord_names is not None:
                self._coord_names = coord_names
            if dims is not None:
                self._dims = dims
            if attrs is not self.__default:
                self._attrs = attrs
            if indexes is not self.__default:
                self._indexes = indexes
            if encoding is not self.__default:
                self._encoding = encoding
            obj = self
        else:
            if variables is None:
                variables = self._variables.copy()
            if coord_names is None:
                coord_names = self._coord_names.copy()
            if dims is None:
                dims = self._dims.copy()
            if attrs is self.__default:
                attrs = copy.copy(self._attrs)
            if indexes is self.__default:
                indexes = copy.copy(self._indexes)
            if encoding is self.__default:
                encoding = copy.copy(self._encoding)
            obj = self._construct_direct(
                variables, coord_names, dims, attrs, indexes, encoding
            )
        return obj
    def _replace_with_new_dims(  # type: ignore
        self,
        variables: "OrderedDict[Any, Variable]",
        coord_names: set = None,
        attrs: Optional["OrderedDict"] = __default,
        indexes: "OrderedDict[Any, pd.Index]" = __default,
        inplace: bool = False,
    ) -> "Dataset":
        """Replace variables with recalculated dimensions."""
        dims = calculate_dimensions(variables)
        return self._replace(
            variables, coord_names, dims, attrs, indexes, inplace=inplace
        )
    @property
    def indexes(self) -> Indexes:
        """Mapping of pandas.Index objects used for label based indexing
        """
        if self._indexes is None:
            self._indexes = default_indexes(self._variables, self._dims)
        return Indexes(self._indexes)
    @property
    def coords(self) -> DatasetCoordinates:
        """Dictionary of xarray.DataArray objects corresponding to coordinate
        variables
        """
        return DatasetCoordinates(self)
    def quantile(
        self, q, dim=None, interpolation="linear", numeric_only=False, keep_attrs=None
    ):
        """Compute the qth quantile of the data along the specified dimension.

        Returns the qth quantiles(s) of the array elements for each variable
        in the Dataset.

        Parameters
        ----------
        q : float in range of [0,1] or array-like of floats
            Quantile to compute, which must be between 0 and 1 inclusive.
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
            If True, the dataset's attributes (`attrs`) will be copied from
            the original object to the new one.  If False (default), the new
            object will be returned without attributes.
        numeric_only : bool, optional
            If True, only apply ``func`` to variables with a numeric dtype.

        Returns
        -------
        quantiles : Dataset
            If `q` is a single quantile, then the result is a scalar for each
            variable in data_vars. If multiple percentiles are given, first
            axis of the result corresponds to the quantile and a quantile
            dimension is added to the return Dataset. The other dimensions are
            the dimensions that remain after the reduction of the array.

        See Also
        --------
        numpy.nanpercentile, pandas.Series.quantile, DataArray.quantile
        """

        if isinstance(dim, str):
            dims = {dim}
        elif dim is None:
            dims = set(self.dims)
        else:
            dims = set(dim)

        _assert_empty(
            [d for d in dims if d not in self.dims],
            "Dataset does not contain the dimensions: %s",
        )

        q = np.asarray(q, dtype=np.float64)

        variables = OrderedDict()
        for name, var in self.variables.items():
            reduce_dims = [d for d in var.dims if d in dims]
            if reduce_dims or not var.dims:
                if name not in self.coords:
                    if (
                        not numeric_only
                        or np.issubdtype(var.dtype, np.number)
                        or var.dtype == np.bool_
                    ):
                        if len(reduce_dims) == var.ndim:
                            # prefer to aggregate over axis=None rather than
                            # axis=(0, 1) if they will be equivalent, because
                            # the former is often more efficient
                            reduce_dims = None
                        variables[name] = var.quantile(
                            q,
                            dim=reduce_dims,
                            interpolation=interpolation,
                            keep_attrs=keep_attrs,
                        )

            else:
                variables[name] = var

        # construct the new dataset
        coord_names = {k for k in self.coords if k in variables}
        indexes = OrderedDict((k, v) for k, v in self.indexes.items() if k in variables)
        if keep_attrs is None:
            keep_attrs = _get_keep_attrs(default=False)
        attrs = self.attrs if keep_attrs else None
        new = self._replace_with_new_dims(
            variables, coord_names=coord_names, attrs=attrs, indexes=indexes
        )
        if "quantile" in new.dims:
            new.coords["quantile"] = Variable("quantile", q)
        else:
            new.coords["quantile"] = q
        return new