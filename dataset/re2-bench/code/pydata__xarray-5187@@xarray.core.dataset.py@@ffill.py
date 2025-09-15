import copy
import datetime
import inspect
import sys
import warnings
from collections import defaultdict
from distutils.version import LooseVersion
from html import escape
from numbers import Number
from operator import methodcaller
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
    TypeVar,
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
    formatting_html,
    groupby,
    ops,
    resample,
    rolling,
    utils,
    weighted,
)
from .alignment import _broadcast_helper, _get_broadcast_dims_map_common_coords, align
from .arithmetic import DatasetArithmetic
from .common import DataWithCoords, _contains_datetime_like_objects
from .coordinates import (
    DatasetCoordinates,
    assert_coordinate_consistent,
    remap_label_indexers,
)
from .duck_array_ops import datetime_to_numeric
from .indexes import (
    Indexes,
    default_indexes,
    isel_variable_and_index,
    propagate_indexes,
    remove_unused_levels_categories,
    roll_index,
)
from .indexing import is_fancy_indexer
from .merge import (
    dataset_merge_method,
    dataset_update_method,
    merge_coordinates_without_align,
    merge_data_and_coords,
)
from .missing import get_clean_interp_index
from .options import OPTIONS, _get_keep_attrs
from .pycompat import is_duck_dask_array, sparse_array_type
from .utils import (
    Default,
    Frozen,
    HybridMappingProxy,
    SortedKeysDict,
    _default,
    decode_numpy_dict_values,
    drop_dims_from_indexers,
    either_dict_or_kwargs,
    hashable,
    infix_dims,
    is_dict_like,
    is_scalar,
    maybe_wrap_array,
)
from .variable import (
    IndexVariable,
    Variable,
    as_variable,
    assert_unique_multiindex_level_names,
    broadcast_variables,
)
from ..backends import AbstractDataStore, ZarrStore
from .dataarray import DataArray
from .merge import CoercibleMapping
import dask.array as da
from dask.base import tokenize
from dask.delayed import Delayed
from dask.base import normalize_token
import dask
import dask
import dask.array as da
import dask.array as da
import dask
from dask import is_dask_collection
from dask.highlevelgraph import HighLevelGraph
from dask.optimization import cull
from .dataarray import DataArray
from ..backends.api import dump_to_store
from ..backends.api import to_netcdf
from ..backends.api import to_zarr
from .dataarray import DataArray
from .dataarray import DataArray
from . import missing
from .missing import _apply_over_vars_with_dim, interp_na
from .missing import _apply_over_vars_with_dim, ffill
from .missing import _apply_over_vars_with_dim, bfill
from .dataarray import DataArray
from sparse import COO
import dask.array as da
import dask.dataframe as dd
from .dataarray import DataArray
from .dataarray import DataArray
from .dataarray import DataArray
from .variable import Variable
from .variable import Variable
import dask.array
from .parallel import map_blocks
from scipy.optimize import curve_fit
import dask.array as da
import dask
import itertools
from dask.highlevelgraph import HighLevelGraph
from dask import sharedict
from dask.base import flatten, replace_name_in_key

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

class Dataset(DataWithCoords, DatasetArithmetic, Mapping):
    __slots__ = (
        "_attrs",
        "_cache",
        "_coord_names",
        "_dims",
        "_encoding",
        "_close",
        "_indexes",
        "_variables",
        "__weakref__",
    )
    _groupby_cls = groupby.DatasetGroupBy
    _rolling_cls = rolling.DatasetRolling
    _coarsen_cls = rolling.DatasetCoarsen
    _resample_cls = resample.DatasetResample
    _weighted_cls = weighted.DatasetWeighted
    __hash__ = None  # type: ignore[assignment]
    plot = utils.UncachedAccessor(_Dataset_PlotMethods)
    def __init__(
        self,
        # could make a VariableArgs to use more generally, and refine these
        # categories
        data_vars: Mapping[Hashable, Any] = None,
        coords: Mapping[Hashable, Any] = None,
        attrs: Mapping[Hashable, Any] = None,
    ):
        # TODO(shoyer): expose indexes as a public argument in __init__

        if data_vars is None:
            data_vars = {}
        if coords is None:
            coords = {}

        both_data_and_coords = set(data_vars) & set(coords)
        if both_data_and_coords:
            raise ValueError(
                "variables %r are found in both data_vars and coords"
                % both_data_and_coords
            )

        if isinstance(coords, Dataset):
            coords = coords.variables

        variables, coord_names, dims, indexes, _ = merge_data_and_coords(
            data_vars, coords, compat="broadcast_equals"
        )

        self._attrs = dict(attrs) if attrs is not None else None
        self._close = None
        self._encoding = None
        self._variables = variables
        self._coord_names = coord_names
        self._dims = dims
        self._indexes = indexes
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
        self._attrs = dict(value)
    @encoding.setter
    def encoding(self, value: Mapping) -> None:
        self._encoding = dict(value)
    @property
    def dims(self) -> Mapping[Hashable, int]:
        """Mapping from dimension names to lengths.

        Cannot be modified directly, but is updated when adding new variables.

        Note that type of this object differs from `DataArray.dims`.
        See `Dataset.sizes` and `DataArray.sizes` for consistently named
        properties.
        """
        return Frozen(SortedKeysDict(self._dims))
    @property
    def sizes(self) -> Mapping[Hashable, int]:
        """Mapping from dimension names to lengths.

        Cannot be modified directly, but is updated when adding new variables.

        This is an alias for `Dataset.dims` provided for the benefit of
        consistency with `DataArray.sizes`.

        See Also
        --------
        DataArray.sizes
        """
        return self.dims
    @classmethod
    def _construct_direct(
        cls,
        variables,
        coord_names,
        dims=None,
        attrs=None,
        indexes=None,
        encoding=None,
        close=None,
    ):
        """Shortcut around __init__ for internal use when we want to skip
        costly validation
        """
        if dims is None:
            dims = calculate_dimensions(variables)
        obj = object.__new__(cls)
        obj._variables = variables
        obj._coord_names = coord_names
        obj._dims = dims
        obj._indexes = indexes
        obj._attrs = attrs
        obj._close = close
        obj._encoding = encoding
        return obj
    def _replace(
        self,
        variables: Dict[Hashable, Variable] = None,
        coord_names: Set[Hashable] = None,
        dims: Dict[Any, int] = None,
        attrs: Union[Dict[Hashable, Any], None, Default] = _default,
        indexes: Union[Dict[Any, pd.Index], None, Default] = _default,
        encoding: Union[dict, None, Default] = _default,
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
            if attrs is not _default:
                self._attrs = attrs
            if indexes is not _default:
                self._indexes = indexes
            if encoding is not _default:
                self._encoding = encoding
            obj = self
        else:
            if variables is None:
                variables = self._variables.copy()
            if coord_names is None:
                coord_names = self._coord_names.copy()
            if dims is None:
                dims = self._dims.copy()
            if attrs is _default:
                attrs = copy.copy(self._attrs)
            if indexes is _default:
                indexes = copy.copy(self._indexes)
            if encoding is _default:
                encoding = copy.copy(self._encoding)
            obj = self._construct_direct(
                variables, coord_names, dims, attrs, indexes, encoding
            )
        return obj
    def copy(self, deep: bool = False, data: Mapping = None) -> "Dataset":
        """Returns a copy of this dataset.

        If `deep=True`, a deep copy is made of each of the component variables.
        Otherwise, a shallow copy of each of the component variable is made, so
        that the underlying memory region of the new dataset is the same as in
        the original dataset.

        Use `data` to create a new object with the same structure as
        original but entirely new data.

        Parameters
        ----------
        deep : bool, optional
            Whether each component variable is loaded into memory and copied onto
            the new object. Default is False.
        data : dict-like, optional
            Data to use in the new object. Each item in `data` must have same
            shape as corresponding data variable in original. When `data` is
            used, `deep` is ignored for the data variables and only used for
            coords.

        Returns
        -------
        object : Dataset
            New object with dimensions, attributes, coordinates, name, encoding,
            and optionally data copied from original.

        Examples
        --------
        Shallow copy versus deep copy

        >>> da = xr.DataArray(np.random.randn(2, 3))
        >>> ds = xr.Dataset(
        ...     {"foo": da, "bar": ("x", [-1, 2])},
        ...     coords={"x": ["one", "two"]},
        ... )
        >>> ds.copy()
        <xarray.Dataset>
        Dimensions:  (dim_0: 2, dim_1: 3, x: 2)
        Coordinates:
          * x        (x) <U3 'one' 'two'
        Dimensions without coordinates: dim_0, dim_1
        Data variables:
            foo      (dim_0, dim_1) float64 1.764 0.4002 0.9787 2.241 1.868 -0.9773
            bar      (x) int64 -1 2

        >>> ds_0 = ds.copy(deep=False)
        >>> ds_0["foo"][0, 0] = 7
        >>> ds_0
        <xarray.Dataset>
        Dimensions:  (dim_0: 2, dim_1: 3, x: 2)
        Coordinates:
          * x        (x) <U3 'one' 'two'
        Dimensions without coordinates: dim_0, dim_1
        Data variables:
            foo      (dim_0, dim_1) float64 7.0 0.4002 0.9787 2.241 1.868 -0.9773
            bar      (x) int64 -1 2

        >>> ds
        <xarray.Dataset>
        Dimensions:  (dim_0: 2, dim_1: 3, x: 2)
        Coordinates:
          * x        (x) <U3 'one' 'two'
        Dimensions without coordinates: dim_0, dim_1
        Data variables:
            foo      (dim_0, dim_1) float64 7.0 0.4002 0.9787 2.241 1.868 -0.9773
            bar      (x) int64 -1 2

        Changing the data using the ``data`` argument maintains the
        structure of the original object, but with the new data. Original
        object is unaffected.

        >>> ds.copy(data={"foo": np.arange(6).reshape(2, 3), "bar": ["a", "b"]})
        <xarray.Dataset>
        Dimensions:  (dim_0: 2, dim_1: 3, x: 2)
        Coordinates:
          * x        (x) <U3 'one' 'two'
        Dimensions without coordinates: dim_0, dim_1
        Data variables:
            foo      (dim_0, dim_1) int64 0 1 2 3 4 5
            bar      (x) <U1 'a' 'b'

        >>> ds
        <xarray.Dataset>
        Dimensions:  (dim_0: 2, dim_1: 3, x: 2)
        Coordinates:
          * x        (x) <U3 'one' 'two'
        Dimensions without coordinates: dim_0, dim_1
        Data variables:
            foo      (dim_0, dim_1) float64 7.0 0.4002 0.9787 2.241 1.868 -0.9773
            bar      (x) int64 -1 2

        See Also
        --------
        pandas.DataFrame.copy
        """
        if data is None:
            variables = {k: v.copy(deep=deep) for k, v in self._variables.items()}
        elif not utils.is_dict_like(data):
            raise ValueError("Data must be dict-like")
        else:
            var_keys = set(self.data_vars.keys())
            data_keys = set(data.keys())
            keys_not_in_vars = data_keys - var_keys
            if keys_not_in_vars:
                raise ValueError(
                    "Data must only contain variables in original "
                    "dataset. Extra variables: {}".format(keys_not_in_vars)
                )
            keys_missing_from_data = var_keys - data_keys
            if keys_missing_from_data:
                raise ValueError(
                    "Data must contain all variables in original "
                    "dataset. Data is missing {}".format(keys_missing_from_data)
                )
            variables = {
                k: v.copy(deep=deep, data=data.get(k))
                for k, v in self._variables.items()
            }

        attrs = copy.deepcopy(self._attrs) if deep else copy.copy(self._attrs)

        return self._replace(variables, attrs=attrs)
    @property
    def _level_coords(self) -> Dict[str, Hashable]:
        """Return a mapping of all MultiIndex levels and their corresponding
        coordinate name.
        """
        level_coords: Dict[str, Hashable] = {}
        for name, index in self.indexes.items():
            if isinstance(index, pd.MultiIndex):
                level_names = index.names
                (dim,) = self.variables[name].dims
                level_coords.update({lname: dim for lname in level_names})
        return level_coords
    def _construct_dataarray(self, name: Hashable) -> "DataArray":
        """Construct a DataArray by indexing this dataset"""
        from .dataarray import DataArray

        try:
            variable = self._variables[name]
        except KeyError:
            _, name, variable = _get_virtual_variable(
                self._variables, name, self._level_coords, self.dims
            )

        needed_dims = set(variable.dims)

        coords: Dict[Hashable, Variable] = {}
        # preserve ordering
        for k in self._variables:
            if k in self._coord_names and set(self.variables[k].dims) <= needed_dims:
                coords[k] = self.variables[k]

        if self._indexes is None:
            indexes = None
        else:
            indexes = {k: v for k, v in self._indexes.items() if k in coords}

        return DataArray(variable, coords, name=name, indexes=indexes, fastpath=True)
    def __iter__(self) -> Iterator[Hashable]:
        return iter(self.data_vars)
    def __getitem__(self, key):
        """Access variables or coordinates this dataset as a
        :py:class:`~xarray.DataArray`.

        Indexing with a list of names will return a new ``Dataset`` object.
        """
        if utils.is_dict_like(key):
            return self.isel(**cast(Mapping, key))

        if hashable(key):
            return self._construct_dataarray(key)
        else:
            return self._copy_listed(np.asarray(key))
    def __setitem__(self, key: Hashable, value) -> None:
        """Add an array to this dataset.

        If value is a `DataArray`, call its `select_vars()` method, rename it
        to `key` and merge the contents of the resulting dataset into this
        dataset.

        If value is an `Variable` object (or tuple of form
        ``(dims, data[, attrs])``), add it to this dataset as a new
        variable.
        """
        if utils.is_dict_like(key):
            raise NotImplementedError(
                "cannot yet use a dictionary as a key to set Dataset values"
            )

        self.update({key: value})
    @property
    def indexes(self) -> Indexes:
        """Mapping of pandas.Index objects used for label based indexing"""
        if self._indexes is None:
            self._indexes = default_indexes(self._variables, self._dims)
        return Indexes(self._indexes)
    @property
    def coords(self) -> DatasetCoordinates:
        """Dictionary of xarray.DataArray objects corresponding to coordinate
        variables
        """
        return DatasetCoordinates(self)
    @property
    def data_vars(self) -> DataVariables:
        """Dictionary of DataArray objects corresponding to data variables"""
        return DataVariables(self)
    def update(self, other: "CoercibleMapping") -> "Dataset":
        """Update this dataset's variables with those from another dataset.

        Just like :py:meth:`dict.update` this is a in-place operation.

        Parameters
        ----------
        other : Dataset or mapping
            Variables with which to update this dataset. One of:

            - Dataset
            - mapping {var name: DataArray}
            - mapping {var name: Variable}
            - mapping {var name: (dimension name, array-like)}
            - mapping {var name: (tuple of dimension names, array-like)}

        Returns
        -------
        updated : Dataset
            Updated dataset. Note that since the update is in-place this is the input
            dataset.

            It is deprecated since version 0.17 and scheduled to be removed in 0.19.

        Raises
        ------
        ValueError
            If any dimensions would have inconsistent sizes in the updated
            dataset.

        See Also
        --------
        Dataset.assign
        """
        merge_result = dataset_update_method(self, other)
        return self._replace(inplace=True, **merge_result._asdict())
    def ffill(self, dim: Hashable, limit: int = None) -> "Dataset":
        """Fill NaN values by propogating values forward

        *Requires bottleneck.*

        Parameters
        ----------
        dim : Hashable
            Specifies the dimension along which to propagate values when
            filling.
        limit : int, default: None
            The maximum number of consecutive NaN values to forward fill. In
            other words, if there is a gap with more than this number of
            consecutive NaNs, it will only be partially filled. Must be greater
            than 0 or None for no limit. Must be None or greater than or equal
            to axis length if filling along chunked axes (dimensions).

        Returns
        -------
        Dataset
        """
        from .missing import _apply_over_vars_with_dim, ffill

        new = _apply_over_vars_with_dim(ffill, self, dim=dim, limit=limit)
        return new