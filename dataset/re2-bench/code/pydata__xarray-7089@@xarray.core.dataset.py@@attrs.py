from __future__ import annotations
import copy
import datetime
import inspect
import itertools
import math
import sys
import warnings
from collections import defaultdict
from html import escape
from numbers import Number
from operator import methodcaller
from os import PathLike
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    Callable,
    Collection,
    Generic,
    Hashable,
    Iterable,
    Iterator,
    Literal,
    Mapping,
    MutableMapping,
    Sequence,
    cast,
    overload,
)
import numpy as np
import pandas as pd
from ..coding.calendar_ops import convert_calendar, interp_calendar
from ..coding.cftimeindex import CFTimeIndex, _parse_array_of_cftime_strings
from ..plot.dataset_plot import _Dataset_PlotMethods
from . import alignment
from . import dtypes as xrdtypes
from . import duck_array_ops, formatting, formatting_html, ops, utils
from ._reductions import DatasetReductions
from .alignment import _broadcast_helper, _get_broadcast_dims_map_common_coords, align
from .arithmetic import DatasetArithmetic
from .common import DataWithCoords, _contains_datetime_like_objects, get_chunksizes
from .computation import unify_chunks
from .coordinates import DatasetCoordinates, assert_coordinate_consistent
from .duck_array_ops import datetime_to_numeric
from .indexes import (
    Index,
    Indexes,
    PandasIndex,
    PandasMultiIndex,
    assert_no_index_corrupted,
    create_default_index_implicit,
    filter_indexes_from_coords,
    isel_indexes,
    remove_unused_levels_categories,
    roll_indexes,
)
from .indexing import is_fancy_indexer, map_index_queries
from .merge import (
    dataset_merge_method,
    dataset_update_method,
    merge_coordinates_without_align,
    merge_data_and_coords,
)
from .missing import get_clean_interp_index
from .npcompat import QUANTILE_METHODS, ArrayLike
from .options import OPTIONS, _get_keep_attrs
from .pycompat import is_duck_dask_array, sparse_array_type
from .types import T_Dataset
from .utils import (
    Default,
    Frozen,
    HybridMappingProxy,
    OrderedSet,
    _default,
    decode_numpy_dict_values,
    drop_dims_from_indexers,
    either_dict_or_kwargs,
    infix_dims,
    is_dict_like,
    is_scalar,
    maybe_wrap_array,
)
from .variable import (
    IndexVariable,
    Variable,
    as_variable,
    broadcast_variables,
    calculate_dimensions,
)
from ..backends import AbstractDataStore, ZarrStore
from ..backends.api import T_NetcdfEngine, T_NetcdfTypes
from .coordinates import Coordinates
from .dataarray import DataArray
from .groupby import DatasetGroupBy
from .merge import CoercibleMapping
from .resample import DatasetResample
from .rolling import DatasetCoarsen, DatasetRolling
from .types import (
        CFCalendar,
        CoarsenBoundaryOptions,
        CombineAttrsOptions,
        CompatOptions,
        DatetimeUnitOptions,
        Dims,
        ErrorOptions,
        ErrorOptionsWithWarn,
        InterpOptions,
        JoinOptions,
        PadModeOptions,
        PadReflectOptions,
        QueryEngineOptions,
        QueryParserOptions,
        ReindexMethodOptions,
        SideOptions,
        T_Xarray,
    )
from .weighted import DatasetWeighted
from .dataarray import DataArray
import dask.array as da
from dask.base import tokenize
from dask.delayed import Delayed
from dask.dataframe import DataFrame as DaskDataFrame
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
from .dataarray import DataArray
from .alignment import align
from .dataarray import DataArray
from ..backends.api import dump_to_store
from ..backends.api import to_netcdf
from ..backends.api import to_zarr
from ..coding.cftimeindex import CFTimeIndex
from .dataarray import DataArray
from .dataarray import DataArray
from . import missing
from .concat import concat
from .dataarray import DataArray
from .missing import _apply_over_vars_with_dim, interp_na
from .missing import _apply_over_vars_with_dim, ffill
from .missing import _apply_over_vars_with_dim, bfill
from .dataarray import DataArray
from sparse import COO
import dask.array as da
import dask.dataframe as dd
from .dataarray import DataArray
from .groupby import GroupBy
from .dataarray import DataArray
from .groupby import GroupBy
from .dataarray import DataArray
from .variable import Variable
from .variable import Variable
from .parallel import map_blocks
from .dataarray import DataArray
from scipy.optimize import curve_fit
from .alignment import broadcast
from .computation import apply_ufunc
from .dataarray import _THIS_ARRAY, DataArray
from .groupby import DatasetGroupBy
from .groupby import DatasetGroupBy
from .weighted import DatasetWeighted
from .rolling import DatasetRolling
from .rolling import DatasetCoarsen
from .resample import DatasetResample
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

class Dataset(DataWithCoords, DatasetReductions, DatasetArithmetic, Mapping[(Hashable, 'DataArray')]
):
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
    __hash__ = None  # type: ignore[assignment]
    plot = utils.UncachedAccessor(_Dataset_PlotMethods)
    def __init__(
        self,
        # could make a VariableArgs to use more generally, and refine these
        # categories
        data_vars: Mapping[Any, Any] | None = None,
        coords: Mapping[Any, Any] | None = None,
        attrs: Mapping[Any, Any] | None = None,
    ) -> None:
        # TODO(shoyer): expose indexes as a public argument in __init__

        if data_vars is None:
            data_vars = {}
        if coords is None:
            coords = {}

        both_data_and_coords = set(data_vars) & set(coords)
        if both_data_and_coords:
            raise ValueError(
                f"variables {both_data_and_coords!r} are found in both data_vars and coords"
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
    def variables(self) -> Frozen[Hashable, Variable]:
        """Low level interface to Dataset contents as dict of Variable objects.

        This ordered dictionary is frozen to prevent mutation that could
        violate Dataset invariants. It contains all variable objects
        constituting the Dataset, including both data variables and
        coordinates.
        """
        return Frozen(self._variables)
    @attrs.setter
    def attrs(self, value: Mapping[Any, Any]) -> None:
        self._attrs = dict(value)
    @encoding.setter
    def encoding(self, value: Mapping[Any, Any]) -> None:
        self._encoding = dict(value)
    @property
    def dims(self) -> Frozen[Hashable, int]:
        """Mapping from dimension names to lengths.

        Cannot be modified directly, but is updated when adding new variables.

        Note that type of this object differs from `DataArray.dims`.
        See `Dataset.sizes` and `DataArray.sizes` for consistently named
        properties.

        See Also
        --------
        Dataset.sizes
        DataArray.dims
        """
        return Frozen(self._dims)
    @property
    def sizes(self) -> Frozen[Hashable, int]:
        """Mapping from dimension names to lengths.

        Cannot be modified directly, but is updated when adding new variables.

        This is an alias for `Dataset.dims` provided for the benefit of
        consistency with `DataArray.sizes`.

        See Also
        --------
        DataArray.sizes
        """
        return self.dims
    def load(self: T_Dataset, **kwargs) -> T_Dataset:
        """Manually trigger loading and/or computation of this dataset's data
        from disk or a remote source into memory and return this dataset.
        Unlike compute, the original dataset is modified and returned.

        Normally, it should not be necessary to call this method in user code,
        because all xarray functions should either work on deferred data or
        load data automatically. However, this method can be necessary when
        working with many file objects on disk.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments passed on to ``dask.compute``.

        See Also
        --------
        dask.compute
        """
        # access .data to coerce everything to numpy or dask arrays
        lazy_data = {
            k: v._data for k, v in self.variables.items() if is_duck_dask_array(v._data)
        }
        if lazy_data:
            import dask.array as da

            # evaluate all the dask arrays simultaneously
            evaluated_data = da.compute(*lazy_data.values(), **kwargs)

            for k, data in zip(lazy_data, evaluated_data):
                self.variables[k].data = data

        # load everything else sequentially
        for k, v in self.variables.items():
            if k not in lazy_data:
                v.load()

        return self
    @classmethod
    def _construct_direct(
        cls: type[T_Dataset],
        variables: dict[Any, Variable],
        coord_names: set[Hashable],
        dims: dict[Any, int] | None = None,
        attrs: dict | None = None,
        indexes: dict[Any, Index] | None = None,
        encoding: dict | None = None,
        close: Callable[[], None] | None = None,
    ) -> T_Dataset:
        """Shortcut around __init__ for internal use when we want to skip
        costly validation
        """
        if dims is None:
            dims = calculate_dimensions(variables)
        if indexes is None:
            indexes = {}
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
        self: T_Dataset,
        variables: dict[Hashable, Variable] = None,
        coord_names: set[Hashable] | None = None,
        dims: dict[Any, int] | None = None,
        attrs: dict[Hashable, Any] | None | Default = _default,
        indexes: dict[Hashable, Index] | None = None,
        encoding: dict | None | Default = _default,
        inplace: bool = False,
    ) -> T_Dataset:
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
            if indexes is not None:
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
            if indexes is None:
                indexes = self._indexes.copy()
            if encoding is _default:
                encoding = copy.copy(self._encoding)
            obj = self._construct_direct(
                variables, coord_names, dims, attrs, indexes, encoding
            )
        return obj
    def _replace_with_new_dims(
        self: T_Dataset,
        variables: dict[Hashable, Variable],
        coord_names: set | None = None,
        attrs: dict[Hashable, Any] | None | Default = _default,
        indexes: dict[Hashable, Index] | None = None,
        inplace: bool = False,
    ) -> T_Dataset:
        """Replace variables with recalculated dimensions."""
        dims = calculate_dimensions(variables)
        return self._replace(
            variables, coord_names, dims, attrs, indexes, inplace=inplace
        )
    def copy(
        self: T_Dataset, deep: bool = False, data: Mapping[Any, ArrayLike] | None = None
    ) -> T_Dataset:
        """Returns a copy of this dataset.

        If `deep=True`, a deep copy is made of each of the component variables.
        Otherwise, a shallow copy of each of the component variable is made, so
        that the underlying memory region of the new dataset is the same as in
        the original dataset.

        Use `data` to create a new object with the same structure as
        original but entirely new data.

        Parameters
        ----------
        deep : bool, default: False
            Whether each component variable is loaded into memory and copied onto
            the new object. Default is False.
        data : dict-like or None, optional
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
            data = {}
        elif not utils.is_dict_like(data):
            raise ValueError("Data must be dict-like")

        if data:
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

        indexes, index_vars = self.xindexes.copy_indexes(deep=deep)

        variables = {}
        for k, v in self._variables.items():
            if k in index_vars:
                variables[k] = index_vars[k]
            else:
                variables[k] = v.copy(deep=deep, data=data.get(k))

        attrs = copy.deepcopy(self._attrs) if deep else copy.copy(self._attrs)
        encoding = copy.deepcopy(self._encoding) if deep else copy.copy(self._encoding)

        return self._replace(variables, indexes=indexes, attrs=attrs, encoding=encoding)
    def _construct_dataarray(self, name: Hashable) -> DataArray:
        """Construct a DataArray by indexing this dataset"""
        from .dataarray import DataArray

        try:
            variable = self._variables[name]
        except KeyError:
            _, name, variable = _get_virtual_variable(self._variables, name, self.dims)

        needed_dims = set(variable.dims)

        coords: dict[Hashable, Variable] = {}
        # preserve ordering
        for k in self._variables:
            if k in self._coord_names and set(self.variables[k].dims) <= needed_dims:
                coords[k] = self.variables[k]

        indexes = filter_indexes_from_coords(self._indexes, set(coords))

        return DataArray(variable, coords, name=name, indexes=indexes, fastpath=True)
    @property
    def _attr_sources(self) -> Iterable[Mapping[Hashable, Any]]:
        """Places to look-up items for attribute-style access"""
        yield from self._item_sources
        yield self.attrs
    @property
    def _item_sources(self) -> Iterable[Mapping[Hashable, Any]]:
        """Places to look-up items for key-completion"""
        yield self.data_vars
        yield HybridMappingProxy(keys=self._coord_names, mapping=self.coords)

        # virtual coordinates
        yield HybridMappingProxy(keys=self.dims, mapping=self)
    def __iter__(self) -> Iterator[Hashable]:
        return iter(self.data_vars)
    def __getitem__(
        self: T_Dataset, key: Mapping[Any, Any] | Hashable | Iterable[Hashable]
    ) -> T_Dataset | DataArray:
        """Access variables or coordinates of this dataset as a
        :py:class:`~xarray.DataArray` or a subset of variables or a indexed dataset.

        Indexing with a list of names will return a new ``Dataset`` object.
        """
        if utils.is_dict_like(key):
            return self.isel(**key)
        if utils.hashable(key):
            return self._construct_dataarray(key)
        if utils.iterable_of_hashable(key):
            return self._copy_listed(key)
        raise ValueError(f"Unsupported key-type {type(key)}")
    def __setitem__(
        self, key: Hashable | Iterable[Hashable] | Mapping, value: Any
    ) -> None:
        """Add an array to this dataset.
        Multiple arrays can be added at the same time, in which case each of
        the following operations is applied to the respective value.

        If key is dict-like, update all variables in the dataset
        one by one with the given value at the given location.
        If the given value is also a dataset, select corresponding variables
        in the given value and in the dataset to be changed.

        If value is a `
        from .dataarray import DataArray`, call its `select_vars()` method, rename it
        to `key` and merge the contents of the resulting dataset into this
        dataset.

        If value is a `Variable` object (or tuple of form
        ``(dims, data[, attrs])``), add it to this dataset as a new
        variable.
        """
        from .dataarray import DataArray

        if utils.is_dict_like(key):
            # check for consistency and convert value to dataset
            value = self._setitem_check(key, value)
            # loop over dataset variables and set new values
            processed = []
            for name, var in self.items():
                try:
                    var[key] = value[name]
                    processed.append(name)
                except Exception as e:
                    if processed:
                        raise RuntimeError(
                            "An error occurred while setting values of the"
                            f" variable '{name}'. The following variables have"
                            f" been successfully updated:\n{processed}"
                        ) from e
                    else:
                        raise e

        elif utils.hashable(key):
            if isinstance(value, Dataset):
                raise TypeError(
                    "Cannot assign a Dataset to a single key - only a DataArray or Variable "
                    "object can be stored under a single key."
                )
            self.update({key: value})

        elif utils.iterable_of_hashable(key):
            keylist = list(key)
            if len(keylist) == 0:
                raise ValueError("Empty list of variables to be set")
            if len(keylist) == 1:
                self.update({keylist[0]: value})
            else:
                if len(keylist) != len(value):
                    raise ValueError(
                        f"Different lengths of variables to be set "
                        f"({len(keylist)}) and data used as input for "
                        f"setting ({len(value)})"
                    )
                if isinstance(value, Dataset):
                    self.update(dict(zip(keylist, value.data_vars.values())))
                elif isinstance(value, DataArray):
                    raise ValueError("Cannot assign single DataArray to multiple keys")
                else:
                    self.update(dict(zip(keylist, value)))

        else:
            raise ValueError(f"Unsupported key-type {type(key)}")
    def identical(self, other: Dataset) -> bool:
        """Like equals, but also checks all dataset attributes and the
        attributes on all variables and coordinates.

        See Also
        --------
        Dataset.broadcast_equals
        Dataset.equals
        """
        try:
            return utils.dict_equiv(self.attrs, other.attrs) and self._all_compat(
                other, "identical"
            )
        except (TypeError, AttributeError):
            return False
    @property
    def xindexes(self) -> Indexes[Index]:
        """Mapping of xarray Index objects used for label based indexing."""
        return Indexes(self._indexes, {k: self._variables[k] for k in self._indexes})
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
    def chunk(
        self: T_Dataset,
        chunks: (
            int | Literal["auto"] | Mapping[Any, None | int | str | tuple[int, ...]]
        ) = {},  # {} even though it's technically unsafe, is being used intentionally here (#4667)
        name_prefix: str = "xarray-",
        token: str | None = None,
        lock: bool = False,
        inline_array: bool = False,
        **chunks_kwargs: Any,
    ) -> T_Dataset:
        """Coerce all arrays in this dataset into dask arrays with the given
        chunks.

        Non-dask arrays in this dataset will be converted to dask arrays. Dask
        arrays will be rechunked to the given chunk sizes.

        If neither chunks is not provided for one or more dimensions, chunk
        sizes along that dimension will not be updated; non-dask arrays will be
        converted into dask arrays with a single block.

        Parameters
        ----------
        chunks : int, tuple of int, "auto" or mapping of hashable to int, optional
            Chunk sizes along each dimension, e.g., ``5``, ``"auto"``, or
            ``{"x": 5, "y": 5}``.
        name_prefix : str, default: "xarray-"
            Prefix for the name of any new dask arrays.
        token : str, optional
            Token uniquely identifying this dataset.
        lock : bool, default: False
            Passed on to :py:func:`dask.array.from_array`, if the array is not
            already as dask array.
        inline_array: bool, default: False
            Passed on to :py:func:`dask.array.from_array`, if the array is not
            already as dask array.
        **chunks_kwargs : {dim: chunks, ...}, optional
            The keyword arguments form of ``chunks``.
            One of chunks or chunks_kwargs must be provided

        Returns
        -------
        chunked : xarray.Dataset

        See Also
        --------
        Dataset.chunks
        Dataset.chunksizes
        xarray.unify_chunks
        dask.array.from_array
        """
        if chunks is None and chunks_kwargs is None:
            warnings.warn(
                "None value for 'chunks' is deprecated. "
                "It will raise an error in the future. Use instead '{}'",
                category=FutureWarning,
            )
            chunks = {}

        if isinstance(chunks, (Number, str, int)):
            chunks = dict.fromkeys(self.dims, chunks)
        else:
            chunks = either_dict_or_kwargs(chunks, chunks_kwargs, "chunk")

        bad_dims = chunks.keys() - self.dims.keys()
        if bad_dims:
            raise ValueError(
                f"some chunks keys are not dimensions on this object: {bad_dims}"
            )

        variables = {
            k: _maybe_chunk(k, v, chunks, token, lock, name_prefix)
            for k, v in self.variables.items()
        }
        return self._replace(variables)
    def _validate_indexers(
        self, indexers: Mapping[Any, Any], missing_dims: ErrorOptionsWithWarn = "raise"
    ) -> Iterator[tuple[Hashable, int | slice | np.ndarray | Variable]]:
        """Here we make sure
        + indexer has a valid keys
        + indexer is in a valid data type
        + string indexers are cast to the appropriate date type if the
          associated index is a DatetimeIndex or CFTimeIndex
        """
        from ..coding.cftimeindex import CFTimeIndex
        from .dataarray import DataArray

        indexers = drop_dims_from_indexers(indexers, self.dims, missing_dims)

        # all indexers should be int, slice, np.ndarrays, or Variable
        for k, v in indexers.items():
            if isinstance(v, (int, slice, Variable)):
                yield k, v
            elif isinstance(v, DataArray):
                yield k, v.variable
            elif isinstance(v, tuple):
                yield k, as_variable(v)
            elif isinstance(v, Dataset):
                raise TypeError("cannot use a Dataset as an indexer")
            elif isinstance(v, Sequence) and len(v) == 0:
                yield k, np.empty((0,), dtype="int64")
            else:
                v = np.asarray(v)

                if v.dtype.kind in "US":
                    index = self._indexes[k].to_pandas_index()
                    if isinstance(index, pd.DatetimeIndex):
                        v = v.astype("datetime64[ns]")
                    elif isinstance(index, CFTimeIndex):
                        v = _parse_array_of_cftime_strings(v, index.date_type)

                if v.ndim > 1:
                    raise IndexError(
                        "Unlabeled multi-dimensional array cannot be "
                        "used for indexing: {}".format(k)
                    )
                yield k, v
    def _get_indexers_coords_and_indexes(self, indexers):
        """Extract coordinates and indexes from indexers.

        Only coordinate with a name different from any of self.variables will
        be attached.
        """
        from .dataarray import DataArray

        coords_list = []
        for k, v in indexers.items():
            if isinstance(v, DataArray):
                if v.dtype.kind == "b":
                    if v.ndim != 1:  # we only support 1-d boolean array
                        raise ValueError(
                            "{:d}d-boolean array is used for indexing along "
                            "dimension {!r}, but only 1d boolean arrays are "
                            "supported.".format(v.ndim, k)
                        )
                    # Make sure in case of boolean DataArray, its
                    # coordinate also should be indexed.
                    v_coords = v[v.values.nonzero()[0]].coords
                else:
                    v_coords = v.coords
                coords_list.append(v_coords)

        # we don't need to call align() explicitly or check indexes for
        # alignment, because merge_variables already checks for exact alignment
        # between dimension coordinates
        coords, indexes = merge_coordinates_without_align(coords_list)
        assert_coordinate_consistent(self, coords)

        # silently drop the conflicted variables.
        attached_coords = {k: v for k, v in coords.items() if k not in self._variables}
        attached_indexes = {
            k: v for k, v in indexes.items() if k not in self._variables
        }
        return attached_coords, attached_indexes
    def _isel_fancy(
        self: T_Dataset,
        indexers: Mapping[Any, Any],
        *,
        drop: bool,
        missing_dims: ErrorOptionsWithWarn = "raise",
    ) -> T_Dataset:
        valid_indexers = dict(self._validate_indexers(indexers, missing_dims))

        variables: dict[Hashable, Variable] = {}
        indexes, index_variables = isel_indexes(self.xindexes, valid_indexers)

        for name, var in self.variables.items():
            if name in index_variables:
                new_var = index_variables[name]
            else:
                var_indexers = {
                    k: v for k, v in valid_indexers.items() if k in var.dims
                }
                if var_indexers:
                    new_var = var.isel(indexers=var_indexers)
                    # drop scalar coordinates
                    # https://github.com/pydata/xarray/issues/6554
                    if name in self.coords and drop and new_var.ndim == 0:
                        continue
                else:
                    new_var = var.copy(deep=False)
                if name not in indexes:
                    new_var = new_var.to_base_variable()
            variables[name] = new_var

        coord_names = self._coord_names & variables.keys()
        selected = self._replace_with_new_dims(variables, coord_names, indexes)

        # Extract coordinates from indexers
        coord_vars, new_indexes = selected._get_indexers_coords_and_indexes(indexers)
        variables.update(coord_vars)
        indexes.update(new_indexes)
        coord_names = self._coord_names & variables.keys() | coord_vars.keys()
        return self._replace_with_new_dims(variables, coord_names, indexes=indexes)
    def _reindex_callback(
        self,
        aligner: alignment.Aligner,
        dim_pos_indexers: dict[Hashable, Any],
        variables: dict[Hashable, Variable],
        indexes: dict[Hashable, Index],
        fill_value: Any,
        exclude_dims: frozenset[Hashable],
        exclude_vars: frozenset[Hashable],
    ) -> Dataset:
        """Callback called from ``Aligner`` to create a new reindexed Dataset."""

        new_variables = variables.copy()
        new_indexes = indexes.copy()

        # re-assign variable metadata
        for name, new_var in new_variables.items():
            var = self._variables.get(name)
            if var is not None:
                new_var.attrs = var.attrs
                new_var.encoding = var.encoding

        # pass through indexes from excluded dimensions
        # no extra check needed for multi-coordinate indexes, potential conflicts
        # should already have been detected when aligning the indexes
        for name, idx in self._indexes.items():
            var = self._variables[name]
            if set(var.dims) <= exclude_dims:
                new_indexes[name] = idx
                new_variables[name] = var

        if not dim_pos_indexers:
            # fast path for no reindexing necessary
            if set(new_indexes) - set(self._indexes):
                # this only adds new indexes and their coordinate variables
                reindexed = self._overwrite_indexes(new_indexes, new_variables)
            else:
                reindexed = self.copy(deep=aligner.copy)
        else:
            to_reindex = {
                k: v
                for k, v in self.variables.items()
                if k not in variables and k not in exclude_vars
            }
            reindexed_vars = alignment.reindex_variables(
                to_reindex,
                dim_pos_indexers,
                copy=aligner.copy,
                fill_value=fill_value,
                sparse=aligner.sparse,
            )
            new_variables.update(reindexed_vars)
            new_coord_names = self._coord_names | set(new_indexes)
            reindexed = self._replace_with_new_dims(
                new_variables, new_coord_names, indexes=new_indexes
            )

        return reindexed
    def expand_dims(
        self,
        dim: None | Hashable | Sequence[Hashable] | Mapping[Any, Any] = None,
        axis: None | int | Sequence[int] = None,
        **dim_kwargs: Any,
    ) -> Dataset:
        """Return a new object with an additional axis (or axes) inserted at
        the corresponding position in the array shape.  The new object is a
        view into the underlying array, not a copy.

        If dim is already a scalar coordinate, it will be promoted to a 1D
        coordinate consisting of a single value.

        Parameters
        ----------
        dim : hashable, sequence of hashable, mapping, or None
            Dimensions to include on the new variable. If provided as hashable
            or sequence of hashable, then dimensions are inserted with length
            1. If provided as a mapping, then the keys are the new dimensions
            and the values are either integers (giving the length of the new
            dimensions) or array-like (giving the coordinates of the new
            dimensions).
        axis : int, sequence of int, or None, default: None
            Axis position(s) where new axis is to be inserted (position(s) on
            the result array). If a sequence of integers is passed,
            multiple axes are inserted. In this case, dim arguments should be
            same length list. If axis=None is passed, all the axes will be
            inserted to the start of the result array.
        **dim_kwargs : int or sequence or ndarray
            The keywords are arbitrary dimensions being inserted and the values
            are either the lengths of the new dims (if int is given), or their
            coordinates. Note, this is an alternative to passing a dict to the
            dim kwarg and will only be used if dim is None.

        Returns
        -------
        expanded : Dataset
            This object, but with additional dimension(s).

        See Also
        --------
        DataArray.expand_dims
        """
        if dim is None:
            pass
        elif isinstance(dim, Mapping):
            # We're later going to modify dim in place; don't tamper with
            # the input
            dim = dict(dim)
        elif isinstance(dim, int):
            raise TypeError(
                "dim should be hashable or sequence of hashables or mapping"
            )
        elif isinstance(dim, str) or not isinstance(dim, Sequence):
            dim = {dim: 1}
        elif isinstance(dim, Sequence):
            if len(dim) != len(set(dim)):
                raise ValueError("dims should not contain duplicate values.")
            dim = {d: 1 for d in dim}

        dim = either_dict_or_kwargs(dim, dim_kwargs, "expand_dims")
        assert isinstance(dim, MutableMapping)

        if axis is None:
            axis = list(range(len(dim)))
        elif not isinstance(axis, Sequence):
            axis = [axis]

        if len(dim) != len(axis):
            raise ValueError("lengths of dim and axis should be identical.")
        for d in dim:
            if d in self.dims:
                raise ValueError(f"Dimension {d} already exists.")
            if d in self._variables and not utils.is_scalar(self._variables[d]):
                raise ValueError(
                    "{dim} already exists as coordinate or"
                    " variable name.".format(dim=d)
                )

        variables: dict[Hashable, Variable] = {}
        indexes: dict[Hashable, Index] = dict(self._indexes)
        coord_names = self._coord_names.copy()
        # If dim is a dict, then ensure that the values are either integers
        # or iterables.
        for k, v in dim.items():
            if hasattr(v, "__iter__"):
                # If the value for the new dimension is an iterable, then
                # save the coordinates to the variables dict, and set the
                # value within the dim dict to the length of the iterable
                # for later use.
                index = PandasIndex(v, k)
                indexes[k] = index
                variables.update(index.create_variables())
                coord_names.add(k)
                dim[k] = variables[k].size
            elif isinstance(v, int):
                pass  # Do nothing if the dimensions value is just an int
            else:
                raise TypeError(
                    "The value of new dimension {k} must be "
                    "an iterable or an int".format(k=k)
                )

        for k, v in self._variables.items():
            if k not in dim:
                if k in coord_names:  # Do not change coordinates
                    variables[k] = v
                else:
                    result_ndim = len(v.dims) + len(axis)
                    for a in axis:
                        if a < -result_ndim or result_ndim - 1 < a:
                            raise IndexError(
                                f"Axis {a} of variable {k} is out of bounds of the "
                                f"expanded dimension size {result_ndim}"
                            )

                    axis_pos = [a if a >= 0 else result_ndim + a for a in axis]
                    if len(axis_pos) != len(set(axis_pos)):
                        raise ValueError("axis should not contain duplicate values")
                    # We need to sort them to make sure `axis` equals to the
                    # axis positions of the result array.
                    zip_axis_dim = sorted(zip(axis_pos, dim.items()))

                    all_dims = list(zip(v.dims, v.shape))
                    for d, c in zip_axis_dim:
                        all_dims.insert(d, c)
                    variables[k] = v.set_dims(dict(all_dims))
            else:
                if k not in variables:
                    # If dims includes a label of a non-dimension coordinate,
                    # it will be promoted to a 1D coordinate with a single value.
                    index, index_vars = create_default_index_implicit(v.set_dims(k))
                    indexes[k] = index
                    variables.update(index_vars)

        return self._replace_with_new_dims(
            variables, coord_names=coord_names, indexes=indexes
        )
    def update(self: T_Dataset, other: CoercibleMapping) -> T_Dataset:
        """Update this dataset's variables with those from another dataset.

        Just like :py:meth:`dict.update` this is a in-place operation.
        For a non-inplace version, see :py:meth:`Dataset.merge`.

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

            It is deprecated since version 0.17 and scheduled to be removed in 0.21.

        Raises
        ------
        ValueError
            If any dimensions would have inconsistent sizes in the updated
            dataset.

        See Also
        --------
        Dataset.assign
        Dataset.merge
        """
        merge_result = dataset_update_method(self, other)
        return self._replace(inplace=True, **merge_result._asdict())
    def curvefit(
        self: T_Dataset,
        coords: str | DataArray | Iterable[str | DataArray],
        func: Callable[..., Any],
        reduce_dims: Dims = None,
        skipna: bool = True,
        p0: dict[str, Any] | None = None,
        bounds: dict[str, Any] | None = None,
        param_names: Sequence[str] | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> T_Dataset:
        """
        Curve fitting optimization for arbitrary functions.

        Wraps `scipy.optimize.curve_fit` with `apply_ufunc`.

        Parameters
        ----------
        coords : hashable, DataArray, or sequence of hashable or DataArray
            Independent coordinate(s) over which to perform the curve fitting. Must share
            at least one dimension with the calling object. When fitting multi-dimensional
            functions, supply `coords` as a sequence in the same order as arguments in
            `func`. To fit along existing dimensions of the calling object, `coords` can
            also be specified as a str or sequence of strs.
        func : callable
            User specified function in the form `f(x, *params)` which returns a numpy
            array of length `len(x)`. `params` are the fittable parameters which are optimized
            by scipy curve_fit. `x` can also be specified as a sequence containing multiple
            coordinates, e.g. `f((x0, x1), *params)`.
        reduce_dims : str, Iterable of Hashable or None, optional
            Additional dimension(s) over which to aggregate while fitting. For example,
            calling `ds.curvefit(coords='time', reduce_dims=['lat', 'lon'], ...)` will
            aggregate all lat and lon points and fit the specified function along the
            time dimension.
        skipna : bool, default: True
            Whether to skip missing values when fitting. Default is True.
        p0 : dict-like, optional
            Optional dictionary of parameter names to initial guesses passed to the
            `curve_fit` `p0` arg. If none or only some parameters are passed, the rest will
            be assigned initial values following the default scipy behavior.
        bounds : dict-like, optional
            Optional dictionary of parameter names to bounding values passed to the
            `curve_fit` `bounds` arg. If none or only some parameters are passed, the rest
            will be unbounded following the default scipy behavior.
        param_names : sequence of hashable, optional
            Sequence of names for the fittable parameters of `func`. If not supplied,
            this will be automatically determined by arguments of `func`. `param_names`
            should be manually supplied when fitting a function that takes a variable
            number of parameters.
        **kwargs : optional
            Additional keyword arguments to passed to scipy curve_fit.

        Returns
        -------
        curvefit_results : Dataset
            A single dataset which contains:

            [var]_curvefit_coefficients
                The coefficients of the best fit.
            [var]_curvefit_covariance
                The covariance matrix of the coefficient estimates.

        See Also
        --------
        Dataset.polyfit
        scipy.optimize.curve_fit
        """
        from scipy.optimize import curve_fit

        from .alignment import broadcast
        from .computation import apply_ufunc
        from .dataarray import _THIS_ARRAY, DataArray

        if p0 is None:
            p0 = {}
        if bounds is None:
            bounds = {}
        if kwargs is None:
            kwargs = {}

        reduce_dims_: list[Hashable]
        if not reduce_dims:
            reduce_dims_ = []
        elif isinstance(reduce_dims, str) or not isinstance(reduce_dims, Iterable):
            reduce_dims_ = [reduce_dims]
        else:
            reduce_dims_ = list(reduce_dims)

        if (
            isinstance(coords, str)
            or isinstance(coords, DataArray)
            or not isinstance(coords, Iterable)
        ):
            coords = [coords]
        coords_ = [self[coord] if isinstance(coord, str) else coord for coord in coords]

        # Determine whether any coords are dims on self
        for coord in coords_:
            reduce_dims_ += [c for c in self.dims if coord.equals(self[c])]
        reduce_dims_ = list(set(reduce_dims_))
        preserved_dims = list(set(self.dims) - set(reduce_dims_))
        if not reduce_dims_:
            raise ValueError(
                "No arguments to `coords` were identified as a dimension on the calling "
                "object, and no dims were supplied to `reduce_dims`. This would result "
                "in fitting on scalar data."
            )

        # Broadcast all coords with each other
        coords_ = broadcast(*coords_)
        coords_ = [
            coord.broadcast_like(self, exclude=preserved_dims) for coord in coords_
        ]

        params, func_args = _get_func_args(func, param_names)
        param_defaults, bounds_defaults = _initialize_curvefit_params(
            params, p0, bounds, func_args
        )
        n_params = len(params)
        kwargs.setdefault("p0", [param_defaults[p] for p in params])
        kwargs.setdefault(
            "bounds",
            [
                [bounds_defaults[p][0] for p in params],
                [bounds_defaults[p][1] for p in params],
            ],
        )

        def _wrapper(Y, *coords_, **kwargs):
            # Wrap curve_fit with raveled coordinates and pointwise NaN handling
            x = np.vstack([c.ravel() for c in coords_])
            y = Y.ravel()
            if skipna:
                mask = np.all([np.any(~np.isnan(x), axis=0), ~np.isnan(y)], axis=0)
                x = x[:, mask]
                y = y[mask]
                if not len(y):
                    popt = np.full([n_params], np.nan)
                    pcov = np.full([n_params, n_params], np.nan)
                    return popt, pcov
            x = np.squeeze(x)
            popt, pcov = curve_fit(func, x, y, **kwargs)
            return popt, pcov

        result = type(self)()
        for name, da in self.data_vars.items():
            if name is _THIS_ARRAY:
                name = ""
            else:
                name = f"{str(name)}_"

            popt, pcov = apply_ufunc(
                _wrapper,
                da,
                *coords_,
                vectorize=True,
                dask="parallelized",
                input_core_dims=[reduce_dims_ for d in range(len(coords_) + 1)],
                output_core_dims=[["param"], ["cov_i", "cov_j"]],
                dask_gufunc_kwargs={
                    "output_sizes": {
                        "param": n_params,
                        "cov_i": n_params,
                        "cov_j": n_params,
                    },
                },
                output_dtypes=(np.float64, np.float64),
                exclude_dims=set(reduce_dims_),
                kwargs=kwargs,
            )
            result[name + "curvefit_coefficients"] = popt
            result[name + "curvefit_covariance"] = pcov

        result = result.assign_coords(
            {"param": params, "cov_i": params, "cov_j": params}
        )
        result.attrs = self.attrs.copy()

        return result