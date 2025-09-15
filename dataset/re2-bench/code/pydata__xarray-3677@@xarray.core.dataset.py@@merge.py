import copy
import functools
import sys
import warnings
from collections import defaultdict
from html import escape
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
    TypeVar,
    Union,
    cast,
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
)
from .alignment import _broadcast_helper, _get_broadcast_dims_map_common_coords, align
from .common import (
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
from .indexes import (
    Indexes,
    default_indexes,
    isel_variable_and_index,
    propagate_indexes,
    roll_index,
)
from .indexing import is_fancy_indexer
from .merge import (
    dataset_merge_method,
    dataset_update_method,
    merge_coordinates_without_align,
    merge_data_and_coords,
)
from .options import OPTIONS, _get_keep_attrs
from .pycompat import dask_array_type
from .utils import (
    Default,
    Frozen,
    SortedKeysDict,
    _check_inplace,
    _default,
    decode_numpy_dict_values,
    either_dict_or_kwargs,
    hashable,
    is_dict_like,
    is_list_like,
    is_scalar,
    maybe_wrap_array,
)
from .variable import (
    IndexVariable,
    Variable,
    as_variable,
    broadcast_variables,
    assert_unique_multiindex_level_names,
)
from ..backends import AbstractDataStore, ZarrStore
from .dataarray import DataArray
from .merge import CoercibleMapping
from dask.delayed import Delayed
from dask.base import normalize_token
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
from dask.base import tokenize
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
import dask.array
from .parallel import map_blocks
import dask.array as da
import dask
import itertools
from .dataarray import DataArray
from .dataarray import DataArray
from dask.highlevelgraph import HighLevelGraph
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
        "_attrs",
        "_cache",
        "_coord_names",
        "_dims",
        "_encoding",
        "_file_obj",
        "_indexes",
        "_variables",
        "__weakref__",
    )
    _groupby_cls = groupby.DatasetGroupBy
    _rolling_cls = rolling.DatasetRolling
    _coarsen_cls = rolling.DatasetCoarsen
    _resample_cls = resample.DatasetResample
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

        See also
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
        file_obj=None,
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
        obj._file_obj = file_obj
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
                self._attrs = attrs  # type: ignore # FIXME need mypy 0.750
            if indexes is not _default:
                self._indexes = indexes  # type: ignore # FIXME need mypy 0.750
            if encoding is not _default:
                self._encoding = encoding  # type: ignore # FIXME need mypy 0.750
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
    def _replace_with_new_dims(
        self,
        variables: Dict[Hashable, Variable],
        coord_names: set = None,
        attrs: Union[Dict[Hashable, Any], None, Default] = _default,
        indexes: Union[Dict[Hashable, pd.Index], None, Default] = _default,
        inplace: bool = False,
    ) -> "Dataset":
        """Replace variables with recalculated dimensions."""
        dims = calculate_dimensions(variables)
        return self._replace(
            variables, coord_names, dims, attrs, indexes, inplace=inplace
        )
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
        >>> ds = xr.Dataset({'foo': da, 'bar': ('x', [-1, 2])},
                            coords={'x': ['one', 'two']})
        >>> ds.copy()
        <xarray.Dataset>
        Dimensions:  (dim_0: 2, dim_1: 3, x: 2)
        Coordinates:
        * x        (x) <U3 'one' 'two'
        Dimensions without coordinates: dim_0, dim_1
        Data variables:
            foo      (dim_0, dim_1) float64 -0.8079 0.3897 -1.862 -0.6091 -1.051 -0.3003
            bar      (x) int64 -1 2
        >>> ds_0 = ds.copy(deep=False)
        >>> ds_0['foo'][0, 0] = 7
        >>> ds_0
        <xarray.Dataset>
        Dimensions:  (dim_0: 2, dim_1: 3, x: 2)
        Coordinates:
        * x        (x) <U3 'one' 'two'
        Dimensions without coordinates: dim_0, dim_1
        Data variables:
            foo      (dim_0, dim_1) float64 7.0 0.3897 -1.862 -0.6091 -1.051 -0.3003
            bar      (x) int64 -1 2
        >>> ds
        <xarray.Dataset>
        Dimensions:  (dim_0: 2, dim_1: 3, x: 2)
        Coordinates:
        * x        (x) <U3 'one' 'two'
        Dimensions without coordinates: dim_0, dim_1
        Data variables:
            foo      (dim_0, dim_1) float64 7.0 0.3897 -1.862 -0.6091 -1.051 -0.3003
            bar      (x) int64 -1 2

        Changing the data using the ``data`` argument maintains the
        structure of the original object, but with the new data. Original
        object is unaffected.

        >>> ds.copy(data={'foo': np.arange(6).reshape(2, 3), 'bar': ['a', 'b']})
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
            foo      (dim_0, dim_1) float64 7.0 0.3897 -1.862 -0.6091 -1.051 -0.3003
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
    def _construct_dataarray(self, name: Hashable) -> "DataArray":
        """Construct a DataArray by indexing this dataset
        """
        from .dataarray import DataArray

        try:
            variable = self._variables[name]
        except KeyError:
            _, name, variable = _get_virtual_variable(
                self._variables, name, self._level_coords, self.dims
            )

        needed_dims = set(variable.dims)

        coords: Dict[Hashable, Variable] = {}
        for k in self.coords:
            if set(self.variables[k].dims) <= needed_dims:
                coords[k] = self.variables[k]

        if self._indexes is None:
            indexes = None
        else:
            indexes = {k: v for k, v in self._indexes.items() if k in coords}

        return DataArray(variable, coords, name=name, indexes=indexes, fastpath=True)
    def __iter__(self) -> Iterator[Hashable]:
        return iter(self.data_vars)
    def __getitem__(self, key: Any) -> "Union[DataArray, Dataset]":
        """Access variables or coordinates this dataset as a
        :py:class:`~xarray.DataArray`.

        Indexing with a list of names will return a new ``Dataset`` object.
        """
        # TODO(shoyer): type this properly: https://github.com/python/mypy/issues/7328
        if utils.is_dict_like(key):
            return self.isel(**cast(Mapping, key))

        if hashable(key):
            return self._construct_dataarray(key)
        else:
            return self._copy_listed(np.asarray(key))
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
    @property
    def data_vars(self) -> DataVariables:
        """Dictionary of DataArray objects corresponding to data variables
        """
        return DataVariables(self)
    def reindex(
        self,
        indexers: Mapping[Hashable, Any] = None,
        method: str = None,
        tolerance: Number = None,
        copy: bool = True,
        fill_value: Any = dtypes.NA,
        **indexers_kwargs: Any,
    ) -> "Dataset":
        """Conform this object onto a new set of indexes, filling in
        missing values with ``fill_value``. The default fill value is NaN.

        Parameters
        ----------
        indexers : dict. optional
            Dictionary with keys given by dimension names and values given by
            arrays of coordinates tick labels. Any mis-matched coordinate
            values will be filled in with NaN, and any mis-matched dimension
            names will simply be ignored.
            One of indexers or indexers_kwargs must be provided.
        method : {None, 'nearest', 'pad'/'ffill', 'backfill'/'bfill'}, optional
            Method to use for filling index values in ``indexers`` not found in
            this dataset:

            * None (default): don't fill gaps
            * pad / ffill: propagate last valid index value forward
            * backfill / bfill: propagate next valid index value backward
            * nearest: use nearest valid index value
        tolerance : optional
            Maximum distance between original and new labels for inexact
            matches. The values of the index at the matching locations must
            satisfy the equation ``abs(index[indexer] - target) <= tolerance``.
        copy : bool, optional
            If ``copy=True``, data in the return value is always copied. If
            ``copy=False`` and reindexing is unnecessary, or can be performed
            with only slice operations, then the output may share memory with
            the input. In either case, a new xarray object is always returned.
        fill_value : scalar, optional
            Value to use for newly missing values
        sparse: use sparse-array. By default, False
        **indexers_kwargs : {dim: indexer, ...}, optional
            Keyword arguments in the same form as ``indexers``.
            One of indexers or indexers_kwargs must be provided.

        Returns
        -------
        reindexed : Dataset
            Another dataset, with this dataset's data but replaced coordinates.

        See Also
        --------
        Dataset.reindex_like
        align
        pandas.Index.get_indexer

        Examples
        --------

        Create a dataset with some fictional data.

        >>> import xarray as xr
        >>> import pandas as pd
        >>> x = xr.Dataset(
        ...     {
        ...         "temperature": ("station", 20 * np.random.rand(4)),
        ...         "pressure": ("station", 500 * np.random.rand(4))
        ...     },
        ...     coords={"station": ["boston", "nyc", "seattle", "denver"]})
        >>> x
        <xarray.Dataset>
        Dimensions:      (station: 4)
        Coordinates:
        * station      (station) <U7 'boston' 'nyc' 'seattle' 'denver'
        Data variables:
            temperature  (station) float64 18.84 14.59 19.22 17.16
            pressure     (station) float64 324.1 194.3 122.8 244.3
        >>> x.indexes
        station: Index(['boston', 'nyc', 'seattle', 'denver'], dtype='object', name='station')

        Create a new index and reindex the dataset. By default values in the new index that
        do not have corresponding records in the dataset are assigned `NaN`.

        >>> new_index = ['boston', 'austin', 'seattle', 'lincoln']
        >>> x.reindex({'station': new_index})
        <xarray.Dataset>
        Dimensions:      (station: 4)
        Coordinates:
        * station      (station) object 'boston' 'austin' 'seattle' 'lincoln'
        Data variables:
            temperature  (station) float64 18.84 nan 19.22 nan
            pressure     (station) float64 324.1 nan 122.8 nan

        We can fill in the missing values by passing a value to the keyword `fill_value`.

        >>> x.reindex({'station': new_index}, fill_value=0)
        <xarray.Dataset>
        Dimensions:      (station: 4)
        Coordinates:
        * station      (station) object 'boston' 'austin' 'seattle' 'lincoln'
        Data variables:
            temperature  (station) float64 18.84 0.0 19.22 0.0
            pressure     (station) float64 324.1 0.0 122.8 0.0

        Because the index is not monotonically increasing or decreasing, we cannot use arguments
        to the keyword method to fill the `NaN` values.

        >>> x.reindex({'station': new_index}, method='nearest')
        Traceback (most recent call last):
        ...
            raise ValueError('index must be monotonic increasing or decreasing')
        ValueError: index must be monotonic increasing or decreasing

        To further illustrate the filling functionality in reindex, we will create a
        dataset with a monotonically increasing index (for example, a sequence of dates).

        >>> x2 = xr.Dataset(
        ...     {
        ...         "temperature": ("time", [15.57, 12.77, np.nan, 0.3081, 16.59, 15.12]),
        ...         "pressure": ("time", 500 * np.random.rand(6))
        ...     },
        ...     coords={"time": pd.date_range('01/01/2019', periods=6, freq='D')})
        >>> x2
        <xarray.Dataset>
        Dimensions:      (time: 6)
        Coordinates:
        * time         (time) datetime64[ns] 2019-01-01 2019-01-02 ... 2019-01-06
        Data variables:
            temperature  (time) float64 15.57 12.77 nan 0.3081 16.59 15.12
            pressure     (time) float64 103.4 122.7 452.0 444.0 399.2 486.0

        Suppose we decide to expand the dataset to cover a wider date range.

        >>> time_index2 = pd.date_range('12/29/2018', periods=10, freq='D')
        >>> x2.reindex({'time': time_index2})
        <xarray.Dataset>
        Dimensions:      (time: 10)
        Coordinates:
        * time         (time) datetime64[ns] 2018-12-29 2018-12-30 ... 2019-01-07
        Data variables:
            temperature  (time) float64 nan nan nan 15.57 ... 0.3081 16.59 15.12 nan
            pressure     (time) float64 nan nan nan 103.4 ... 444.0 399.2 486.0 nan

        The index entries that did not have a value in the original data frame (for example, `2018-12-29`)
        are by default filled with NaN. If desired, we can fill in the missing values using one of several options.

        For example, to back-propagate the last valid value to fill the `NaN` values,
        pass `bfill` as an argument to the `method` keyword.

        >>> x3 = x2.reindex({'time': time_index2}, method='bfill')
        >>> x3
        <xarray.Dataset>
        Dimensions:      (time: 10)
        Coordinates:
        * time         (time) datetime64[ns] 2018-12-29 2018-12-30 ... 2019-01-07
        Data variables:
            temperature  (time) float64 15.57 15.57 15.57 15.57 ... 16.59 15.12 nan
            pressure     (time) float64 103.4 103.4 103.4 103.4 ... 399.2 486.0 nan

        Please note that the `NaN` value present in the original dataset (at index value `2019-01-03`)
        will not be filled by any of the value propagation schemes.

        >>> x2.where(x2.temperature.isnull(), drop=True)
        <xarray.Dataset>
        Dimensions:      (time: 1)
        Coordinates:
        * time         (time) datetime64[ns] 2019-01-03
        Data variables:
            temperature  (time) float64 nan
            pressure     (time) float64 452.0
        >>> x3.where(x3.temperature.isnull(), drop=True)
        <xarray.Dataset>
        Dimensions:      (time: 2)
        Coordinates:
        * time         (time) datetime64[ns] 2019-01-03 2019-01-07
        Data variables:
            temperature  (time) float64 nan nan
            pressure     (time) float64 452.0 nan

        This is because filling while reindexing does not look at dataset values, but only compares
        the original and desired indexes. If you do want to fill in the `NaN` values present in the
        original dataset, use the :py:meth:`~Dataset.fillna()` method.

        """
        return self._reindex(
            indexers,
            method,
            tolerance,
            copy,
            fill_value,
            sparse=False,
            **indexers_kwargs,
        )
    def _reindex(
        self,
        indexers: Mapping[Hashable, Any] = None,
        method: str = None,
        tolerance: Number = None,
        copy: bool = True,
        fill_value: Any = dtypes.NA,
        sparse: bool = False,
        **indexers_kwargs: Any,
    ) -> "Dataset":
        """
        same to _reindex but support sparse option
        """
        indexers = utils.either_dict_or_kwargs(indexers, indexers_kwargs, "reindex")

        bad_dims = [d for d in indexers if d not in self.dims]
        if bad_dims:
            raise ValueError("invalid reindex dimensions: %s" % bad_dims)

        variables, indexes = alignment.reindex_variables(
            self.variables,
            self.sizes,
            self.indexes,
            indexers,
            method,
            tolerance,
            copy=copy,
            fill_value=fill_value,
            sparse=sparse,
        )
        coord_names = set(self._coord_names)
        coord_names.update(indexers)
        return self._replace_with_new_dims(variables, coord_names, indexes=indexes)
    def merge(
        self,
        other: "CoercibleMapping",
        inplace: bool = None,
        overwrite_vars: Union[Hashable, Iterable[Hashable]] = frozenset(),
        compat: str = "no_conflicts",
        join: str = "outer",
        fill_value: Any = dtypes.NA,
    ) -> "Dataset":
        """Merge the arrays of two datasets into a single dataset.

        This method generally does not allow for overriding data, with the
        exception of attributes, which are ignored on the second dataset.
        Variables with the same name are checked for conflicts via the equals
        or identical methods.

        Parameters
        ----------
        other : Dataset or castable to Dataset
            Dataset or variables to merge with this dataset.
        overwrite_vars : Hashable or iterable of Hashable, optional
            If provided, update variables of these name(s) without checking for
            conflicts in this dataset.
        compat : {'broadcast_equals', 'equals', 'identical',
                  'no_conflicts'}, optional
            String indicating how to compare variables of the same name for
            potential conflicts:

            - 'broadcast_equals': all values must be equal when variables are
              broadcast against each other to ensure common dimensions.
            - 'equals': all values and dimensions must be the same.
            - 'identical': all values, dimensions and attributes must be the
              same.
            - 'no_conflicts': only values which are not null in both datasets
              must be equal. The returned dataset then contains the combination
              of all non-null values.

        join : {'outer', 'inner', 'left', 'right', 'exact'}, optional
            Method for joining ``self`` and ``other`` along shared dimensions:

            - 'outer': use the union of the indexes
            - 'inner': use the intersection of the indexes
            - 'left': use indexes from ``self``
            - 'right': use indexes from ``other``
            - 'exact': error instead of aligning non-equal indexes
        fill_value: scalar, optional
            Value to use for newly missing values

        Returns
        -------
        merged : Dataset
            Merged dataset.

        Raises
        ------
        MergeError
            If any variables conflict (see ``compat``).
        """
        _check_inplace(inplace)
        other = other.to_dataset() if isinstance(other, xr.DataArray) else other
        merge_result = dataset_merge_method(
            self,
            other,
            overwrite_vars=overwrite_vars,
            compat=compat,
            join=join,
            fill_value=fill_value,
        )
        return self._replace(**merge_result._asdict())