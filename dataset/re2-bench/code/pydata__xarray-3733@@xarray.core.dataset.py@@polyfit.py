import copy
import datetime
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
    weighted,
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
    _weighted_cls = weighted.DatasetWeighted
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
        >>> ds = xr.Dataset(
        ...     {"foo": da, "bar": ("x", [-1, 2])}, coords={"x": ["one", "two"]},
        ... )
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
        >>> ds_0["foo"][0, 0] = 7
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

        >>> ds.copy(
        ...     data={"foo": np.arange(6).reshape(2, 3), "bar": ["a", "b"]}
        ... )
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
    def _stack_once(self, dims, new_dim):
        if ... in dims:
            dims = list(infix_dims(dims, self.dims))
        variables = {}
        for name, var in self.variables.items():
            if name not in dims:
                if any(d in var.dims for d in dims):
                    add_dims = [d for d in dims if d not in var.dims]
                    vdims = list(var.dims) + add_dims
                    shape = [self.dims[d] for d in vdims]
                    exp_var = var.set_dims(vdims, shape)
                    stacked_var = exp_var.stack(**{new_dim: dims})
                    variables[name] = stacked_var
                else:
                    variables[name] = var.copy(deep=False)

        # consider dropping levels that are unused?
        levels = [self.get_index(dim) for dim in dims]
        idx = utils.multiindex_from_product_levels(levels, names=dims)
        variables[new_dim] = IndexVariable(new_dim, idx)

        coord_names = set(self._coord_names) - set(dims) | {new_dim}

        indexes = {k: v for k, v in self.indexes.items() if k not in dims}
        indexes[new_dim] = idx

        return self._replace_with_new_dims(
            variables, coord_names=coord_names, indexes=indexes
        )
    def stack(
        self,
        dimensions: Mapping[Hashable, Sequence[Hashable]] = None,
        **dimensions_kwargs: Sequence[Hashable],
    ) -> "Dataset":
        """
        Stack any number of existing dimensions into a single new dimension.

        New dimensions will be added at the end, and the corresponding
        coordinate variables will be combined into a MultiIndex.

        Parameters
        ----------
        dimensions : Mapping of the form new_name=(dim1, dim2, ...)
            Names of new dimensions, and the existing dimensions that they
            replace. An ellipsis (`...`) will be replaced by all unlisted dimensions.
            Passing a list containing an ellipsis (`stacked_dim=[...]`) will stack over
            all dimensions.
        **dimensions_kwargs:
            The keyword arguments form of ``dimensions``.
            One of dimensions or dimensions_kwargs must be provided.

        Returns
        -------
        stacked : Dataset
            Dataset with stacked data.

        See also
        --------
        Dataset.unstack
        """
        dimensions = either_dict_or_kwargs(dimensions, dimensions_kwargs, "stack")
        result = self
        for new_dim, dims in dimensions.items():
            result = result._stack_once(dims, new_dim)
        return result
    def _unstack_once(self, dim: Hashable, fill_value, sparse) -> "Dataset":
        index = self.get_index(dim)
        index = remove_unused_levels_categories(index)
        full_idx = pd.MultiIndex.from_product(index.levels, names=index.names)

        # take a shortcut in case the MultiIndex was not modified.
        if index.equals(full_idx):
            obj = self
        else:
            obj = self._reindex(
                {dim: full_idx}, copy=False, fill_value=fill_value, sparse=sparse
            )

        new_dim_names = index.names
        new_dim_sizes = [lev.size for lev in index.levels]

        variables: Dict[Hashable, Variable] = {}
        indexes = {k: v for k, v in self.indexes.items() if k != dim}

        for name, var in obj.variables.items():
            if name != dim:
                if dim in var.dims:
                    new_dims = dict(zip(new_dim_names, new_dim_sizes))
                    variables[name] = var.unstack({dim: new_dims})
                else:
                    variables[name] = var

        for name, lev in zip(new_dim_names, index.levels):
            variables[name] = IndexVariable(name, lev)
            indexes[name] = lev

        coord_names = set(self._coord_names) - {dim} | set(new_dim_names)

        return self._replace_with_new_dims(
            variables, coord_names=coord_names, indexes=indexes
        )
    def unstack(
        self,
        dim: Union[Hashable, Iterable[Hashable]] = None,
        fill_value: Any = dtypes.NA,
        sparse: bool = False,
    ) -> "Dataset":
        """
        Unstack existing dimensions corresponding to MultiIndexes into
        multiple new dimensions.

        New dimensions will be added at the end.

        Parameters
        ----------
        dim : Hashable or iterable of Hashable, optional
            Dimension(s) over which to unstack. By default unstacks all
            MultiIndexes.
        fill_value: value to be filled. By default, np.nan
        sparse: use sparse-array if True

        Returns
        -------
        unstacked : Dataset
            Dataset with unstacked data.

        See also
        --------
        Dataset.stack
        """
        if dim is None:
            dims = [
                d for d in self.dims if isinstance(self.get_index(d), pd.MultiIndex)
            ]
        else:
            if isinstance(dim, str) or not isinstance(dim, Iterable):
                dims = [dim]
            else:
                dims = list(dim)

            missing_dims = [d for d in dims if d not in self.dims]
            if missing_dims:
                raise ValueError(
                    "Dataset does not contain the dimensions: %s" % missing_dims
                )

            non_multi_dims = [
                d for d in dims if not isinstance(self.get_index(d), pd.MultiIndex)
            ]
            if non_multi_dims:
                raise ValueError(
                    "cannot unstack dimensions that do not "
                    "have a MultiIndex: %s" % non_multi_dims
                )

        result = self.copy(deep=False)
        for dim in dims:
            result = result._unstack_once(dim, fill_value, sparse)
        return result
    def polyfit(
        self,
        dim: Hashable,
        deg: int,
        skipna: bool = None,
        rcond: float = None,
        w: Union[Hashable, Any] = None,
        full: bool = False,
        cov: Union[bool, str] = False,
    ):
        """
        Least squares polynomial fit.

        This replicates the behaviour of `numpy.polyfit` but differs by skipping
        invalid values when `skipna = True`.

        Parameters
        ----------
        dim : hashable
            Coordinate along which to fit the polynomials.
        deg : int
            Degree of the fitting polynomial.
        skipna : bool, optional
            If True, removes all invalid values before fitting each 1D slices of the array.
            Default is True if data is stored in a dask.array or if there is any
            invalid values, False otherwise.
        rcond : float, optional
            Relative condition number to the fit.
        w : Union[Hashable, Any], optional
            Weights to apply to the y-coordinate of the sample points.
            Can be an array-like object or the name of a coordinate in the dataset.
        full : bool, optional
            Whether to return the residuals, matrix rank and singular values in addition
            to the coefficients.
        cov : Union[bool, str], optional
            Whether to return to the covariance matrix in addition to the coefficients.
            The matrix is not scaled if `cov='unscaled'`.


        Returns
        -------
        polyfit_results : Dataset
            A single dataset which contains (for each "var" in the input dataset):

            [var]_polyfit_coefficients
                The coefficients of the best fit for each variable in this dataset.
            [var]_polyfit_residuals
                The residuals of the least-square computation for each variable (only included if `full=True`)
            [dim]_matrix_rank
                The effective rank of the scaled Vandermonde coefficient matrix (only included if `full=True`)
            [dim]_singular_values
                The singular values of the scaled Vandermonde coefficient matrix (only included if `full=True`)
            [var]_polyfit_covariance
                The covariance matrix of the polynomial coefficient estimates (only included if `full=False` and `cov=True`)

        See also
        --------
        numpy.polyfit
        """
        variables = {}
        skipna_da = skipna

        x = get_clean_interp_index(self, dim)
        xname = "{}_".format(self[dim].name)
        order = int(deg) + 1
        lhs = np.vander(x, order)

        if rcond is None:
            rcond = x.shape[0] * np.core.finfo(x.dtype).eps

        # Weights:
        if w is not None:
            if isinstance(w, Hashable):
                w = self.coords[w]
            w = np.asarray(w)
            if w.ndim != 1:
                raise TypeError("Expected a 1-d array for weights.")
            if w.shape[0] != lhs.shape[0]:
                raise TypeError("Expected w and {} to have the same length".format(dim))
            lhs *= w[:, np.newaxis]

        # Scaling
        scale = np.sqrt((lhs * lhs).sum(axis=0))
        lhs /= scale

        degree_dim = utils.get_temp_dimname(self.dims, "degree")

        rank = np.linalg.matrix_rank(lhs)
        if rank != order and not full:
            warnings.warn(
                "Polyfit may be poorly conditioned", np.RankWarning, stacklevel=4
            )

        if full:
            rank = xr.DataArray(rank, name=xname + "matrix_rank")
            variables[rank.name] = rank
            sing = np.linalg.svd(lhs, compute_uv=False)
            sing = xr.DataArray(
                sing,
                dims=(degree_dim,),
                coords={degree_dim: np.arange(order)[::-1]},
                name=xname + "singular_values",
            )
            variables[sing.name] = sing

        for name, da in self.data_vars.items():
            if dim not in da.dims:
                continue

            if skipna is None:
                if isinstance(da.data, dask_array_type):
                    skipna_da = True
                else:
                    skipna_da = np.any(da.isnull())

            dims_to_stack = [dimname for dimname in da.dims if dimname != dim]
            stacked_coords = {}
            if dims_to_stack:
                stacked_dim = utils.get_temp_dimname(dims_to_stack, "stacked")
                rhs = da.transpose(dim, *dims_to_stack).stack(
                    {stacked_dim: dims_to_stack}
                )
                stacked_coords = {stacked_dim: rhs[stacked_dim]}
                scale_da = scale[:, np.newaxis]
            else:
                rhs = da
                scale_da = scale

            if w is not None:
                rhs *= w[:, np.newaxis]

            coeffs, residuals = duck_array_ops.least_squares(
                lhs, rhs.data, rcond=rcond, skipna=skipna_da
            )

            if isinstance(name, str):
                name = "{}_".format(name)
            else:
                # Thus a ReprObject => polyfit was called on a DataArray
                name = ""

            coeffs = xr.DataArray(
                coeffs / scale_da,
                dims=[degree_dim] + list(stacked_coords.keys()),
                coords={degree_dim: np.arange(order)[::-1], **stacked_coords},
                name=name + "polyfit_coefficients",
            )
            if dims_to_stack:
                coeffs = coeffs.unstack(stacked_dim)
            variables[coeffs.name] = coeffs

            if full or (cov is True):
                residuals = xr.DataArray(
                    residuals if dims_to_stack else residuals.squeeze(),
                    dims=list(stacked_coords.keys()),
                    coords=stacked_coords,
                    name=name + "polyfit_residuals",
                )
                if dims_to_stack:
                    residuals = residuals.unstack(stacked_dim)
                variables[residuals.name] = residuals

            if cov:
                Vbase = np.linalg.inv(np.dot(lhs.T, lhs))
                Vbase /= np.outer(scale, scale)
                if cov == "unscaled":
                    fac = 1
                else:
                    if x.shape[0] <= order:
                        raise ValueError(
                            "The number of data points must exceed order to scale the covariance matrix."
                        )
                    fac = residuals / (x.shape[0] - order)
                covariance = xr.DataArray(Vbase, dims=("cov_i", "cov_j"),) * fac
                variables[name + "polyfit_covariance"] = covariance

        return Dataset(data_vars=variables, attrs=self.attrs.copy())