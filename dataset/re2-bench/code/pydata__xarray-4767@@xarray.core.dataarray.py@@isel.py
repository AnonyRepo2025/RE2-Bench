import datetime
import functools
import warnings
from numbers import Number
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
)
import numpy as np
import pandas as pd
from ..plot.plot import _PlotMethods
from . import (
    computation,
    dtypes,
    groupby,
    indexing,
    ops,
    pdcompat,
    resample,
    rolling,
    utils,
    weighted,
)
from .accessor_dt import CombinedDatetimelikeAccessor
from .accessor_str import StringAccessor
from .alignment import (
    _broadcast_helper,
    _get_broadcast_dims_map_common_coords,
    align,
    reindex_like_indexers,
)
from .common import AbstractArray, DataWithCoords
from .coordinates import (
    DataArrayCoordinates,
    assert_coordinate_consistent,
    remap_label_indexers,
)
from .dataset import Dataset, split_indexes
from .formatting import format_item
from .indexes import Indexes, default_indexes, propagate_indexes
from .indexing import is_fancy_indexer
from .merge import PANDAS_TYPES, MergeError, _extract_indexes_from_coords
from .options import OPTIONS, _get_keep_attrs
from .utils import (
    Default,
    HybridMappingProxy,
    ReprObject,
    _default,
    either_dict_or_kwargs,
)
from .variable import (
    IndexVariable,
    Variable,
    as_compatible_data,
    as_variable,
    assert_unique_multiindex_level_names,
)
from dask.delayed import Delayed
from cdms2 import Variable as cdms2_Variable
from iris.cube import Cube as iris_Cube
from .dataset import _get_virtual_variable
from dask.base import normalize_token
from .missing import interp_na
from .missing import ffill
from .missing import bfill
from ..backends.api import DATAARRAY_NAME, DATAARRAY_VARIABLE
from ..convert import to_cdms2
from ..convert import from_cdms2
from ..convert import to_iris
from ..convert import from_iris
from .parallel import map_blocks

_THIS_ARRAY = ReprObject("<this-array>")

class DataArray(AbstractArray, DataWithCoords):
    __slots__ = (
        "_cache",
        "_coords",
        "_file_obj",
        "_indexes",
        "_name",
        "_variable",
        "__weakref__",
    )
    _groupby_cls = groupby.DataArrayGroupBy
    _rolling_cls = rolling.DataArrayRolling
    _coarsen_cls = rolling.DataArrayCoarsen
    _resample_cls = resample.DataArrayResample
    _weighted_cls = weighted.DataArrayWeighted
    dt = utils.UncachedAccessor(CombinedDatetimelikeAccessor)
    __hash__ = None  # type: ignore
    plot = utils.UncachedAccessor(_PlotMethods)
    str = utils.UncachedAccessor(StringAccessor)
    def __init__(
        self,
        data: Any = dtypes.NA,
        coords: Union[Sequence[Tuple], Mapping[Hashable, Any], None] = None,
        dims: Union[Hashable, Sequence[Hashable], None] = None,
        name: Hashable = None,
        attrs: Mapping = None,
        # internal parameters
        indexes: Dict[Hashable, pd.Index] = None,
        fastpath: bool = False,
    ):
        if fastpath:
            variable = data
            assert dims is None
            assert attrs is None
        else:
            # try to fill in arguments from data if they weren't supplied
            if coords is None:

                if isinstance(data, DataArray):
                    coords = data.coords
                elif isinstance(data, pd.Series):
                    coords = [data.index]
                elif isinstance(data, pd.DataFrame):
                    coords = [data.index, data.columns]
                elif isinstance(data, (pd.Index, IndexVariable)):
                    coords = [data]
                elif isinstance(data, pdcompat.Panel):
                    coords = [data.items, data.major_axis, data.minor_axis]

            if dims is None:
                dims = getattr(data, "dims", getattr(coords, "dims", None))
            if name is None:
                name = getattr(data, "name", None)
            if attrs is None and not isinstance(data, PANDAS_TYPES):
                attrs = getattr(data, "attrs", None)

            data = _check_data_shape(data, coords, dims)
            data = as_compatible_data(data)
            coords, dims = _infer_coords_and_dims(data.shape, coords, dims)
            variable = Variable(dims, data, attrs, fastpath=True)
            indexes = dict(
                _extract_indexes_from_coords(coords)
            )  # needed for to_dataset

        # These fully describe a DataArray
        self._variable = variable
        assert isinstance(coords, dict)
        self._coords = coords
        self._name = name

        # TODO(shoyer): document this argument, once it becomes part of the
        # public interface.
        self._indexes = indexes

        self._file_obj = None
    def _to_temp_dataset(self) -> Dataset:
        return self._to_dataset_whole(name=_THIS_ARRAY, shallow_copy=False)
    def _to_dataset_whole(
        self, name: Hashable = None, shallow_copy: bool = True
    ) -> Dataset:
        if name is None:
            name = self.name
        if name is None:
            raise ValueError(
                "unable to convert unnamed DataArray to a "
                "Dataset without providing an explicit name"
            )
        if name in self.coords:
            raise ValueError(
                "cannot create a Dataset from a DataArray with "
                "the same name as one of its coordinates"
            )
        # use private APIs for speed: this is called by _to_temp_dataset(),
        # which is used in the guts of a lot of operations (e.g., reindex)
        variables = self._coords.copy()
        variables[name] = self.variable
        if shallow_copy:
            for k in variables:
                variables[k] = variables[k].copy(deep=False)
        indexes = self._indexes

        coord_names = set(self._coords)
        dataset = Dataset._construct_direct(variables, coord_names, indexes=indexes)
        return dataset
    @property
    def variable(self) -> Variable:
        """Low level interface to the Variable object for this DataArray."""
        return self._variable
    @property
    def dtype(self) -> np.dtype:
        return self.variable.dtype
    @dims.setter
    def dims(self, value):
        raise AttributeError(
            "you cannot assign dims on a DataArray. Use "
            ".rename() or .swap_dims() instead."
        )
    @property
    def indexes(self) -> Indexes:
        """Mapping of pandas.Index objects used for label based indexing"""
        if self._indexes is None:
            self._indexes = default_indexes(self._coords, self.dims)
        return Indexes(self._indexes)
    @property
    def coords(self) -> DataArrayCoordinates:
        """Dictionary-like container of coordinate arrays."""
        return DataArrayCoordinates(self)
    def isel(
        self,
        indexers: Mapping[Hashable, Any] = None,
        drop: bool = False,
        missing_dims: str = "raise",
        **indexers_kwargs: Any,
    ) -> "DataArray":
        """Return a new DataArray whose data is given by integer indexing
        along the specified dimension(s).

        Parameters
        ----------
        indexers : dict, optional
            A dict with keys matching dimensions and values given
            by integers, slice objects or arrays.
            indexer can be a integer, slice, array-like or DataArray.
            If DataArrays are passed as indexers, xarray-style indexing will be
            carried out. See :ref:`indexing` for the details.
            One of indexers or indexers_kwargs must be provided.
        drop : bool, optional
            If ``drop=True``, drop coordinates variables indexed by integers
            instead of making them scalar.
        missing_dims : {"raise", "warn", "ignore"}, default: "raise"
            What to do if dimensions that should be selected from are not present in the
            DataArray:
            - "raise": raise an exception
            - "warning": raise a warning, and ignore the missing dimensions
            - "ignore": ignore the missing dimensions
        **indexers_kwargs : {dim: indexer, ...}, optional
            The keyword arguments form of ``indexers``.

        See Also
        --------
        Dataset.isel
        DataArray.sel
        """

        indexers = either_dict_or_kwargs(indexers, indexers_kwargs, "isel")

        if any(is_fancy_indexer(idx) for idx in indexers.values()):
            ds = self._to_temp_dataset()._isel_fancy(
                indexers, drop=drop, missing_dims=missing_dims
            )
            return self._from_temp_dataset(ds)

        # Much faster algorithm for when all indexers are ints, slices, one-dimensional
        # lists, or zero or one-dimensional np.ndarray's

        variable = self._variable.isel(indexers, missing_dims=missing_dims)

        coords = {}
        for coord_name, coord_value in self._coords.items():
            coord_indexers = {
                k: v for k, v in indexers.items() if k in coord_value.dims
            }
            if coord_indexers:
                coord_value = coord_value.isel(coord_indexers)
                if drop and coord_value.ndim == 0:
                    continue
            coords[coord_name] = coord_value

        return self._replace(variable=variable, coords=coords)