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
    def _replace(
        self,
        variable: Variable = None,
        coords=None,
        name: Union[Hashable, None, Default] = _default,
        indexes=None,
    ) -> "DataArray":
        if variable is None:
            variable = self.variable
        if coords is None:
            coords = self._coords
        if name is _default:
            name = self.name
        return type(self)(variable, coords, name=name, fastpath=True, indexes=indexes)
    def _replace_maybe_drop_dims(
        self, variable: Variable, name: Union[Hashable, None, Default] = _default
    ) -> "DataArray":
        if variable.dims == self.dims and variable.shape == self.shape:
            coords = self._coords.copy()
            indexes = self._indexes
        elif variable.dims == self.dims:
            # Shape has changed (e.g. from reduce(..., keepdims=True)
            new_sizes = dict(zip(self.dims, variable.shape))
            coords = {
                k: v
                for k, v in self._coords.items()
                if v.shape == tuple(new_sizes[d] for d in v.dims)
            }
            changed_dims = [
                k for k in variable.dims if variable.sizes[k] != self.sizes[k]
            ]
            indexes = propagate_indexes(self._indexes, exclude=changed_dims)
        else:
            allowed_dims = set(variable.dims)
            coords = {
                k: v for k, v in self._coords.items() if set(v.dims) <= allowed_dims
            }
            indexes = propagate_indexes(
                self._indexes, exclude=(set(self.dims) - allowed_dims)
            )
        return self._replace(variable, coords, name, indexes=indexes)
    @property
    def variable(self) -> Variable:
        """Low level interface to the Variable object for this DataArray."""
        return self._variable
    @property
    def shape(self) -> Tuple[int, ...]:
        return self.variable.shape
    @dims.setter
    def dims(self, value):
        raise AttributeError(
            "you cannot assign dims on a DataArray. Use "
            ".rename() or .swap_dims() instead."
        )
    def _getitem_coord(self, key):
        from .dataset import _get_virtual_variable

        try:
            var = self._coords[key]
        except KeyError:
            dim_sizes = dict(zip(self.dims, self.shape))
            _, key, var = _get_virtual_variable(
                self._coords, key, self._level_coords, dim_sizes
            )

        return self._replace_maybe_drop_dims(var, name=key)
    @property
    def coords(self) -> DataArrayCoordinates:
        """Dictionary-like container of coordinate arrays."""
        return DataArrayCoordinates(self)
    def transpose(
        self,
        *dims: Hashable,
        transpose_coords: bool = True,
        missing_dims: str = "raise",
    ) -> "DataArray":
        """Return a new DataArray object with transposed dimensions.

        Parameters
        ----------
        *dims : hashable, optional
            By default, reverse the dimensions. Otherwise, reorder the
            dimensions to this order.
        transpose_coords : bool, default: True
            If True, also transpose the coordinates of this DataArray.
        missing_dims : {"raise", "warn", "ignore"}, default: "raise"
            What to do if dimensions that should be selected from are not present in the
            DataArray:
            - "raise": raise an exception
            - "warning": raise a warning, and ignore the missing dimensions
            - "ignore": ignore the missing dimensions

        Returns
        -------
        transposed : DataArray
            The returned DataArray's array is transposed.

        Notes
        -----
        This operation returns a view of this array's data. It is
        lazy for dask-backed DataArrays but not for numpy-backed DataArrays
        -- the data will be fully loaded.

        See Also
        --------
        numpy.transpose
        Dataset.transpose
        """
        if dims:
            dims = tuple(utils.infix_dims(dims, self.dims, missing_dims))
        variable = self.variable.transpose(*dims)
        if transpose_coords:
            coords: Dict[Hashable, Variable] = {}
            for name, coord in self.coords.items():
                coord_dims = tuple(dim for dim in dims if dim in coord.dims)
                coords[name] = coord.variable.transpose(*coord_dims)
            return self._replace(variable, coords)
        else:
            return self._replace(variable)