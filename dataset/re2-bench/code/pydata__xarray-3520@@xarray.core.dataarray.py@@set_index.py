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
)
from .accessor_dt import DatetimeAccessor
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
    LevelCoordinatesSource,
    assert_coordinate_consistent,
    remap_label_indexers,
)
from .dataset import Dataset, split_indexes
from .formatting import format_item
from .indexes import Indexes, default_indexes
from .merge import PANDAS_TYPES
from .options import OPTIONS
from .utils import Default, ReprObject, _check_inplace, _default, either_dict_or_kwargs
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
        "_accessors",
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
    dt = property(DatetimeAccessor)
    __hash__ = None  # type: ignore
    str = property(StringAccessor)
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
        coord_names = set(self._coords)
        dataset = Dataset._from_vars_and_coord_names(variables, coord_names)
        return dataset
    @property
    def variable(self) -> Variable:
        """Low level interface to the Variable object for this DataArray."""
        return self._variable
    @property
    def coords(self) -> DataArrayCoordinates:
        """Dictionary-like container of coordinate arrays.
        """
        return DataArrayCoordinates(self)
    def set_index(
        self,
        indexes: Mapping[Hashable, Union[Hashable, Sequence[Hashable]]] = None,
        append: bool = False,
        inplace: bool = None,
        **indexes_kwargs: Union[Hashable, Sequence[Hashable]],
    ) -> Optional["DataArray"]:
        """Set DataArray (multi-)indexes using one or more existing
        coordinates.

        Parameters
        ----------
        indexes : {dim: index, ...}
            Mapping from names matching dimensions and values given
            by (lists of) the names of existing coordinates or variables to set
            as new (multi-)index.
        append : bool, optional
            If True, append the supplied index(es) to the existing index(es).
            Otherwise replace the existing index(es) (default).
        **indexes_kwargs: optional
            The keyword arguments form of ``indexes``.
            One of indexes or indexes_kwargs must be provided.

        Returns
        -------
        obj : DataArray
            Another DataArray, with this data but replaced coordinates.

        Examples
        --------
        >>> arr = xr.DataArray(data=np.ones((2, 3)),
        ...                    dims=['x', 'y'],
        ...                    coords={'x':
        ...                        range(2), 'y':
        ...                        range(3), 'a': ('x', [3, 4])
        ...                    })
        >>> arr
        <xarray.DataArray (x: 2, y: 3)>
        array([[1., 1., 1.],
               [1., 1., 1.]])
        Coordinates:
          * x        (x) int64 0 1
          * y        (y) int64 0 1 2
            a        (x) int64 3 4
        >>> arr.set_index(x='a')
        <xarray.DataArray (x: 2, y: 3)>
        array([[1., 1., 1.],
               [1., 1., 1.]])
        Coordinates:
          * x        (x) int64 3 4
          * y        (y) int64 0 1 2

        See Also
        --------
        DataArray.reset_index
        """
        ds = self._to_temp_dataset().set_index(
            indexes, append=append, inplace=inplace, **indexes_kwargs
        )
        return self._from_temp_dataset(ds)