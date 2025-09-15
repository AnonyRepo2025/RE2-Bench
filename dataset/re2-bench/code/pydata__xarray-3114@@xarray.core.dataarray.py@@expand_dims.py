import functools
import sys
import warnings
from collections import OrderedDict
from numbers import Number
from typing import (Any, Callable, Dict, Hashable, Iterable, List, Mapping,
                    Optional, Sequence, Tuple, Union, cast)
import numpy as np
import pandas as pd
from ..plot.plot import _PlotMethods
from . import (
    computation, dtypes, groupby, indexing, ops, pdcompat, resample, rolling,
    utils)
from .accessor_dt import DatetimeAccessor
from .accessor_str import StringAccessor
from .alignment import align, reindex_like_indexers
from .common import AbstractArray, DataWithCoords
from .coordinates import (
    DataArrayCoordinates, LevelCoordinatesSource, assert_coordinate_consistent,
    remap_label_indexers)
from .dataset import Dataset, merge_indexes, split_indexes
from .formatting import format_item
from .indexes import Indexes, default_indexes
from .pycompat import TYPE_CHECKING
from .options import OPTIONS
from .utils import _check_inplace, either_dict_or_kwargs, ReprObject
from .variable import (
    IndexVariable, Variable, as_compatible_data, as_variable,
    assert_unique_multiindex_level_names)
from dask.delayed import Delayed
from cdms2 import Variable as cdms2_Variable
from iris.cube import Cube as iris_Cube
from .dataset import _get_virtual_variable
from .missing import interp_na
from .missing import ffill
from .missing import bfill
from ..backends.api import DATAARRAY_NAME, DATAARRAY_VARIABLE
from ..convert import to_cdms2
from ..convert import from_cdms2
from ..convert import to_iris
from ..convert import from_iris

_THIS_ARRAY = ReprObject('<this-array>')

class DataArray(AbstractArray, DataWithCoords):
    _groupby_cls = groupby.DataArrayGroupBy
    _rolling_cls = rolling.DataArrayRolling
    _coarsen_cls = rolling.DataArrayCoarsen
    _resample_cls = resample.DataArrayResample
    __default = ReprObject('<default>')
    dt = property(DatetimeAccessor)
    __hash__ = None  # type: ignore
    __default_name = object()
    str = property(StringAccessor)
    def _to_temp_dataset(self) -> Dataset:
        return self._to_dataset_whole(name=_THIS_ARRAY,
                                      shallow_copy=False)
    def _to_dataset_whole(
            self,
            name: Optional[Hashable] = None,
            shallow_copy: bool = True
    ) -> Dataset:
        if name is None:
            name = self.name
        if name is None:
            raise ValueError('unable to convert unnamed DataArray to a '
                             'Dataset without providing an explicit name')
        if name in self.coords:
            raise ValueError('cannot create a Dataset from a DataArray with '
                             'the same name as one of its coordinates')
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
    def expand_dims(self, dim: Union[None, Hashable, Sequence[Hashable],
                                     Mapping[Hashable, Any]] = None,
                    axis=None, **dim_kwargs: Any) -> 'DataArray':
        """Return a new object with an additional axis (or axes) inserted at
        the corresponding position in the array shape. The new object is a
        view into the underlying array, not a copy.


        If dim is already a scalar coordinate, it will be promoted to a 1D
        coordinate consisting of a single value.

        Parameters
        ----------
        dim : hashable, sequence of hashable, dict, or None
            Dimensions to include on the new variable.
            If provided as str or sequence of str, then dimensions are inserted
            with length 1. If provided as a dict, then the keys are the new
            dimensions and the values are either integers (giving the length of
            the new dimensions) or sequence/ndarray (giving the coordinates of
            the new dimensions). **WARNING** for python 3.5, if ``dim`` is
            dict-like, then it must be an ``OrderedDict``. This is to ensure
            that the order in which the dims are given is maintained.
        axis : integer, list (or tuple) of integers, or None
            Axis position(s) where new axis is to be inserted (position(s) on
            the result array). If a list (or tuple) of integers is passed,
            multiple axes are inserted. In this case, dim arguments should be
            same length list. If axis=None is passed, all the axes will be
            inserted to the start of the result array.
        **dim_kwargs : int or sequence/ndarray
            The keywords are arbitrary dimensions being inserted and the values
            are either the lengths of the new dims (if int is given), or their
            coordinates. Note, this is an alternative to passing a dict to the
            dim kwarg and will only be used if dim is None. **WARNING** for
            python 3.5 ``dim_kwargs`` is not available.

        Returns
        -------
        expanded : same type as caller
            This object, but with an additional dimension(s).
        """
        if isinstance(dim, int):
            raise TypeError('dim should be hashable or sequence/mapping of '
                            'hashables')
        elif isinstance(dim, Sequence) and not isinstance(dim, str):
            if len(dim) != len(set(dim)):
                raise ValueError('dims should not contain duplicate values.')
            dim = OrderedDict(((d, 1) for d in dim))
        elif dim is not None and not isinstance(dim, Mapping):
            dim = OrderedDict(((cast(Hashable, dim), 1),))

        # TODO: get rid of the below code block when python 3.5 is no longer
        #   supported.
        python36_plus = sys.version_info[0] == 3 and sys.version_info[1] > 5
        not_ordereddict = dim is not None and not isinstance(dim, OrderedDict)
        if not python36_plus and not_ordereddict:
            raise TypeError("dim must be an OrderedDict for python <3.6")
        elif not python36_plus and dim_kwargs:
            raise ValueError("dim_kwargs isn't available for python <3.6")
        dim_kwargs = OrderedDict(dim_kwargs)

        dim = either_dict_or_kwargs(dim, dim_kwargs, 'expand_dims')
        ds = self._to_temp_dataset().expand_dims(dim, axis)
        return self._from_temp_dataset(ds)