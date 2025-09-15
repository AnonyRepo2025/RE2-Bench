import copy
import functools
import sys
import warnings
from collections import OrderedDict, defaultdict
from distutils.version import LooseVersion
from numbers import Number
from pathlib import Path
from typing import (Any, Dict, Hashable, Iterable, Iterator, List,
                    Mapping, MutableMapping, Optional, Sequence, Set, Tuple,
                    Union, cast)
import numpy as np
import pandas as pd
import xarray as xr
from ..coding.cftimeindex import _parse_array_of_cftime_strings
from . import (alignment, dtypes, duck_array_ops, formatting, groupby,
               indexing, ops, pdcompat, resample, rolling, utils)
from .alignment import align
from .common import (ALL_DIMS, DataWithCoords, ImplementsDatasetReduce,
                     _contains_datetime_like_objects)
from .coordinates import (DatasetCoordinates, LevelCoordinatesSource,
                          assert_coordinate_consistent, remap_label_indexers)
from .duck_array_ops import datetime_to_numeric
from .indexes import (
    Indexes, default_indexes, isel_variable_and_index, roll_index,
)
from .merge import (
    dataset_merge_method, dataset_update_method, merge_data_and_coords,
    merge_variables)
from .options import OPTIONS, _get_keep_attrs
from .pycompat import TYPE_CHECKING, dask_array_type
from .utils import (Frozen, SortedKeysDict, _check_inplace,
                    decode_numpy_dict_values, either_dict_or_kwargs, hashable,
                    maybe_wrap_array)
from .variable import IndexVariable, Variable, as_variable, broadcast_variables
from .pycompat import Mapping  # noqa: F811
from typing import DefaultDict
from ..backends import AbstractDataStore, ZarrStore
from .dataarray import DataArray
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

_DATETIMEINDEX_COMPONENTS = ['year', 'month', 'day', 'hour', 'minute',
                             'second', 'microsecond', 'nanosecond', 'date',
                             'time', 'dayofyear', 'weekofyear', 'dayofweek',
                             'quarter']

class Dataset(Mapping, ImplementsDatasetReduce, DataWithCoords):
    _groupby_cls = groupby.DatasetGroupBy
    _rolling_cls = rolling.DatasetRolling
    _coarsen_cls = rolling.DatasetCoarsen
    _resample_cls = resample.DatasetResample
    __default = object()
    __hash__ = None  # type: ignore
    @property
    def dims(self) -> Mapping[Hashable, int]:
        """Mapping from dimension names to lengths.

        Cannot be modified directly, but is updated when adding new variables.

        Note that type of this object differs from `DataArray.dims`.
        See `Dataset.sizes` and `DataArray.sizes` for consistently named
        properties.
        """
        return Frozen(SortedKeysDict(self._dims))
    def expand_dims(self, dim=None, axis=None, **dim_kwargs):
        """Return a new object with an additional axis (or axes) inserted at
        the corresponding position in the array shape.  The new object is a
        view into the underlying array, not a copy.

        If dim is already a scalar coordinate, it will be promoted to a 1D
        coordinate consisting of a single value.

        Parameters
        ----------
        dim : str, sequence of str, dict, or None
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
            raise TypeError('dim should be str or sequence of strs or dict')
        elif isinstance(dim, str):
            dim = OrderedDict(((dim, 1),))
        elif isinstance(dim, (list, tuple)):
            if len(dim) != len(set(dim)):
                raise ValueError('dims should not contain duplicate values.')
            dim = OrderedDict(((d, 1) for d in dim))

        # TODO: get rid of the below code block when python 3.5 is no longer
        #   supported.
        python36_plus = sys.version_info[0] == 3 and sys.version_info[1] > 5
        not_ordereddict = dim is not None and not isinstance(dim, OrderedDict)
        if not python36_plus and not_ordereddict:
            raise TypeError("dim must be an OrderedDict for python <3.6")
        elif not python36_plus and dim_kwargs:
            raise ValueError("dim_kwargs isn't available for python <3.6")

        dim = either_dict_or_kwargs(dim, dim_kwargs, 'expand_dims')

        if axis is not None and not isinstance(axis, (list, tuple)):
            axis = [axis]

        if axis is None:
            axis = list(range(len(dim)))

        if len(dim) != len(axis):
            raise ValueError('lengths of dim and axis should be identical.')
        for d in dim:
            if d in self.dims:
                raise ValueError(
                    'Dimension {dim} already exists.'.format(dim=d))
            if (d in self._variables
                    and not utils.is_scalar(self._variables[d])):
                raise ValueError(
                    '{dim} already exists as coordinate or'
                    ' variable name.'.format(dim=d))

        variables = OrderedDict()
        coord_names = self._coord_names.copy()
        # If dim is a dict, then ensure that the values are either integers
        # or iterables.
        for k, v in dim.items():
            if hasattr(v, "__iter__"):
                # If the value for the new dimension is an iterable, then
                # save the coordinates to the variables dict, and set the
                # value within the dim dict to the length of the iterable
                # for later use.
                variables[k] = xr.IndexVariable((k,), v)
                coord_names.add(k)
                dim[k] = variables[k].size
            elif isinstance(v, int):
                pass  # Do nothing if the dimensions value is just an int
            else:
                raise TypeError('The value of new dimension {k} must be '
                                'an iterable or an int'.format(k=k))

        for k, v in self._variables.items():
            if k not in dim:
                if k in coord_names:  # Do not change coordinates
                    variables[k] = v
                else:
                    result_ndim = len(v.dims) + len(axis)
                    for a in axis:
                        if a < -result_ndim or result_ndim - 1 < a:
                            raise IndexError(
                                'Axis {a} is out of bounds of the expanded'
                                ' dimension size {dim}.'.format(
                                    a=a, v=k, dim=result_ndim))

                    axis_pos = [a if a >= 0 else result_ndim + a
                                for a in axis]
                    if len(axis_pos) != len(set(axis_pos)):
                        raise ValueError('axis should not contain duplicate'
                                         ' values.')
                    # We need to sort them to make sure `axis` equals to the
                    # axis positions of the result array.
                    zip_axis_dim = sorted(zip(axis_pos, dim.items()))

                    all_dims = list(zip(v.dims, v.shape))
                    for d, c in zip_axis_dim:
                        all_dims.insert(d, c)
                    all_dims = OrderedDict(all_dims)

                    variables[k] = v.set_dims(all_dims)
            else:
                # If dims includes a label of a non-dimension coordinate,
                # it will be promoted to a 1D coordinate with a single value.
                variables[k] = v.set_dims(k).to_index_variable()

        new_dims = self._dims.copy()
        new_dims.update(dim)

        return self._replace_vars_and_dims(
            variables, dims=new_dims, coord_names=coord_names)