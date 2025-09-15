import functools
import itertools
from collections import OrderedDict, defaultdict
from datetime import timedelta
from distutils.version import LooseVersion
from typing import Any, Hashable, Mapping, MutableMapping, Union
import numpy as np
import pandas as pd
import xarray as xr  # only for Dataset and DataArray
from . import (
    arithmetic, common, dtypes, duck_array_ops, indexing, nputils, ops, utils)
from .indexing import (
    BasicIndexer, OuterIndexer, PandasIndexAdapter, VectorizedIndexer,
    as_indexable)
from .options import _get_keep_attrs
from .pycompat import dask_array_type, integer_types
from .utils import (
    OrderedSet, decode_numpy_dict_values, either_dict_or_kwargs,
    ensure_us_time_resolution)
import dask.array as da
from .dataarray import DataArray
import dask
import dask.array as da
import warnings
import bottleneck as bn

NON_NUMPY_SUPPORTED_ARRAY_TYPES = (
    indexing.ExplicitlyIndexed, pd.Index) + dask_array_type
BASIC_INDEXING_TYPES = integer_types + (slice,)  # type: ignore
Coordinate = utils.alias(IndexVariable, 'Coordinate')

class IndexVariable(Variable):
    to_coord = utils.alias(to_index_variable, 'to_coord')
    def copy(self, deep=True, data=None):
        """Returns a copy of this object.

        `deep` is ignored since data is stored in the form of
        pandas.Index, which is already immutable. Dimensions, attributes
        and encodings are always copied.

        Use `data` to create a new object with the same structure as
        original but entirely new data.

        Parameters
        ----------
        deep : bool, optional
            Deep is ignored when data is given. Whether the data array is
            loaded into memory and copied onto the new object. Default is True.
        data : array_like, optional
            Data to use in the new object. Must have same shape as original.

        Returns
        -------
        object : Variable
            New object with dimensions, attributes, encodings, and optionally
            data copied from original.
        """
        if data is None:
            data = self._data.copy(deep=deep)
        else:
            data = as_compatible_data(data)
            if self.shape != data.shape:
                raise ValueError("Data shape {} must match shape of object {}"
                                 .format(data.shape, self.shape))
        return type(self)(self.dims, data, self._attrs,
                          self._encoding, fastpath=True)