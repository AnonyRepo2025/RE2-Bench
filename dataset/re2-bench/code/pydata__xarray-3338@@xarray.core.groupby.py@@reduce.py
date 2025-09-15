import datetime
import functools
import warnings
import numpy as np
import pandas as pd
from . import dtypes, duck_array_ops, nputils, ops
from .arithmetic import SupportsArithmetic
from .common import ALL_DIMS, ImplementsArrayReduce, ImplementsDatasetReduce
from .concat import concat
from .formatting import format_array_flat
from .options import _get_keep_attrs
from .pycompat import integer_types
from .utils import (
    either_dict_or_kwargs,
    hashable,
    maybe_wrap_array,
    peek_at,
    safe_cast_to_index,
)
from .variable import IndexVariable, Variable, as_variable
from .dataset import Dataset
from .dataarray import DataArray
from .dataarray import DataArray
from .resample_cftime import CFTimeGrouper



class DataArrayGroupBy(GroupBy, ImplementsArrayReduce):
    def reduce(
        self, func, dim=None, axis=None, keep_attrs=None, shortcut=True, **kwargs
    ):
        """Reduce the items in this group by applying `func` along some
        dimension(s).

        Parameters
        ----------
        func : function
            Function which can be called in the form
            `func(x, axis=axis, **kwargs)` to return the result of collapsing
            an np.ndarray over an integer valued axis.
        dim : xarray.ALL_DIMS, str or sequence of str, optional
            Dimension(s) over which to apply `func`.
        axis : int or sequence of int, optional
            Axis(es) over which to apply `func`. Only one of the 'dimension'
            and 'axis' arguments can be supplied. If neither are supplied, then
            `func` is calculated over all dimension for each group item.
        keep_attrs : bool, optional
            If True, the datasets's attributes (`attrs`) will be copied from
            the original object to the new one.  If False (default), the new
            object will be returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to `func`.

        Returns
        -------
        reduced : Array
            Array with summarized data and the indicated dimension(s)
            removed.
        """
        if dim is None:
            dim = self._group_dim

        if keep_attrs is None:
            keep_attrs = _get_keep_attrs(default=False)

        if dim is not ALL_DIMS and dim not in self.dims:
            raise ValueError(
                "cannot reduce over dimension %r. expected either xarray.ALL_DIMS to reduce over all dimensions or one or more of %r."
                % (dim, self.dims)
            )

        def reduce_array(ar):
            return ar.reduce(func, dim, axis, keep_attrs=keep_attrs, **kwargs)

        return self.apply(reduce_array, shortcut=shortcut)