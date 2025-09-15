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
    def _restore_dim_order(self, stacked):
        def lookup_order(dimension):
            if dimension == self._group.name:
                dimension, = self._group.dims
            if dimension in self._obj.dims:
                axis = self._obj.get_axis_num(dimension)
            else:
                axis = 1e6  # some arbitrarily high value
            return axis

        new_order = sorted(stacked.dims, key=lookup_order)
        return stacked.transpose(*new_order, transpose_coords=self._restore_coord_dims)
    def apply(self, func, shortcut=False, args=(), **kwargs):
        """Apply a function over each array in the group and concatenate them
        together into a new array.

        `func` is called like `func(ar, *args, **kwargs)` for each array `ar`
        in this group.

        Apply uses heuristics (like `pandas.GroupBy.apply`) to figure out how
        to stack together the array. The rule is:

        1. If the dimension along which the group coordinate is defined is
           still in the first grouped array after applying `func`, then stack
           over this dimension.
        2. Otherwise, stack over the new dimension given by name of this
           grouping (the argument to the `groupby` function).

        Parameters
        ----------
        func : function
            Callable to apply to each array.
        shortcut : bool, optional
            Whether or not to shortcut evaluation under the assumptions that:
            (1) The action of `func` does not depend on any of the array
                metadata (attributes or coordinates) but only on the data and
                dimensions.
            (2) The action of `func` creates arrays with homogeneous metadata,
                that is, with the same dimensions and attributes.
            If these conditions are satisfied `shortcut` provides significant
            speedup. This should be the case for many common groupby operations
            (e.g., applying numpy ufuncs).
        args : tuple, optional
            Positional arguments passed to `func`.
        **kwargs
            Used to call `func(ar, **kwargs)` for each array `ar`.

        Returns
        -------
        applied : DataArray or DataArray
            The result of splitting, applying and combining this array.
        """
        if shortcut:
            grouped = self._iter_grouped_shortcut()
        else:
            grouped = self._iter_grouped()
        applied = (maybe_wrap_array(arr, func(arr, *args, **kwargs)) for arr in grouped)
        return self._combine(applied, shortcut=shortcut)
    def _combine(self, applied, restore_coord_dims=False, shortcut=False):
        """Recombine the applied objects like the original."""
        applied_example, applied = peek_at(applied)
        coord, dim, positions = self._infer_concat_args(applied_example)
        if shortcut:
            combined = self._concat_shortcut(applied, dim, positions)
        else:
            combined = concat(applied, dim)
            combined = _maybe_reorder(combined, dim, positions)

        if isinstance(combined, type(self._obj)):
            # only restore dimension order for arrays
            combined = self._restore_dim_order(combined)
        if coord is not None:
            if shortcut:
                combined._coords[coord.name] = as_variable(coord)
            else:
                combined.coords[coord.name] = coord
        combined = self._maybe_restore_empty_groups(combined)
        combined = self._maybe_unstack(combined)
        return combined
    def quantile(self, q, dim=None, interpolation="linear", keep_attrs=None):
        """Compute the qth quantile over each array in the groups and
        concatenate them together into a new array.

        Parameters
        ----------
        q : float in range of [0,1] (or sequence of floats)
            Quantile to compute, which must be between 0 and 1
            inclusive.
        dim : xarray.ALL_DIMS, str or sequence of str, optional
            Dimension(s) over which to apply quantile.
            Defaults to the grouped dimension.
        interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
            This optional parameter specifies the interpolation method to
            use when the desired quantile lies between two data points
            ``i < j``:
                * linear: ``i + (j - i) * fraction``, where ``fraction`` is
                  the fractional part of the index surrounded by ``i`` and
                  ``j``.
                * lower: ``i``.
                * higher: ``j``.
                * nearest: ``i`` or ``j``, whichever is nearest.
                * midpoint: ``(i + j) / 2``.

        Returns
        -------
        quantiles : Variable
            If `q` is a single quantile, then the result
            is a scalar. If multiple percentiles are given, first axis of
            the result corresponds to the quantile and a quantile dimension
            is added to the return array. The other dimensions are the
            dimensions that remain after the reduction of the array.

        See Also
        --------
        numpy.nanpercentile, pandas.Series.quantile, Dataset.quantile,
        DataArray.quantile
        """
        if dim is None:
            dim = self._group_dim

        out = self.apply(
            self._obj.__class__.quantile,
            shortcut=False,
            q=q,
            dim=dim,
            interpolation=interpolation,
            keep_attrs=keep_attrs,
        )

        if np.asarray(q, dtype=np.float64).ndim == 0:
            out = out.drop("quantile")
        return out