from __future__ import annotations
import datetime
import warnings
from typing import Any, Callable, Hashable, Sequence
import numpy as np
import pandas as pd
from . import dtypes, duck_array_ops, nputils, ops
from ._reductions import DataArrayGroupByReductions, DatasetGroupByReductions
from .arithmetic import DataArrayGroupbyArithmetic, DatasetGroupbyArithmetic
from .concat import concat
from .formatting import format_array_flat
from .indexes import create_default_index_implicit, filter_indexes_from_coords
from .options import _get_keep_attrs
from .pycompat import integer_types
from .utils import (
    either_dict_or_kwargs,
    hashable,
    is_scalar,
    maybe_wrap_array,
    peek_at,
    safe_cast_to_index,
)
from .variable import IndexVariable, Variable
from .dataarray import DataArray
from .dataset import Dataset
from .dataarray import DataArray
from .resample_cftime import CFTimeGrouper



class DataArrayGroupByBase(GroupBy, DataArrayGroupbyArithmetic):
    __slots__ = ()
    def _iter_grouped_shortcut(self):
        """Fast version of `_iter_grouped` that yields Variables without
        metadata
        """
        var = self._obj.variable
        for indices in self._group_indices:
            yield var[{self._group_dim: indices}]
    def _concat_shortcut(self, applied, dim, positions=None):
        # nb. don't worry too much about maintaining this method -- it does
        # speed things up, but it's not very interpretable and there are much
        # faster alternatives (e.g., doing the grouped aggregation in a
        # compiled language)
        # TODO: benbovy - explicit indexes: this fast implementation doesn't
        # create an explicit index for the stacked dim coordinate
        stacked = Variable.concat(applied, dim, shortcut=True)
        reordered = _maybe_reorder(stacked, dim, positions)
        return self._obj._replace_maybe_drop_dims(reordered)
    def _restore_dim_order(self, stacked):
        def lookup_order(dimension):
            if dimension == self._group.name:
                (dimension,) = self._group.dims
            if dimension in self._obj.dims:
                axis = self._obj.get_axis_num(dimension)
            else:
                axis = 1e6  # some arbitrarily high value
            return axis

        new_order = sorted(stacked.dims, key=lookup_order)
        return stacked.transpose(*new_order, transpose_coords=self._restore_coord_dims)
    def map(self, func, shortcut=False, args=(), **kwargs):
        """Apply a function to each array in the group and concatenate them
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
        func : callable
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
        *args : tuple, optional
            Positional arguments passed to `func`.
        **kwargs
            Used to call `func(ar, **kwargs)` for each array `ar`.

        Returns
        -------
        applied : DataArray or DataArray
            The result of splitting, applying and combining this array.
        """
        grouped = self._iter_grouped_shortcut() if shortcut else self._iter_grouped()
        applied = (maybe_wrap_array(arr, func(arr, *args, **kwargs)) for arr in grouped)
        return self._combine(applied, shortcut=shortcut)
    def _combine(self, applied, shortcut=False):
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
        # assign coord and index when the applied function does not return that coord
        if coord is not None and dim not in applied_example.dims:
            index, index_vars = create_default_index_implicit(coord)
            indexes = {k: index for k in index_vars}
            combined = combined._overwrite_indexes(indexes, index_vars)
        combined = self._maybe_restore_empty_groups(combined)
        combined = self._maybe_unstack(combined)
        return combined
    def reduce(
        self,
        func: Callable[..., Any],
        dim: None | Hashable | Sequence[Hashable] = None,
        *,
        axis: None | int | Sequence[int] = None,
        keep_attrs: bool = None,
        keepdims: bool = False,
        shortcut: bool = True,
        **kwargs: Any,
    ):
        """Reduce the items in this group by applying `func` along some
        dimension(s).

        Parameters
        ----------
        func : callable
            Function which can be called in the form
            `func(x, axis=axis, **kwargs)` to return the result of collapsing
            an np.ndarray over an integer valued axis.
        dim : ..., str or sequence of str, optional
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

        def reduce_array(ar):
            return ar.reduce(
                func=func,
                dim=dim,
                axis=axis,
                keep_attrs=keep_attrs,
                keepdims=keepdims,
                **kwargs,
            )

        check_reduce_dims(dim, self.dims)

        return self.map(reduce_array, shortcut=shortcut)