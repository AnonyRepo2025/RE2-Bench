import datetime
import functools
import warnings
import numpy as np
import pandas as pd
from . import dtypes, duck_array_ops, nputils, ops
from .arithmetic import SupportsArithmetic
from .common import ImplementsArrayReduce, ImplementsDatasetReduce
from .concat import concat
from .formatting import format_array_flat
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
from .variable import IndexVariable, Variable, as_variable
from .dataset import Dataset
from .dataarray import DataArray
from .dataarray import DataArray
from .resample_cftime import CFTimeGrouper



class GroupBy(SupportsArithmetic):
    __slots__ = (
        "_full_index",
        "_inserted_dims",
        "_group",
        "_group_dim",
        "_group_indices",
        "_groups",
        "_obj",
        "_restore_coord_dims",
        "_stacked_dim",
        "_unique_coord",
        "_dims",
    )
    def _iter_grouped(self):
        """Iterate over each element in this group"""
        for indices in self._group_indices:
            yield self._obj.isel(**{self._group_dim: indices})
    def _infer_concat_args(self, applied_example):
        if self._group_dim in applied_example.dims:
            coord = self._group
            positions = self._group_indices
        else:
            coord = self._unique_coord
            positions = None
        (dim,) = coord.dims
        if isinstance(coord, _DummyGroup):
            coord = None
        return coord, dim, positions
    def _maybe_restore_empty_groups(self, combined):
        """Our index contained empty groups (e.g., from a resampling). If we
        reduced on that dimension, we want to restore the full index.
        """
        if self._full_index is not None and self._group.name in combined.dims:
            indexers = {self._group.name: self._full_index}
            combined = combined.reindex(**indexers)
        return combined
    def _maybe_unstack(self, obj):
        """This gets called if we are applying on an array with a
        multidimensional group."""
        if self._stacked_dim is not None and self._stacked_dim in obj.dims:
            obj = obj.unstack(self._stacked_dim)
            for dim in self._inserted_dims:
                if dim in obj.coords:
                    del obj.coords[dim]
                    del obj.indexes[dim]
        return obj
    def quantile(self, q, dim=None, interpolation="linear", keep_attrs=None):
        """Compute the qth quantile over each array in the groups and
        concatenate them together into a new array.

        Parameters
        ----------
        q : float in range of [0,1] (or sequence of floats)
            Quantile to compute, which must be between 0 and 1
            inclusive.
        dim : `...`, str or sequence of str, optional
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
            If `q` is a single quantile, then the result is a
            scalar. If multiple percentiles are given, first axis of
            the result corresponds to the quantile. In either case a
            quantile dimension is added to the return array. The other
            dimensions are the dimensions that remain after the
            reduction of the array.

        See Also
        --------
        numpy.nanpercentile, pandas.Series.quantile, Dataset.quantile,
        DataArray.quantile
        """
        if dim is None:
            dim = self._group_dim

        out = self.map(
            self._obj.__class__.quantile,
            shortcut=False,
            q=q,
            dim=dim,
            interpolation=interpolation,
            keep_attrs=keep_attrs,
        )

        return out