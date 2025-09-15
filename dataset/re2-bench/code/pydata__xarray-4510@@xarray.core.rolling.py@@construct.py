import functools
import warnings
from typing import Any, Callable, Dict
import numpy as np
from . import dtypes, duck_array_ops, utils
from .dask_array_ops import dask_rolling_wrapper
from .ops import inject_reduce_methods
from .options import _get_keep_attrs
from .pycompat import is_duck_dask_array
import bottleneck
from .dataarray import DataArray
from .dataarray import DataArray
from .dataset import Dataset
from .dataset import Dataset
from .dataarray import DataArray
from .dataset import Dataset

_ROLLING_REDUCE_DOCSTRING_TEMPLATE = """\
Reduce this object's data windows by applying `{name}` along its dimension.

Parameters
----------
keep_attrs : bool, default: None
    If True, the attributes (``attrs``) will be copied from the original
    object to the new one. If False, the new object will be returned
    without attributes. If None uses the global default.
**kwargs : dict
    Additional keyword arguments passed on to `{name}`.

Returns
-------
reduced : same type as caller
    New object with `{name}` applied along its rolling dimnension.
"""

class DataArrayRolling(Rolling):
    __slots__ = ("window_labels",)
    def construct(
        self,
        window_dim=None,
        stride=1,
        fill_value=dtypes.NA,
        keep_attrs=None,
        **window_dim_kwargs,
    ):
        """
        Convert this rolling object to xr.DataArray,
        where the window dimension is stacked as a new dimension

        Parameters
        ----------
        window_dim : str or mapping, optional
            A mapping from dimension name to the new window dimension names.
        stride : int or mapping of int, default: 1
            Size of stride for the rolling window.
        fill_value : default: dtypes.NA
            Filling value to match the dimension size.
        keep_attrs : bool, default: None
            If True, the attributes (``attrs``) will be copied from the original
            object to the new one. If False, the new object will be returned
            without attributes. If None uses the global default.
        **window_dim_kwargs : {dim: new_name, ...}, optional
            The keyword arguments form of ``window_dim``.

        Returns
        -------
        DataArray that is a view of the original array. The returned array is
        not writeable.

        Examples
        --------
        >>> da = xr.DataArray(np.arange(8).reshape(2, 4), dims=("a", "b"))

        >>> rolling = da.rolling(b=3)
        >>> rolling.construct("window_dim")
        <xarray.DataArray (a: 2, b: 4, window_dim: 3)>
        array([[[nan, nan,  0.],
                [nan,  0.,  1.],
                [ 0.,  1.,  2.],
                [ 1.,  2.,  3.]],
        <BLANKLINE>
               [[nan, nan,  4.],
                [nan,  4.,  5.],
                [ 4.,  5.,  6.],
                [ 5.,  6.,  7.]]])
        Dimensions without coordinates: a, b, window_dim

        >>> rolling = da.rolling(b=3, center=True)
        >>> rolling.construct("window_dim")
        <xarray.DataArray (a: 2, b: 4, window_dim: 3)>
        array([[[nan,  0.,  1.],
                [ 0.,  1.,  2.],
                [ 1.,  2.,  3.],
                [ 2.,  3., nan]],
        <BLANKLINE>
               [[nan,  4.,  5.],
                [ 4.,  5.,  6.],
                [ 5.,  6.,  7.],
                [ 6.,  7., nan]]])
        Dimensions without coordinates: a, b, window_dim

        """

        from .dataarray import DataArray

        keep_attrs = self._get_keep_attrs(keep_attrs)

        if window_dim is None:
            if len(window_dim_kwargs) == 0:
                raise ValueError(
                    "Either window_dim or window_dim_kwargs need to be specified."
                )
            window_dim = {d: window_dim_kwargs[d] for d in self.dim}

        window_dim = self._mapping_to_list(
            window_dim, allow_default=False, allow_allsame=False
        )
        stride = self._mapping_to_list(stride, default=1)

        window = self.obj.variable.rolling_window(
            self.dim, self.window, window_dim, self.center, fill_value=fill_value
        )

        attrs = self.obj.attrs if keep_attrs else {}

        result = DataArray(
            window,
            dims=self.obj.dims + tuple(window_dim),
            coords=self.obj.coords,
            attrs=attrs,
            name=self.obj.name,
        )
        return result.isel(
            **{d: slice(None, None, s) for d, s in zip(self.dim, stride)}
        )