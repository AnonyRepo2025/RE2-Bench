from __future__ import annotations
import functools
import itertools
import operator
import warnings
from collections import Counter
from typing import (
    TYPE_CHECKING,
    AbstractSet,
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
    Union,
)
import numpy as np
from . import dtypes, duck_array_ops, utils
from .alignment import align, deep_align
from .merge import merge_attrs, merge_coordinates_without_align
from .options import OPTIONS, _get_keep_attrs
from .pycompat import is_duck_dask_array
from .utils import is_dict_like
from .variable import Variable
from .coordinates import Coordinates
from .dataarray import DataArray
from .dataset import Dataset
from .types import T_Xarray
from .dataarray import DataArray
from .dataset import Dataset
from .dataset import Dataset
from .groupby import _dummy_copy
from .groupby import GroupBy, peek_at
from .variable import Variable
from .variable import Variable, as_compatible_data
from .dataarray import DataArray
from .groupby import GroupBy
from .variable import Variable
from .dataarray import DataArray
from .dataarray import DataArray
from .dataarray import DataArray
from .variable import Variable
from .dataarray import DataArray
from .missing import get_clean_interp_index
from .dataarray import DataArray
from dask.array.core import unify_chunks
import dask.array
import dask.array as da

_NO_FILL_VALUE = utils.ReprObject("<no-fill-value>")
_DEFAULT_NAME = utils.ReprObject("<default-name>")
_JOINS_WITHOUT_FILL_VALUES = frozenset({"inner", "exact"})
SLICE_NONE = slice(None)

def cross(
    a: Union[DataArray, Variable], b: Union[DataArray, Variable], *, dim: Hashable
) -> Union[DataArray, Variable]:
    """
    Compute the cross product of two (arrays of) vectors.

    The cross product of `a` and `b` in :math:`R^3` is a vector
    perpendicular to both `a` and `b`. The vectors in `a` and `b` are
    defined by the values along the dimension `dim` and can have sizes
    1, 2 or 3. Where the size of either `a` or `b` is
    1 or 2, the remaining components of the input vector is assumed to
    be zero and the cross product calculated accordingly. In cases where
    both input vectors have dimension 2, the z-component of the cross
    product is returned.

    Parameters
    ----------
    a, b : DataArray or Variable
        Components of the first and second vector(s).
    dim : hashable
        The dimension along which the cross product will be computed.
        Must be available in both vectors.

    Examples
    --------
    Vector cross-product with 3 dimensions:

    >>> a = xr.DataArray([1, 2, 3])
    >>> b = xr.DataArray([4, 5, 6])
    >>> xr.cross(a, b, dim="dim_0")
    <xarray.DataArray (dim_0: 3)>
    array([-3,  6, -3])
    Dimensions without coordinates: dim_0

    Vector cross-product with 2 dimensions, returns in the perpendicular
    direction:

    >>> a = xr.DataArray([1, 2])
    >>> b = xr.DataArray([4, 5])
    >>> xr.cross(a, b, dim="dim_0")
    <xarray.DataArray ()>
    array(-3)

    Vector cross-product with 3 dimensions but zeros at the last axis
    yields the same results as with 2 dimensions:

    >>> a = xr.DataArray([1, 2, 0])
    >>> b = xr.DataArray([4, 5, 0])
    >>> xr.cross(a, b, dim="dim_0")
    <xarray.DataArray (dim_0: 3)>
    array([ 0,  0, -3])
    Dimensions without coordinates: dim_0

    One vector with dimension 2:

    >>> a = xr.DataArray(
    ...     [1, 2],
    ...     dims=["cartesian"],
    ...     coords=dict(cartesian=(["cartesian"], ["x", "y"])),
    ... )
    >>> b = xr.DataArray(
    ...     [4, 5, 6],
    ...     dims=["cartesian"],
    ...     coords=dict(cartesian=(["cartesian"], ["x", "y", "z"])),
    ... )
    >>> xr.cross(a, b, dim="cartesian")
    <xarray.DataArray (cartesian: 3)>
    array([12, -6, -3])
    Coordinates:
      * cartesian  (cartesian) <U1 'x' 'y' 'z'

    One vector with dimension 2 but coords in other positions:

    >>> a = xr.DataArray(
    ...     [1, 2],
    ...     dims=["cartesian"],
    ...     coords=dict(cartesian=(["cartesian"], ["x", "z"])),
    ... )
    >>> b = xr.DataArray(
    ...     [4, 5, 6],
    ...     dims=["cartesian"],
    ...     coords=dict(cartesian=(["cartesian"], ["x", "y", "z"])),
    ... )
    >>> xr.cross(a, b, dim="cartesian")
    <xarray.DataArray (cartesian: 3)>
    array([-10,   2,   5])
    Coordinates:
      * cartesian  (cartesian) <U1 'x' 'y' 'z'

    Multiple vector cross-products. Note that the direction of the
    cross product vector is defined by the right-hand rule:

    >>> a = xr.DataArray(
    ...     [[1, 2, 3], [4, 5, 6]],
    ...     dims=("time", "cartesian"),
    ...     coords=dict(
    ...         time=(["time"], [0, 1]),
    ...         cartesian=(["cartesian"], ["x", "y", "z"]),
    ...     ),
    ... )
    >>> b = xr.DataArray(
    ...     [[4, 5, 6], [1, 2, 3]],
    ...     dims=("time", "cartesian"),
    ...     coords=dict(
    ...         time=(["time"], [0, 1]),
    ...         cartesian=(["cartesian"], ["x", "y", "z"]),
    ...     ),
    ... )
    >>> xr.cross(a, b, dim="cartesian")
    <xarray.DataArray (time: 2, cartesian: 3)>
    array([[-3,  6, -3],
           [ 3, -6,  3]])
    Coordinates:
      * time       (time) int64 0 1
      * cartesian  (cartesian) <U1 'x' 'y' 'z'

    Cross can be called on Datasets by converting to DataArrays and later
    back to a Dataset:

    >>> ds_a = xr.Dataset(dict(x=("dim_0", [1]), y=("dim_0", [2]), z=("dim_0", [3])))
    >>> ds_b = xr.Dataset(dict(x=("dim_0", [4]), y=("dim_0", [5]), z=("dim_0", [6])))
    >>> c = xr.cross(
    ...     ds_a.to_array("cartesian"), ds_b.to_array("cartesian"), dim="cartesian"
    ... )
    >>> c.to_dataset(dim="cartesian")
    <xarray.Dataset>
    Dimensions:  (dim_0: 1)
    Dimensions without coordinates: dim_0
    Data variables:
        x        (dim_0) int64 -3
        y        (dim_0) int64 6
        z        (dim_0) int64 -3

    See Also
    --------
    numpy.cross : Corresponding numpy function
    """

    if dim not in a.dims:
        raise ValueError(f"Dimension {dim!r} not on a")
    elif dim not in b.dims:
        raise ValueError(f"Dimension {dim!r} not on b")

    if not 1 <= a.sizes[dim] <= 3:
        raise ValueError(
            f"The size of {dim!r} on a must be 1, 2, or 3 to be "
            f"compatible with a cross product but is {a.sizes[dim]}"
        )
    elif not 1 <= b.sizes[dim] <= 3:
        raise ValueError(
            f"The size of {dim!r} on b must be 1, 2, or 3 to be "
            f"compatible with a cross product but is {b.sizes[dim]}"
        )

    all_dims = list(dict.fromkeys(a.dims + b.dims))

    if a.sizes[dim] != b.sizes[dim]:
        # Arrays have different sizes. Append zeros where the smaller
        # array is missing a value, zeros will not affect np.cross:

        if (
            not isinstance(a, Variable)  # Only used to make mypy happy.
            and dim in getattr(a, "coords", {})
            and not isinstance(b, Variable)  # Only used to make mypy happy.
            and dim in getattr(b, "coords", {})
        ):
            # If the arrays have coords we know which indexes to fill
            # with zeros:
            a, b = align(
                a,
                b,
                fill_value=0,
                join="outer",
                exclude=set(all_dims) - {dim},
            )
        elif min(a.sizes[dim], b.sizes[dim]) == 2:
            # If the array doesn't have coords we can only infer
            # that it has composite values if the size is at least 2.
            # Once padded, rechunk the padded array because apply_ufunc
            # requires core dimensions not to be chunked:
            if a.sizes[dim] < b.sizes[dim]:
                a = a.pad({dim: (0, 1)}, constant_values=0)
                # TODO: Should pad or apply_ufunc handle correct chunking?
                a = a.chunk({dim: -1}) if is_duck_dask_array(a.data) else a
            else:
                b = b.pad({dim: (0, 1)}, constant_values=0)
                # TODO: Should pad or apply_ufunc handle correct chunking?
                b = b.chunk({dim: -1}) if is_duck_dask_array(b.data) else b
        else:
            raise ValueError(
                f"{dim!r} on {'a' if a.sizes[dim] == 1 else 'b'} is incompatible:"
                " dimensions without coordinates must have have a length of 2 or 3"
            )

    c = apply_ufunc(
        np.cross,
        a,
        b,
        input_core_dims=[[dim], [dim]],
        output_core_dims=[[dim] if a.sizes[dim] == 3 else []],
        dask="parallelized",
        output_dtypes=[np.result_type(a, b)],
    )
    c = c.transpose(*all_dims, missing_dims="ignore")

    return c
