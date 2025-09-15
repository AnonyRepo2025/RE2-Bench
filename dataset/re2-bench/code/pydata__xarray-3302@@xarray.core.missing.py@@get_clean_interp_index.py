import warnings
from functools import partial
from numbers import Number
from typing import Any, Callable, Dict, Hashable, Sequence, Union
import numpy as np
import pandas as pd
from . import utils
from .common import _contains_datetime_like_objects, ones_like
from .computation import apply_ufunc
from .duck_array_ops import dask_array_type
from .utils import OrderedSet, is_scalar
from .variable import Variable, broadcast_variables
import bottleneck as bn
import bottleneck as bn
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
from scipy import interpolate
from scipy import interpolate
import dask.array as da



def get_clean_interp_index(arr, dim: Hashable, use_coordinate: Union[str, bool] = True):
    """get index to use for x values in interpolation.

    If use_coordinate is True, the coordinate that shares the name of the
    dimension along which interpolation is being performed will be used as the
    x values.

    If use_coordinate is False, the x values are set as an equally spaced
    sequence.
    """
    if use_coordinate:
        if use_coordinate is True:
            index = arr.get_index(dim)
        else:
            index = arr.coords[use_coordinate]
            if index.ndim != 1:
                raise ValueError(
                    f"Coordinates used for interpolation must be 1D, "
                    f"{use_coordinate} is {index.ndim}D."
                )
            index = index.to_index()

        # TODO: index.name is None for multiindexes
        # set name for nice error messages below
        if isinstance(index, pd.MultiIndex):
            index.name = dim

        if not index.is_monotonic:
            raise ValueError(f"Index {index.name!r} must be monotonically increasing")

        if not index.is_unique:
            raise ValueError(f"Index {index.name!r} has duplicate values")

        # raise if index cannot be cast to a float (e.g. MultiIndex)
        try:
            index = index.values.astype(np.float64)
        except (TypeError, ValueError):
            # pandas raises a TypeError
            # xarray/numpy raise a ValueError
            raise TypeError(
                f"Index {index.name!r} must be castable to float64 to support "
                f"interpolation, got {type(index).__name__}."
            )

    else:
        axis = arr.get_axis_num(dim)
        index = np.arange(arr.shape[axis], dtype=np.float64)

    return index
