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



def interp_na(
    self,
    dim: Hashable = None,
    use_coordinate: Union[bool, str] = True,
    method: str = "linear",
    limit: int = None,
    max_gap: Union[int, float, str, pd.Timedelta, np.timedelta64] = None,
    **kwargs,
):
    """Interpolate values according to different methods.
    """
    if dim is None:
        raise NotImplementedError("dim is a required argument")

    if limit is not None:
        valids = _get_valid_fill_mask(self, dim, limit)

    if max_gap is not None:
        max_type = type(max_gap).__name__
        if not is_scalar(max_gap):
            raise ValueError("max_gap must be a scalar.")

        if (
            dim in self.indexes
            and isinstance(self.indexes[dim], pd.DatetimeIndex)
            and use_coordinate
        ):
            if not isinstance(max_gap, (np.timedelta64, pd.Timedelta, str)):
                raise TypeError(
                    f"Underlying index is DatetimeIndex. Expected max_gap of type str, pandas.Timedelta or numpy.timedelta64 but received {max_type}"
                )

            if isinstance(max_gap, str):
                try:
                    max_gap = pd.to_timedelta(max_gap)
                except ValueError:
                    raise ValueError(
                        f"Could not convert {max_gap!r} to timedelta64 using pandas.to_timedelta"
                    )

            if isinstance(max_gap, pd.Timedelta):
                max_gap = np.timedelta64(max_gap.value, "ns")

            max_gap = np.timedelta64(max_gap, "ns").astype(np.float64)

        if not use_coordinate:
            if not isinstance(max_gap, (Number, np.number)):
                raise TypeError(
                    f"Expected integer or floating point max_gap since use_coordinate=False. Received {max_type}."
                )

    # method
    index = get_clean_interp_index(self, dim, use_coordinate=use_coordinate)
    interp_class, kwargs = _get_interpolator(method, **kwargs)
    interpolator = partial(func_interpolate_na, interp_class, **kwargs)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "overflow", RuntimeWarning)
        warnings.filterwarnings("ignore", "invalid value", RuntimeWarning)
        arr = apply_ufunc(
            interpolator,
            index,
            self,
            input_core_dims=[[dim], [dim]],
            output_core_dims=[[dim]],
            output_dtypes=[self.dtype],
            dask="parallelized",
            vectorize=True,
            keep_attrs=True,
        ).transpose(*self.dims)

    if limit is not None:
        arr = arr.where(valids)

    if max_gap is not None:
        if dim not in self.coords:
            raise NotImplementedError(
                "max_gap not implemented for unlabeled coordinates yet."
            )
        nan_block_lengths = _get_nan_block_lengths(self, dim, index)
        arr = arr.where(nan_block_lengths <= max_gap)

    return arr
