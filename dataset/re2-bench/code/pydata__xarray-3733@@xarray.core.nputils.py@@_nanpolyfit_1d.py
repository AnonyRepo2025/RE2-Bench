import warnings
import numpy as np
import pandas as pd
from numpy.core.multiarray import normalize_axis_index
import bottleneck as bn

nanmin = _create_bottleneck_method("nanmin")
nanmax = _create_bottleneck_method("nanmax")
nanmean = _create_bottleneck_method("nanmean")
nanmedian = _create_bottleneck_method("nanmedian")
nanvar = _create_bottleneck_method("nanvar")
nanstd = _create_bottleneck_method("nanstd")
nanprod = _create_bottleneck_method("nanprod")
nancumsum = _create_bottleneck_method("nancumsum")
nancumprod = _create_bottleneck_method("nancumprod")
nanargmin = _create_bottleneck_method("nanargmin")
nanargmax = _create_bottleneck_method("nanargmax")

def _nanpolyfit_1d(arr, x, rcond=None):
    out = np.full((x.shape[1] + 1,), np.nan)
    mask = np.isnan(arr)
    if not np.all(mask):
        out[:-1], out[-1], _, _ = np.linalg.lstsq(x[~mask, :], arr[~mask], rcond=rcond)
    return out
