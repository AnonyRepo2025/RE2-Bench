import os
import re
import inspect
import warnings
import colorsys
from contextlib import contextmanager
from urllib.request import urlopen, urlretrieve
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.colors import to_rgb
import matplotlib.pyplot as plt
from matplotlib.cbook import normalize_kwargs
from .external.version import Version
from .external.appdirs import user_cache_dir
from seaborn.axisgrid import Grid  # Avoid circular import
from statistics import NormalDist
from scipy.stats import norm

__all__ = ["desaturate", "saturate", "set_hls_values", "move_legend",
           "despine", "get_dataset_names", "get_data_home", "load_dataset"]

def locator_to_legend_entries(locator, limits, dtype):
    """Return levels and formatted levels for brief numeric legends."""
    raw_levels = locator.tick_values(*limits).astype(dtype)

    # The locator can return ticks outside the limits, clip them here
    raw_levels = [l for l in raw_levels if l >= limits[0] and l <= limits[1]]

    class dummy_axis:
        def get_view_interval(self):
            return limits

    if isinstance(locator, mpl.ticker.LogLocator):
        formatter = mpl.ticker.LogFormatter()
    else:
        formatter = mpl.ticker.ScalarFormatter()
        # Avoid having an offset/scientific notation which we don't currently
        # have any way of representing in the legend
        formatter.set_useOffset(False)
        formatter.set_scientific(False)
    formatter.axis = dummy_axis()

    # TODO: The following two lines should be replaced
    # once pinned matplotlib>=3.1.0 with:
    # formatted_levels = formatter.format_ticks(raw_levels)
    formatter.set_locs(raw_levels)
    formatted_levels = [formatter(x) for x in raw_levels]

    return raw_levels, formatted_levels
