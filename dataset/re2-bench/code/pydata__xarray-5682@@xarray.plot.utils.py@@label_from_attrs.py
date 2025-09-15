import itertools
import textwrap
import warnings
from datetime import datetime
from inspect import getfullargspec
from typing import Any, Iterable, Mapping, Tuple, Union
import numpy as np
import pandas as pd
from ..core.options import OPTIONS
from ..core.pycompat import DuckArrayModule
from ..core.utils import is_scalar
import nc_time_axis  # noqa: F401
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
import matplotlib as mpl
import matplotlib as mpl
import matplotlib.pyplot as plt
import cftime
from seaborn import color_palette

ROBUST_PERCENTILE = 2.0
_registered = False

def label_from_attrs(da, extra=""):
    """Makes informative labels if variable metadata (attrs) follows
    CF conventions."""

    if da.attrs.get("long_name"):
        name = da.attrs["long_name"]
    elif da.attrs.get("standard_name"):
        name = da.attrs["standard_name"]
    elif da.name is not None:
        name = da.name
    else:
        name = ""

    def _get_units_from_attrs(da):
        if da.attrs.get("units"):
            units = " [{}]".format(da.attrs["units"])
        elif da.attrs.get("unit"):
            units = " [{}]".format(da.attrs["unit"])
        else:
            units = ""
        return units

    pint_array_type = DuckArrayModule("pint").type
    if isinstance(da.data, pint_array_type):
        units = " [{}]".format(str(da.data.units))
    else:
        units = _get_units_from_attrs(da)

    # Treat `name` differently if it's a latex sequence
    if name.startswith("$") and (name.count("$") % 2 == 0):
        return "$\n$".join(
            textwrap.wrap(name + extra + units, 60, break_long_words=False)
        )
    else:
        return "\n".join(textwrap.wrap(name + extra + units, 30))
