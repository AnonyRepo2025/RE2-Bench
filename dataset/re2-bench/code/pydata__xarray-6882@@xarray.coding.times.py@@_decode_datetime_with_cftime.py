from __future__ import annotations
import re
import warnings
from datetime import datetime, timedelta
from functools import partial
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
from pandas.errors import OutOfBoundsDatetime, OutOfBoundsTimedelta
from ..core import indexing
from ..core.common import contains_cftime_datetimes, is_np_datetime_like
from ..core.formatting import first_n_items, format_timestamp, last_item
from ..core.pycompat import is_duck_dask_array
from ..core.variable import Variable
from .variables import (
    SerializationWarning,
    VariableCoder,
    lazy_elemwise_func,
    pop_to,
    safe_setitem,
    unpack_for_decoding,
    unpack_for_encoding,
)
import cftime
from ..core.types import CFCalendar

_STANDARD_CALENDARS = {"standard", "gregorian", "proleptic_gregorian"}
_NS_PER_TIME_DELTA = {
    "ns": 1,
    "us": int(1e3),
    "ms": int(1e6),
    "s": int(1e9),
    "m": int(1e9) * 60,
    "h": int(1e9) * 60 * 60,
    "D": int(1e9) * 60 * 60 * 24,
}
_US_PER_TIME_DELTA = {
    "microseconds": 1,
    "milliseconds": 1_000,
    "seconds": 1_000_000,
    "minutes": 60 * 1_000_000,
    "hours": 60 * 60 * 1_000_000,
    "days": 24 * 60 * 60 * 1_000_000,
}
_NETCDF_TIME_UNITS_CFTIME = [
    "days",
    "hours",
    "minutes",
    "seconds",
    "milliseconds",
    "microseconds",
]
_NETCDF_TIME_UNITS_NUMPY = _NETCDF_TIME_UNITS_CFTIME + ["nanoseconds"]
TIME_UNITS = frozenset(
    [
        "days",
        "hours",
        "minutes",
        "seconds",
        "milliseconds",
        "microseconds",
        "nanoseconds",
    ]
)

def _decode_datetime_with_cftime(num_dates, units, calendar):
    if cftime is None:
        raise ModuleNotFoundError("No module named 'cftime'")
    if num_dates.size > 0:
        return np.asarray(
            cftime.num2date(num_dates, units, calendar, only_use_cftime_datetimes=True)
        )
    else:
        return np.array([], dtype=object)
