import re
import warnings
from datetime import datetime, timedelta
from distutils.version import LooseVersion
from functools import partial
import numpy as np
import pandas as pd
from pandas.errors import OutOfBoundsDatetime
from ..core import indexing
from ..core.common import contains_cftime_datetimes
from ..core.formatting import first_n_items, format_timestamp, last_item
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
import cftime

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

def decode_cf_datetime(num_dates, units, calendar=None, use_cftime=None):
    """Given an array of numeric dates in netCDF format, convert it into a
    numpy array of date time objects.

    For standard (Gregorian) calendars, this function uses vectorized
    operations, which makes it much faster than cftime.num2date. In such a
    case, the returned array will be of type np.datetime64.

    Note that time unit in `units` must not be smaller than microseconds and
    not larger than days.

    See Also
    --------
    cftime.num2date
    """
    num_dates = np.asarray(num_dates)
    flat_num_dates = num_dates.ravel()
    if calendar is None:
        calendar = "standard"

    if use_cftime is None:
        try:
            dates = _decode_datetime_with_pandas(flat_num_dates, units, calendar)
        except (KeyError, OutOfBoundsDatetime, OverflowError):
            dates = _decode_datetime_with_cftime(
                flat_num_dates.astype(float), units, calendar
            )

            if (
                dates[np.nanargmin(num_dates)].year < 1678
                or dates[np.nanargmax(num_dates)].year >= 2262
            ):
                if _is_standard_calendar(calendar):
                    warnings.warn(
                        "Unable to decode time axis into full "
                        "numpy.datetime64 objects, continuing using "
                        "cftime.datetime objects instead, reason: dates out "
                        "of range",
                        SerializationWarning,
                        stacklevel=3,
                    )
            else:
                if _is_standard_calendar(calendar):
                    dates = cftime_to_nptime(dates)
    elif use_cftime:
        dates = _decode_datetime_with_cftime(flat_num_dates, units, calendar)
    else:
        dates = _decode_datetime_with_pandas(flat_num_dates, units, calendar)

    return dates.reshape(num_dates.shape)
