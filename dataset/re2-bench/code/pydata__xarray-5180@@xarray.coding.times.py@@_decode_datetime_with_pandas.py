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

def _decode_datetime_with_pandas(flat_num_dates, units, calendar):
    if not _is_standard_calendar(calendar):
        raise OutOfBoundsDatetime(
            "Cannot decode times from a non-standard calendar, {!r}, using "
            "pandas.".format(calendar)
        )

    delta, ref_date = _unpack_netcdf_time_units(units)
    delta = _netcdf_to_numpy_timeunit(delta)
    try:
        ref_date = pd.Timestamp(ref_date)
    except ValueError:
        # ValueError is raised by pd.Timestamp for non-ISO timestamp
        # strings, in which case we fall back to using cftime
        raise OutOfBoundsDatetime

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "invalid value encountered", RuntimeWarning)
        pd.to_timedelta(flat_num_dates.min(), delta) + ref_date
        pd.to_timedelta(flat_num_dates.max(), delta) + ref_date

    # To avoid integer overflow when converting to nanosecond units for integer
    # dtypes smaller than np.int64 cast all integer-dtype arrays to np.int64
    # (GH 2002).
    if flat_num_dates.dtype.kind == "i":
        flat_num_dates = flat_num_dates.astype(np.int64)

    # Cast input ordinals to integers of nanoseconds because pd.to_timedelta
    # works much faster when dealing with integers (GH 1399).
    flat_num_dates_ns_int = (flat_num_dates * _NS_PER_TIME_DELTA[delta]).astype(
        np.int64
    )

    # Use pd.to_timedelta to safely cast integer values to timedeltas,
    # and add those to a Timestamp to safely produce a DatetimeIndex.  This
    # ensures that we do not encounter integer overflow at any point in the
    # process without raising OutOfBoundsDatetime.
    return (pd.to_timedelta(flat_num_dates_ns_int, "ns") + ref_date).values
