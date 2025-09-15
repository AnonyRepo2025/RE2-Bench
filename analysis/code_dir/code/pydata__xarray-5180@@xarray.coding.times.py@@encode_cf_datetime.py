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
from .variables import SerializationWarning, VariableCoder, lazy_elemwise_func, pop_to, safe_setitem, unpack_for_decoding, unpack_for_encoding
import cftime
import cftime
_STANDARD_CALENDARS = {'standard', 'gregorian', 'proleptic_gregorian'}
_NS_PER_TIME_DELTA = {'ns': 1, 'us': int(1000.0), 'ms': int(1000000.0), 's': int(1000000000.0), 'm': int(1000000000.0) * 60, 'h': int(1000000000.0) * 60 * 60, 'D': int(1000000000.0) * 60 * 60 * 24}
_US_PER_TIME_DELTA = {'microseconds': 1, 'milliseconds': 1000, 'seconds': 1000000, 'minutes': 60 * 1000000, 'hours': 60 * 60 * 1000000, 'days': 24 * 60 * 60 * 1000000}
_NETCDF_TIME_UNITS_CFTIME = ['days', 'hours', 'minutes', 'seconds', 'milliseconds', 'microseconds']
_NETCDF_TIME_UNITS_NUMPY = _NETCDF_TIME_UNITS_CFTIME + ['nanoseconds']
TIME_UNITS = frozenset(['days', 'hours', 'minutes', 'seconds', 'milliseconds', 'microseconds', 'nanoseconds'])

def encode_cf_datetime(dates, units=None, calendar=None):
    dates = np.asarray(dates)
    if units is None:
        units = infer_datetime_units(dates)
    else:
        units = _cleanup_netcdf_time_units(units)
    if calendar is None:
        calendar = infer_calendar_name(dates)
    delta, ref_date = _unpack_netcdf_time_units(units)
    try:
        if not _is_standard_calendar(calendar) or dates.dtype.kind == 'O':
            raise OutOfBoundsDatetime
        assert dates.dtype == 'datetime64[ns]'
        delta_units = _netcdf_to_numpy_timeunit(delta)
        time_delta = np.timedelta64(1, delta_units).astype('timedelta64[ns]')
        ref_date = pd.Timestamp(ref_date)
        if ref_date.tz is not None:
            ref_date = ref_date.tz_convert(None)
        dates_as_index = pd.DatetimeIndex(dates.ravel())
        time_deltas = dates_as_index - ref_date
        if np.all(time_deltas % time_delta == np.timedelta64(0, 'ns')):
            num = time_deltas // time_delta
        else:
            num = time_deltas / time_delta
        num = num.values.reshape(dates.shape)
    except (OutOfBoundsDatetime, OverflowError):
        num = _encode_datetime_with_cftime(dates, units, calendar)
    num = cast_to_int_if_safe(num)
    return (num, units, calendar)