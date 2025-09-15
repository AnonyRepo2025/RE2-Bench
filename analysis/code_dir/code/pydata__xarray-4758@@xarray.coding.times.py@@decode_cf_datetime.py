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

def decode_cf_datetime(num_dates, units, calendar=None, use_cftime=None):
    num_dates = np.asarray(num_dates)
    flat_num_dates = num_dates.ravel()
    if calendar is None:
        calendar = 'standard'
    if use_cftime is None:
        try:
            dates = _decode_datetime_with_pandas(flat_num_dates, units, calendar)
        except (KeyError, OutOfBoundsDatetime, OverflowError):
            dates = _decode_datetime_with_cftime(flat_num_dates.astype(float), units, calendar)
            if dates[np.nanargmin(num_dates)].year < 1678 or dates[np.nanargmax(num_dates)].year >= 2262:
                if calendar in _STANDARD_CALENDARS:
                    warnings.warn('Unable to decode time axis into full numpy.datetime64 objects, continuing using cftime.datetime objects instead, reason: dates out of range', SerializationWarning, stacklevel=3)
            elif calendar in _STANDARD_CALENDARS:
                dates = cftime_to_nptime(dates)
    elif use_cftime:
        dates = _decode_datetime_with_cftime(flat_num_dates, units, calendar)
    else:
        dates = _decode_datetime_with_pandas(flat_num_dates, units, calendar)
    return dates.reshape(num_dates.shape)