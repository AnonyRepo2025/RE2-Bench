def _cleanup_netcdf_time_units(units):
    delta, ref_date = _unpack_netcdf_time_units(units)
    try:
        units = '{} since {}'.format(delta, format_timestamp(ref_date))
    except OutOfBoundsDatetime:
        pass
    return units

def _unpack_netcdf_time_units(units):
    matches = re.match('(.+) since (.+)', units)
    if not matches:
        raise ValueError(f'invalid time units: {units}')
    delta_units, ref_date = [s.strip() for s in matches.groups()]
    ref_date = _ensure_padded_year(ref_date)
    return (delta_units, ref_date)

def _ensure_padded_year(ref_date):
    matches_year = re.match('.*\\d{4}.*', ref_date)
    if matches_year:
        return ref_date
    matches_start_digits = re.match('(\\d+)(.*)', ref_date)
    ref_year, everything_else = [s for s in matches_start_digits.groups()]
    ref_date_padded = '{:04d}{}'.format(int(ref_year), everything_else)
    warning_msg = f'Ambiguous reference date string: {ref_date}. The first value is assumed to be the year hence will be padded with zeros to remove the ambiguity (the padded reference date string is: {ref_date_padded}). To remove this message, remove the ambiguity by padding your reference date strings with zeros.'
    warnings.warn(warning_msg, SerializationWarning)
    return ref_date_padded

def format_timestamp(t):
    try:
        datetime_str = str(pd.Timestamp(t))
    except OutOfBoundsDatetime:
        datetime_str = str(t)
    try:
        date_str, time_str = datetime_str.split()
    except ValueError:
        return datetime_str
    else:
        if time_str == '00:00:00':
            return date_str
        else:
            return f'{date_str}T{time_str}'

def _is_standard_calendar(calendar):
    return calendar.lower() in _STANDARD_CALENDARS

def _netcdf_to_numpy_timeunit(units):
    units = units.lower()
    if not units.endswith('s'):
        units = '%ss' % units
    return {'nanoseconds': 'ns', 'microseconds': 'us', 'milliseconds': 'ms', 'seconds': 's', 'minutes': 'm', 'hours': 'h', 'days': 'D'}[units]

def _encode_datetime_with_cftime(dates, units, calendar):
    import cftime
    if np.issubdtype(dates.dtype, np.datetime64):
        dates = dates.astype('M8[us]').astype(datetime)

    def encode_datetime(d):
        return np.nan if d is None else cftime.date2num(d, units, calendar)
    return np.array([encode_datetime(d) for d in dates.ravel()]).reshape(dates.shape)

def encode_datetime(d):
    return np.nan if d is None else cftime.date2num(d, units, calendar)

def cast_to_int_if_safe(num):
    int_num = np.array(num, dtype=np.int64)
    if (num == int_num).all():
        num = int_num
    return num

def infer_datetime_units(dates):
    dates = np.asarray(dates).ravel()
    if np.asarray(dates).dtype == 'datetime64[ns]':
        dates = to_datetime_unboxed(dates)
        dates = dates[pd.notnull(dates)]
        reference_date = dates[0] if len(dates) > 0 else '1970-01-01'
        reference_date = pd.Timestamp(reference_date)
    else:
        reference_date = dates[0] if len(dates) > 0 else '1970-01-01'
        reference_date = format_cftime_datetime(reference_date)
    unique_timedeltas = np.unique(np.diff(dates))
    units = _infer_time_units_from_diff(unique_timedeltas)
    return f'{units} since {reference_date}'

def to_datetime_unboxed(value, **kwargs):
    if LooseVersion(pd.__version__) < '0.25.0':
        result = pd.to_datetime(value, **kwargs, box=False)
    else:
        result = pd.to_datetime(value, **kwargs).to_numpy()
    assert result.dtype == 'datetime64[ns]'
    return result

def _infer_time_units_from_diff(unique_timedeltas):
    if unique_timedeltas.dtype == np.dtype('O'):
        time_units = _NETCDF_TIME_UNITS_CFTIME
        unit_timedelta = _unit_timedelta_cftime
        zero_timedelta = timedelta(microseconds=0)
        timedeltas = unique_timedeltas
    else:
        time_units = _NETCDF_TIME_UNITS_NUMPY
        unit_timedelta = _unit_timedelta_numpy
        zero_timedelta = np.timedelta64(0, 'ns')
        timedeltas = pd.TimedeltaIndex(unique_timedeltas)
    for time_unit in time_units:
        if np.all(timedeltas % unit_timedelta(time_unit) == zero_timedelta):
            return time_unit
    return 'seconds'

def _unit_timedelta_numpy(units):
    numpy_units = _netcdf_to_numpy_timeunit(units)
    return np.timedelta64(_NS_PER_TIME_DELTA[numpy_units], 'ns')

def infer_calendar_name(dates):
    if np.asarray(dates).dtype == 'datetime64[ns]':
        return 'proleptic_gregorian'
    else:
        return np.asarray(dates).ravel()[0].calendar

def format_cftime_datetime(date):
    return '{:04d}-{:02d}-{:02d} {:02d}:{:02d}:{:02d}.{:06d}'.format(date.year, date.month, date.day, date.hour, date.minute, date.second, date.microsecond)



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