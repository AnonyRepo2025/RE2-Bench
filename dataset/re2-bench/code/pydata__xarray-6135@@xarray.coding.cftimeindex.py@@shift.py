from __future__ import annotations
import re
import warnings
from datetime import timedelta
import numpy as np
import pandas as pd
from packaging.version import Version
from xarray.core.utils import is_scalar
from ..core.common import _contains_cftime_datetimes
from ..core.options import OPTIONS
from .times import _STANDARD_CALENDARS, cftime_to_nptime, infer_calendar_name
import cftime
from ..core.resample_cftime import exact_cftime_datetime_difference
from .times import infer_calendar_name
from .frequencies import infer_freq
from .cftime_offsets import CFTIME_TICKS, to_offset
from .cftime_offsets import to_offset

CFTIME_REPR_LENGTH = 19
ITEMS_IN_REPR_MAX_ELSE_ELLIPSIS = 100
REPR_ELLIPSIS_SHOW_ITEMS_FRONT_END = 10
_BASIC_PATTERN = build_pattern(date_sep="", time_sep="")
_EXTENDED_PATTERN = build_pattern()
_CFTIME_PATTERN = build_pattern(datetime_sep=" ")
_PATTERNS = [_BASIC_PATTERN, _EXTENDED_PATTERN, _CFTIME_PATTERN]

class CFTimeIndex(Index):
    year = _field_accessor("year", "The year of the datetime")
    month = _field_accessor("month", "The month of the datetime")
    day = _field_accessor("day", "The days of the datetime")
    hour = _field_accessor("hour", "The hours of the datetime")
    minute = _field_accessor("minute", "The minutes of the datetime")
    second = _field_accessor("second", "The seconds of the datetime")
    microsecond = _field_accessor("microsecond", "The microseconds of the datetime")
    dayofyear = _field_accessor(
        "dayofyr", "The ordinal day of year of the datetime", "1.0.2.1"
    )
    dayofweek = _field_accessor("dayofwk", "The day of week of the datetime", "1.0.2.1")
    days_in_month = _field_accessor(
        "daysinmonth", "The number of days in the month of the datetime", "1.1.0.0"
    )
    date_type = property(get_date_type)
    def shift(self, n: int | float, freq: str | timedelta):
        """Shift the CFTimeIndex a multiple of the given frequency.

        See the documentation for :py:func:`~xarray.cftime_range` for a
        complete listing of valid frequency strings.

        Parameters
        ----------
        n : int, float if freq of days or below
            Periods to shift by
        freq : str or datetime.timedelta
            A frequency string or datetime.timedelta object to shift by

        Returns
        -------
        CFTimeIndex

        See Also
        --------
        pandas.DatetimeIndex.shift

        Examples
        --------
        >>> index = xr.cftime_range("2000", periods=1, freq="M")
        >>> index
        CFTimeIndex([2000-01-31 00:00:00],
                    dtype='object', length=1, calendar='standard', freq=None)
        >>> index.shift(1, "M")
        CFTimeIndex([2000-02-29 00:00:00],
                    dtype='object', length=1, calendar='standard', freq=None)
        >>> index.shift(1.5, "D")
        CFTimeIndex([2000-02-01 12:00:00],
                    dtype='object', length=1, calendar='standard', freq=None)
        """
        if isinstance(freq, timedelta):
            return self + n * freq
        elif isinstance(freq, str):
            from .cftime_offsets import to_offset

            return self + n * to_offset(freq)
        else:
            raise TypeError(
                "'freq' must be of type "
                "str or datetime.timedelta, got {}.".format(freq)
            )