import datetime
import functools
import logging
import math
from dateutil.rrule import (rrule, MO, TU, WE, TH, FR, SA, SU, YEARLY,
                            MONTHLY, WEEKLY, DAILY, HOURLY, MINUTELY,
                            SECONDLY)
from dateutil.relativedelta import relativedelta
import dateutil.parser
import dateutil.tz
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook, ticker, units

__all__ = ('datestr2num', 'date2num', 'num2date', 'num2timedelta', 'drange',
           'epoch2num', 'num2epoch', 'set_epoch', 'get_epoch', 'DateFormatter',
           'ConciseDateFormatter', 'AutoDateFormatter',
           'DateLocator', 'RRuleLocator', 'AutoDateLocator', 'YearLocator',
           'MonthLocator', 'WeekdayLocator',
           'DayLocator', 'HourLocator', 'MinuteLocator',
           'SecondLocator', 'MicrosecondLocator',
           'rrule', 'MO', 'TU', 'WE', 'TH', 'FR', 'SA', 'SU',
           'YEARLY', 'MONTHLY', 'WEEKLY', 'DAILY',
           'HOURLY', 'MINUTELY', 'SECONDLY', 'MICROSECONDLY', 'relativedelta',
           'DateConverter', 'ConciseDateConverter', 'rrulewrapper')
_log = logging.getLogger(__name__)
UTC = datetime.timezone.utc
EPOCH_OFFSET = float(datetime.datetime(1970, 1, 1).toordinal())
JULIAN_OFFSET = 1721424.5  # Julian date at 0000-12-31
MICROSECONDLY = SECONDLY + 1
HOURS_PER_DAY = 24.
MIN_PER_HOUR = 60.
SEC_PER_MIN = 60.
MONTHS_PER_YEAR = 12.
DAYS_PER_WEEK = 7.
DAYS_PER_MONTH = 30.
DAYS_PER_YEAR = 365.0
MINUTES_PER_DAY = MIN_PER_HOUR * HOURS_PER_DAY
SEC_PER_HOUR = SEC_PER_MIN * MIN_PER_HOUR
SEC_PER_DAY = SEC_PER_HOUR * HOURS_PER_DAY
SEC_PER_WEEK = SEC_PER_DAY * DAYS_PER_WEEK
MUSECONDS_PER_DAY = 1e6 * SEC_PER_DAY
MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY = (
    MO, TU, WE, TH, FR, SA, SU)
WEEKDAYS = (MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY)
_epoch = None
_from_ordinalf_np_vectorized = np.vectorize(_from_ordinalf, otypes="O")
_dateutil_parser_parse_np_vectorized = np.vectorize(dateutil.parser.parse)
_ordinalf_to_timedelta_np_vectorized = np.vectorize(
    lambda x: datetime.timedelta(days=x), otypes="O")
units.registry[np.datetime64] = \
    units.registry[datetime.date] = \
    units.registry[datetime.datetime] = \
    _SwitchableDateConverter()

class DateLocator(Locator):
    hms0d = {'byhour': 0, 'byminute': 0, 'bysecond': 0}
    def nonsingular(self, vmin, vmax):
        """
        Given the proposed upper and lower extent, adjust the range
        if it is too close to being singular (i.e. a range of ~0).
        """
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            # Except if there is no data, then use 1970 as default.
            return (date2num(datetime.date(1970, 1, 1)),
                    date2num(datetime.date(1970, 1, 2)))
        if vmax < vmin:
            vmin, vmax = vmax, vmin
        unit = self._get_unit()
        interval = self._get_interval()
        if abs(vmax - vmin) < 1e-6:
            vmin -= 2 * unit * interval
            vmax += 2 * unit * interval
        return vmin, vmax