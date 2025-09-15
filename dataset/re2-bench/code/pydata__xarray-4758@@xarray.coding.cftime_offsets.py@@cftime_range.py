import re
from datetime import timedelta
from distutils.version import LooseVersion
from functools import partial
from typing import ClassVar, Optional
import numpy as np
from ..core.pdcompat import count_not_none
from .cftimeindex import CFTimeIndex, _parse_iso8601_with_reso
from .times import format_cftime_datetime
import cftime
import cftime
import cftime
import cftime
import cftime
import cftime
import cftime

_MONTH_ABBREVIATIONS = {
    1: "JAN",
    2: "FEB",
    3: "MAR",
    4: "APR",
    5: "MAY",
    6: "JUN",
    7: "JUL",
    8: "AUG",
    9: "SEP",
    10: "OCT",
    11: "NOV",
    12: "DEC",
}
_FREQUENCIES = {
    "A": YearEnd,
    "AS": YearBegin,
    "Y": YearEnd,
    "YS": YearBegin,
    "Q": partial(QuarterEnd, month=12),
    "QS": partial(QuarterBegin, month=1),
    "M": MonthEnd,
    "MS": MonthBegin,
    "D": Day,
    "H": Hour,
    "T": Minute,
    "min": Minute,
    "S": Second,
    "L": Millisecond,
    "ms": Millisecond,
    "U": Microsecond,
    "us": Microsecond,
    "AS-JAN": partial(YearBegin, month=1),
    "AS-FEB": partial(YearBegin, month=2),
    "AS-MAR": partial(YearBegin, month=3),
    "AS-APR": partial(YearBegin, month=4),
    "AS-MAY": partial(YearBegin, month=5),
    "AS-JUN": partial(YearBegin, month=6),
    "AS-JUL": partial(YearBegin, month=7),
    "AS-AUG": partial(YearBegin, month=8),
    "AS-SEP": partial(YearBegin, month=9),
    "AS-OCT": partial(YearBegin, month=10),
    "AS-NOV": partial(YearBegin, month=11),
    "AS-DEC": partial(YearBegin, month=12),
    "A-JAN": partial(YearEnd, month=1),
    "A-FEB": partial(YearEnd, month=2),
    "A-MAR": partial(YearEnd, month=3),
    "A-APR": partial(YearEnd, month=4),
    "A-MAY": partial(YearEnd, month=5),
    "A-JUN": partial(YearEnd, month=6),
    "A-JUL": partial(YearEnd, month=7),
    "A-AUG": partial(YearEnd, month=8),
    "A-SEP": partial(YearEnd, month=9),
    "A-OCT": partial(YearEnd, month=10),
    "A-NOV": partial(YearEnd, month=11),
    "A-DEC": partial(YearEnd, month=12),
    "QS-JAN": partial(QuarterBegin, month=1),
    "QS-FEB": partial(QuarterBegin, month=2),
    "QS-MAR": partial(QuarterBegin, month=3),
    "QS-APR": partial(QuarterBegin, month=4),
    "QS-MAY": partial(QuarterBegin, month=5),
    "QS-JUN": partial(QuarterBegin, month=6),
    "QS-JUL": partial(QuarterBegin, month=7),
    "QS-AUG": partial(QuarterBegin, month=8),
    "QS-SEP": partial(QuarterBegin, month=9),
    "QS-OCT": partial(QuarterBegin, month=10),
    "QS-NOV": partial(QuarterBegin, month=11),
    "QS-DEC": partial(QuarterBegin, month=12),
    "Q-JAN": partial(QuarterEnd, month=1),
    "Q-FEB": partial(QuarterEnd, month=2),
    "Q-MAR": partial(QuarterEnd, month=3),
    "Q-APR": partial(QuarterEnd, month=4),
    "Q-MAY": partial(QuarterEnd, month=5),
    "Q-JUN": partial(QuarterEnd, month=6),
    "Q-JUL": partial(QuarterEnd, month=7),
    "Q-AUG": partial(QuarterEnd, month=8),
    "Q-SEP": partial(QuarterEnd, month=9),
    "Q-OCT": partial(QuarterEnd, month=10),
    "Q-NOV": partial(QuarterEnd, month=11),
    "Q-DEC": partial(QuarterEnd, month=12),
}
_FREQUENCY_CONDITION = "|".join(_FREQUENCIES.keys())
_PATTERN = fr"^((?P<multiple>\d+)|())(?P<freq>({_FREQUENCY_CONDITION}))$"
CFTIME_TICKS = (Day, Hour, Minute, Second)

def cftime_range(
    start=None,
    end=None,
    periods=None,
    freq="D",
    normalize=False,
    name=None,
    closed=None,
    calendar="standard",
):
    """Return a fixed frequency CFTimeIndex.

    Parameters
    ----------
    start : str or cftime.datetime, optional
        Left bound for generating dates.
    end : str or cftime.datetime, optional
        Right bound for generating dates.
    periods : int, optional
        Number of periods to generate.
    freq : str or None, default: "D"
        Frequency strings can have multiples, e.g. "5H".
    normalize : bool, default: False
        Normalize start/end dates to midnight before generating date range.
    name : str, default: None
        Name of the resulting index
    closed : {"left", "right"} or None, default: None
        Make the interval closed with respect to the given frequency to the
        "left", "right", or both sides (None).
    calendar : str, default: "standard"
        Calendar type for the datetimes.

    Returns
    -------
    CFTimeIndex

    Notes
    -----
    This function is an analog of ``pandas.date_range`` for use in generating
    sequences of ``cftime.datetime`` objects.  It supports most of the
    features of ``pandas.date_range`` (e.g. specifying how the index is
    ``closed`` on either side, or whether or not to ``normalize`` the start and
    end bounds); however, there are some notable exceptions:

    - You cannot specify a ``tz`` (time zone) argument.
    - Start or end dates specified as partial-datetime strings must use the
      `ISO-8601 format <https://en.wikipedia.org/wiki/ISO_8601>`_.
    - It supports many, but not all, frequencies supported by
      ``pandas.date_range``.  For example it does not currently support any of
      the business-related or semi-monthly frequencies.
    - Compound sub-monthly frequencies are not supported, e.g. '1H1min', as
      these can easily be written in terms of the finest common resolution,
      e.g. '61min'.

    Valid simple frequency strings for use with ``cftime``-calendars include
    any multiples of the following.

    +--------+--------------------------+
    | Alias  | Description              |
    +========+==========================+
    | A, Y   | Year-end frequency       |
    +--------+--------------------------+
    | AS, YS | Year-start frequency     |
    +--------+--------------------------+
    | Q      | Quarter-end frequency    |
    +--------+--------------------------+
    | QS     | Quarter-start frequency  |
    +--------+--------------------------+
    | M      | Month-end frequency      |
    +--------+--------------------------+
    | MS     | Month-start frequency    |
    +--------+--------------------------+
    | D      | Day frequency            |
    +--------+--------------------------+
    | H      | Hour frequency           |
    +--------+--------------------------+
    | T, min | Minute frequency         |
    +--------+--------------------------+
    | S      | Second frequency         |
    +--------+--------------------------+
    | L, ms  | Millisecond frequency    |
    +--------+--------------------------+
    | U, us  | Microsecond frequency    |
    +--------+--------------------------+

    Any multiples of the following anchored offsets are also supported.

    +----------+--------------------------------------------------------------------+
    | Alias    | Description                                                        |
    +==========+====================================================================+
    | A(S)-JAN | Annual frequency, anchored at the end (or beginning) of January    |
    +----------+--------------------------------------------------------------------+
    | A(S)-FEB | Annual frequency, anchored at the end (or beginning) of February   |
    +----------+--------------------------------------------------------------------+
    | A(S)-MAR | Annual frequency, anchored at the end (or beginning) of March      |
    +----------+--------------------------------------------------------------------+
    | A(S)-APR | Annual frequency, anchored at the end (or beginning) of April      |
    +----------+--------------------------------------------------------------------+
    | A(S)-MAY | Annual frequency, anchored at the end (or beginning) of May        |
    +----------+--------------------------------------------------------------------+
    | A(S)-JUN | Annual frequency, anchored at the end (or beginning) of June       |
    +----------+--------------------------------------------------------------------+
    | A(S)-JUL | Annual frequency, anchored at the end (or beginning) of July       |
    +----------+--------------------------------------------------------------------+
    | A(S)-AUG | Annual frequency, anchored at the end (or beginning) of August     |
    +----------+--------------------------------------------------------------------+
    | A(S)-SEP | Annual frequency, anchored at the end (or beginning) of September  |
    +----------+--------------------------------------------------------------------+
    | A(S)-OCT | Annual frequency, anchored at the end (or beginning) of October    |
    +----------+--------------------------------------------------------------------+
    | A(S)-NOV | Annual frequency, anchored at the end (or beginning) of November   |
    +----------+--------------------------------------------------------------------+
    | A(S)-DEC | Annual frequency, anchored at the end (or beginning) of December   |
    +----------+--------------------------------------------------------------------+
    | Q(S)-JAN | Quarter frequency, anchored at the end (or beginning) of January   |
    +----------+--------------------------------------------------------------------+
    | Q(S)-FEB | Quarter frequency, anchored at the end (or beginning) of February  |
    +----------+--------------------------------------------------------------------+
    | Q(S)-MAR | Quarter frequency, anchored at the end (or beginning) of March     |
    +----------+--------------------------------------------------------------------+
    | Q(S)-APR | Quarter frequency, anchored at the end (or beginning) of April     |
    +----------+--------------------------------------------------------------------+
    | Q(S)-MAY | Quarter frequency, anchored at the end (or beginning) of May       |
    +----------+--------------------------------------------------------------------+
    | Q(S)-JUN | Quarter frequency, anchored at the end (or beginning) of June      |
    +----------+--------------------------------------------------------------------+
    | Q(S)-JUL | Quarter frequency, anchored at the end (or beginning) of July      |
    +----------+--------------------------------------------------------------------+
    | Q(S)-AUG | Quarter frequency, anchored at the end (or beginning) of August    |
    +----------+--------------------------------------------------------------------+
    | Q(S)-SEP | Quarter frequency, anchored at the end (or beginning) of September |
    +----------+--------------------------------------------------------------------+
    | Q(S)-OCT | Quarter frequency, anchored at the end (or beginning) of October   |
    +----------+--------------------------------------------------------------------+
    | Q(S)-NOV | Quarter frequency, anchored at the end (or beginning) of November  |
    +----------+--------------------------------------------------------------------+
    | Q(S)-DEC | Quarter frequency, anchored at the end (or beginning) of December  |
    +----------+--------------------------------------------------------------------+

    Finally, the following calendar aliases are supported.

    +--------------------------------+---------------------------------------+
    | Alias                          | Date type                             |
    +================================+=======================================+
    | standard, gregorian            | ``cftime.DatetimeGregorian``          |
    +--------------------------------+---------------------------------------+
    | proleptic_gregorian            | ``cftime.DatetimeProlepticGregorian`` |
    +--------------------------------+---------------------------------------+
    | noleap, 365_day                | ``cftime.DatetimeNoLeap``             |
    +--------------------------------+---------------------------------------+
    | all_leap, 366_day              | ``cftime.DatetimeAllLeap``            |
    +--------------------------------+---------------------------------------+
    | 360_day                        | ``cftime.Datetime360Day``             |
    +--------------------------------+---------------------------------------+
    | julian                         | ``cftime.DatetimeJulian``             |
    +--------------------------------+---------------------------------------+

    Examples
    --------
    This function returns a ``CFTimeIndex``, populated with ``cftime.datetime``
    objects associated with the specified calendar type, e.g.

    >>> xr.cftime_range(start="2000", periods=6, freq="2MS", calendar="noleap")
    CFTimeIndex([2000-01-01 00:00:00, 2000-03-01 00:00:00, 2000-05-01 00:00:00,
                 2000-07-01 00:00:00, 2000-09-01 00:00:00, 2000-11-01 00:00:00],
                dtype='object', length=6, calendar='noleap', freq='2MS')

    As in the standard pandas function, three of the ``start``, ``end``,
    ``periods``, or ``freq`` arguments must be specified at a given time, with
    the other set to ``None``.  See the `pandas documentation
    <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.date_range.html>`_
    for more examples of the behavior of ``date_range`` with each of the
    parameters.

    See Also
    --------
    pandas.date_range
    """
    # Adapted from pandas.core.indexes.datetimes._generate_range.
    if count_not_none(start, end, periods, freq) != 3:
        raise ValueError(
            "Of the arguments 'start', 'end', 'periods', and 'freq', three "
            "must be specified at a time."
        )

    if start is not None:
        start = to_cftime_datetime(start, calendar)
        start = _maybe_normalize_date(start, normalize)
    if end is not None:
        end = to_cftime_datetime(end, calendar)
        end = _maybe_normalize_date(end, normalize)

    if freq is None:
        dates = _generate_linear_range(start, end, periods)
    else:
        offset = to_offset(freq)
        dates = np.array(list(_generate_range(start, end, periods, offset)))

    left_closed = False
    right_closed = False

    if closed is None:
        left_closed = True
        right_closed = True
    elif closed == "left":
        left_closed = True
    elif closed == "right":
        right_closed = True
    else:
        raise ValueError("Closed must be either 'left', 'right' or None")

    if not left_closed and len(dates) and start is not None and dates[0] == start:
        dates = dates[1:]
    if not right_closed and len(dates) and end is not None and dates[-1] == end:
        dates = dates[:-1]

    return CFTimeIndex(dates, name=name)
