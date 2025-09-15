from __future__ import annotations
import re
from datetime import datetime, timedelta
from functools import partial
from typing import ClassVar
import numpy as np
import pandas as pd
from ..core.common import _contains_datetime_like_objects, is_np_datetime_like
from ..core.pdcompat import count_not_none
from .cftimeindex import CFTimeIndex, _parse_iso8601_with_reso
from .times import (
    _is_standard_calendar,
    _should_cftime_be_used,
    convert_time_or_go_back,
    format_cftime_datetime,
)
import cftime
from .times import _is_standard_calendar
from ..core.dataarray import DataArray
from .frequencies import infer_freq

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
_PATTERN = rf"^((?P<multiple>\d+)|())(?P<freq>({_FREQUENCY_CONDITION}))$"
CFTIME_TICKS = (Day, Hour, Minute, Second)

class QuarterOffset(BaseCFTimeOffset):
    def __mul__(self, other):
        if isinstance(other, float):
            return NotImplemented
        return type(self)(n=other * self.n, month=self.month)