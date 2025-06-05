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

class QuarterOffset(BaseCFTimeOffset):
    def __mul__(self, other):
        if isinstance(other, float):
            return NotImplemented
        return type(self)(n=other * self.n, month=self.month)